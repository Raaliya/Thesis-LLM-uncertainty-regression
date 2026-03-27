import os
import re
import json
import time
import hashlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# =========================
# CONFIG
# =========================
CSV_PATH = "amazon_reviews.csv"
OUT_DIR = "outputs_exp02_amazon_ollama_2000"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
FEATURES_OUT = os.path.join(OUT_DIR, "amazon_llm_features_2000.csv")
BAD_ROWS_OUT = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")

# Ollama
OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

# speed + stability
MAX_CHARS = 1200
TEMPERATURE = 0.0
NUM_PREDICT = 300
N_ROWS = 2000

# heartbeat prints
PRINT_EVERY = 10

THEMES = [
    "product_quality",
    "durability_reliability",
    "ease_of_use",
    "value_price",
    "shipping_delivery",
    "packaging_condition",
    "customer_service",
    "authenticity_trust",
    "expectation_match",
    "recommendation_intent"
]

# =========================
# QUALITATIVE CODING PROMPT
# =========================
SYSTEM_INSTRUCTIONS = f"""
You are a qualitative researcher performing DEDUCTIVE qualitative coding using a fixed codebook.

CRITICAL RULES:
- Only code what is explicitly stated in the text. Do NOT guess.
- If a theme is not mentioned: present=0, polarity=0, intensity=0.
- Polarity: -1 negative, 0 neutral/mixed/unclear, +1 positive.
- Intensity: 0 none, 1 mild, 2 moderate, 3 strong.
- Output MUST be valid JSON ONLY. No extra text. No markdown. No explanations.

CODEBOOK THEMES:
1) product_quality
2) durability_reliability
3) ease_of_use
4) value_price
5) shipping_delivery
6) packaging_condition
7) customer_service
8) authenticity_trust
9) expectation_match
10) recommendation_intent

GLOBAL:
- overall_sentiment: -1 / 0 / +1
- confidence: 0.0–1.0

REQUIRED JSON SCHEMA (exact keys):
{{
  "overall_sentiment": -1|0|1,
  "confidence": 0.0-1.0,
  "themes": {{
    "{THEMES[0]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[1]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[2]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[3]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[4]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[5]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[6]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[7]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[8]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}},
    "{THEMES[9]}": {{"present":0|1,"polarity":-1|0|1,"intensity":0-3}}
  }}
}}
""".strip()


# =========================
# CACHE / IO HELPERS
# =========================
def load_cache(path: str) -> Dict[str, Any]:
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[obj["key"]] = obj["value"]
    return cache


def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# TEXT + HASH
# =========================
def make_text(summary, text) -> str:
    s = "" if pd.isna(summary) else str(summary)
    t = "" if pd.isna(text) else str(text)
    combined = f"SUMMARY: {s}\nTEXT: {t}".strip()
    return combined[:MAX_CHARS] + ("…" if len(combined) > MAX_CHARS else "")


def row_hash(summary, text) -> str:
    return hashlib.sha256((str(summary) + "||" + str(text)).encode("utf-8", errors="ignore")).hexdigest()


# =========================
# JSON ROBUSTNESS
# =========================
def extract_json_block(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    return m.group(0)


def clamp_int(v, lo, hi, default=0):
    try:
        v = int(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


def clamp_float(v, lo=0.0, hi=1.0, default=0.5):
    try:
        v = float(v)
        return max(lo, min(hi, v))
    except Exception:
        return default


def zero_schema() -> Dict[str, Any]:
    return {
        "overall_sentiment": 0,
        "confidence": 0.0,
        "themes": {t: {"present": 0, "polarity": 0, "intensity": 0} for t in THEMES}
    }


def normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "overall_sentiment": clamp_int(parsed.get("overall_sentiment", 0), -1, 1, 0),
        "confidence": clamp_float(parsed.get("confidence", 0.5), 0.0, 1.0, 0.5),
        "themes": {}
    }
    themes = parsed.get("themes", {})
    if not isinstance(themes, dict):
        themes = {}

    for theme in THEMES:
        t = themes.get(theme, {})
        if not isinstance(t, dict):
            t = {}
        present = clamp_int(t.get("present", 0), 0, 1, 0)
        polarity = clamp_int(t.get("polarity", 0), -1, 1, 0)
        intensity = clamp_int(t.get("intensity", 0), 0, 3, 0)

        if present == 0:
            polarity = 0
            intensity = 0

        out["themes"][theme] = {"present": present, "polarity": polarity, "intensity": intensity}

    return out


# =========================
# OLLAMA CALL WITH AUTO-FALLBACK
# =========================
USING_ENDPOINT = None  # "chat" or "generate"

def _post_chat(payload):
    return requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=180)

def _post_generate(payload):
    # /api/generate expects prompt not messages
    prompt = ""
    msgs = payload.get("messages", [])
    for m in msgs:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        prompt += f"{role}:\n{content}\n\n"

    gen_payload = {
        "model": payload["model"],
        "prompt": prompt,
        "stream": False,
        "options": payload.get("options", {})
    }
    return requests.post(f"{OLLAMA_BASE}/api/generate", json=gen_payload, timeout=180)

def ollama_chat(messages):
    """
    Calls Ollama robustly:
    - Try /api/chat
    - If 404 -> switch to /api/generate
    - Retry forever on connection drops/timeouts
    """
    global USING_ENDPOINT

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    }

    while True:
        try:
            # If we already decided to use generate
            if USING_ENDPOINT == "generate":
                r = _post_generate(payload)
                if r.status_code == 404:
                    # extremely rare: generate missing too
                    print("[WARN] /api/generate returned 404. Waiting 5s and retrying...")
                    time.sleep(5)
                    continue
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")

            # Default: try chat
            r = _post_chat(payload)

            if r.status_code == 404:
                if USING_ENDPOINT != "generate":
                    print("[INFO] /api/chat returned 404. Switching to /api/generate for the rest of the run.")
                USING_ENDPOINT = "generate"
                continue

            r.raise_for_status()
            data = r.json()
            return (data.get("message", {}) or {}).get("content", "")

        except requests.exceptions.ConnectionError:
            print("[WARN] Ollama connection refused. Waiting 5s and retrying...")
            time.sleep(5)
        except requests.exceptions.ReadTimeout:
            print("[WARN] Ollama read timeout. Retrying...")
        except requests.exceptions.HTTPError as e:
            print(f"[WARN] Ollama HTTP error: {e}. Retrying in 2s...")
            time.sleep(2)
        except Exception as e:
            print(f"[WARN] Unexpected Ollama error: {e}. Retrying in 2s...")
            time.sleep(2)


# =========================
# QUAL CODING (2-PASS REPAIR)
# =========================
def strict_code_review(review_text: str) -> Dict[str, Any]:
    user_msg = (
        "Return ONLY valid JSON for the schema given. "
        "No prose, no markdown, no backticks.\n\n"
        f"REVIEW:\n{review_text}"
    )

    raw1 = ollama_chat([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_msg}
    ])

    jb1 = extract_json_block(raw1)
    if jb1:
        try:
            return normalize(json.loads(jb1))
        except json.JSONDecodeError:
            pass

    repair_msg = (
        "Convert the following text into STRICT VALID JSON matching the required schema. "
        "Return JSON only. If something is missing, use zeros.\n\n"
        f"TEXT TO FIX:\n{raw1}"
    )

    raw2 = ollama_chat([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": repair_msg}
    ])

    jb2 = extract_json_block(raw2)
    if jb2:
        try:
            return normalize(json.loads(jb2))
        except json.JSONDecodeError:
            pass

    append_jsonl(BAD_ROWS_OUT, {"error": "json_decode_failed", "raw1": raw1, "raw2": raw2})
    return zero_schema()


def flatten(codes: Dict[str, Any]) -> Dict[str, float]:
    feats = {
        "llm_overall_sentiment": float(codes["overall_sentiment"]),
        "llm_confidence": float(codes["confidence"]),
    }
    for theme, v in codes["themes"].items():
        feats[f"llm_{theme}_present"] = float(v["present"])
        feats[f"llm_{theme}_polarity"] = float(v["polarity"])
        feats[f"llm_{theme}_intensity"] = float(v["intensity"])
    return feats


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)

    colmap = {c.lower(): c for c in df.columns}
    summary_col = colmap.get("summary", "Summary")
    text_col = colmap.get("text", "Text")
    id_col = colmap.get("id", "Id")

    cache = load_cache(CACHE_PATH)
    print("Loaded cache entries:", len(cache))
    print("Using model:", OLLAMA_MODEL)
    print("Using base :", OLLAMA_BASE)

    print("\nStarting LLM qualitative coding loop...")
    print("NOTE: The FIRST request may take time due to model warm-up.\n")

    rows = []

    for i, (_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if i == 0:
            print("▶ Sending FIRST review to Ollama now (model warm-up)...")

        summary = row.get(summary_col, "")
        text = row.get(text_col, "")
        key = row_hash(summary, text)

        if key in cache:
            codes = cache[key]
        else:
            review_text = make_text(summary, text)
            codes = strict_code_review(review_text)

            if i == 0:
                print("✔ First response received from Ollama. Progress will now be steady.")

            append_jsonl(CACHE_PATH, {"key": key, "value": codes})
            cache[key] = codes

        rows.append({
            "row_index": i,
            "Id": row.get(id_col, np.nan),
            **flatten(codes)
        })

        if (i % PRINT_EVERY) == 0:
            print(f"Processed {i+1}/{N_ROWS} reviews")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURES_OUT, index=False)

    print("\nDONE")
    print("Rows processed:", len(out_df))
    print("Saved to:", FEATURES_OUT)
    if os.path.exists(BAD_ROWS_OUT):
        print("Any JSON failures logged to:", BAD_ROWS_OUT)


if __name__ == "__main__":
    main()
