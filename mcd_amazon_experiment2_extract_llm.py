import os
import re
import json
import hashlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt


# =========================
# CONFIG
# =========================
CSV_PATH = "amazon_reviews.csv"
OUT_DIR = "outputs_exp02_amazon_ollama_2000"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
FEATURES_OUT = os.path.join(OUT_DIR, "amazon_llm_features_2000.csv")
BAD_ROWS_OUT = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

# speed + stability
MAX_CHARS = 1200          # smaller = faster + fewer partial outputs
TEMPERATURE = 0.0
NUM_PREDICT = 300         # cap output tokens so it doesn't ramble
N_ROWS = 2000             # HARD LIMIT

# heartbeat prints so it never "looks stuck"
PRINT_EVERY = 10          # print every N rows

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
# HELPERS
# =========================
def make_text(summary, text) -> str:
    s = "" if pd.isna(summary) else str(summary)
    t = "" if pd.isna(text) else str(text)
    combined = f"SUMMARY: {s}\nTEXT: {t}".strip()
    return combined[:MAX_CHARS] + ("…" if len(combined) > MAX_CHARS else "")

def row_hash(summary, text) -> str:
    return hashlib.sha256((str(summary) + "||" + str(text)).encode("utf-8", errors="ignore")).hexdigest()

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


@retry(wait=wait_exponential(min=1, max=15), stop=stop_after_attempt(4))
def ollama_chat(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "")


def strict_code_review(review_text: str) -> Dict[str, Any]:
    """
    1) Ask model for JSON-only
    2) Parse
    3) If invalid, ask model to REPAIR to valid JSON
    4) If still invalid, return zeros (and log debug)
    """
    # Step 1: primary coding
    user_msg = (
        "Return ONLY valid JSON for the schema given. "
        "No prose, no markdown, no backticks.\n\n"
        f"REVIEW:\n{review_text}"
    )

    raw1 = ollama_chat([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_msg}
    ])

    json_block = extract_json_block(raw1)
    if json_block:
        try:
            return normalize(json.loads(json_block))
        except json.JSONDecodeError:
            pass

    # Step 2: repair pass
    repair_msg = (
        "Convert the following text into STRICT VALID JSON matching the required schema. "
        "Return JSON only. If something is missing, use zeros.\n\n"
        f"TEXT TO FIX:\n{raw1}"
    )

    raw2 = ollama_chat([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": repair_msg}
    ])

    json_block2 = extract_json_block(raw2)
    if json_block2:
        try:
            return normalize(json.loads(json_block2))
        except json.JSONDecodeError:
            pass

    # Step 3: fail-safe
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
    print("NOTE: The FIRST request may take a few minutes due to model warm-up.\n")

    rows = []

    # iterate with enumerate so we can print row counts reliably
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

        # heartbeat prints
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
