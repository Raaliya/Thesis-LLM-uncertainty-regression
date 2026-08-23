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
# CONFIG (Amazon-style + Ollama stability)
# =========================
CSV_PATH = "skincare_products.csv"
OUT_DIR = "outputs_exp02_skincare_ollama_2000"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
FEATURES_OUT = os.path.join(OUT_DIR, "skincare_llm_features_2000.csv")
BAD_ROWS_OUT = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

# speed + stability (highlights are short)
MAX_CHARS = 100
TEMPERATURE = 0.0
NUM_PREDICT = 80          # reduced to prevent rambling/timeouts
N_ROWS = 2000             # HARD LIMIT

# heartbeat prints
PRINT_EVERY = 5

# request timeouts
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 900


# =========================
# THEMES (Deductive codebook)
# Keep same structure as Amazon: 10 themes, each has present/polarity/intensity
# =========================
THEMES = [
    "hydration_moisturizing",
    "anti_aging_firming",
    "brightening_even_tone",
    "acne_blemish_oil_control",
    "sensitive_skin_gentle",
    "clean_vegan_free_from",
    "derm_clinical_proven",
    "texture_finish_absorption",
    "fragrance_irritants_free",
    "sustainability_eco_packaging"
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
  *In highlights, polarity is usually +1, but if the text includes warnings/negatives (e.g., "may irritate"), use -1.
  *If mixed/unclear, use 0.
- Intensity: 0 none, 1 mild, 2 moderate, 3 strong.
  *Strong = repeated emphasis, "clinically proven", "powerful", "dramatically", "targets", "visibly", etc.
- Output MUST be valid JSON ONLY. No extra text. No markdown. No explanations.

CODEBOOK THEMES:
1) hydration_moisturizing
2) anti_aging_firming
3) brightening_even_tone
4) acne_blemish_oil_control
5) sensitive_skin_gentle
6) clean_vegan_free_from
7) derm_clinical_proven
8) texture_finish_absorption
9) fragrance_irritants_free
10) sustainability_eco_packaging

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
def make_text(highlights) -> str:
    t = "" if pd.isna(highlights) else str(highlights)
    combined = f"HIGHLIGHTS: {t}".strip()
    return combined[:MAX_CHARS] + ("…" if len(combined) > MAX_CHARS else "")

def row_hash(highlights) -> str:
    return hashlib.sha256(str(highlights).encode("utf-8", errors="ignore")).hexdigest()

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


# =========================
# OLLAMA CALL (timeout + stop tokens + retries)
# =========================
@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(6))
def ollama_chat(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "stop": ["}\n", "}\r\n", "}"]
        }
    }

    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "")


def strict_code_highlights(highlights_text: str) -> Dict[str, Any]:
    # Step 1
    user_msg = (
        "Return ONLY valid JSON for the schema given. "
        "No prose, no markdown, no backticks.\n\n"
        f"TEXT:\n{highlights_text}"
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

    # Step 2 (repair)
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

    # FIXED 2000 rows, fixed seed (consistency with your experiments)
    df = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)

    colmap = {c.lower(): c for c in df.columns}
    highlights_col = colmap.get("highlights", "highlights")

    cache = load_cache(CACHE_PATH)
    print("Loaded cache entries:", len(cache))
    print("Using model:", OLLAMA_MODEL)
    print("Using base :", OLLAMA_BASE)
    print("Highlights col:", highlights_col)

    print("\nStarting LLM qualitative coding loop...")
    print("NOTE: The FIRST request may take a few minutes due to model warm-up.\n")

    rows = []

    for i, (_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if i == 0:
            print("▶ Sending FIRST highlights text to Ollama now (model warm-up)...")

        highlights = row.get(highlights_col, "")
        key = row_hash(highlights)

        if key in cache:
            codes = cache[key]
        else:
            text_in = make_text(highlights)
            codes = strict_code_highlights(text_in)

            if i == 0:
                print("✔ First response received from Ollama. Progress will now be steady.")

            append_jsonl(CACHE_PATH, {"key": key, "value": codes})
            cache[key] = codes

        # OUTPUT ONLY LLM FEATURES (no product_id, no names, no brand columns)
        rows.append({
            "row_index": i,
            **flatten(codes)
        })

        if (i % PRINT_EVERY) == 0:
            print(f"Processed {i+1}/{N_ROWS} rows")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURES_OUT, index=False)

    print("\nDONE")
    print("Rows processed:", len(out_df))
    print("Saved to:", FEATURES_OUT)
    if os.path.exists(BAD_ROWS_OUT):
        print("Any JSON failures logged to:", BAD_ROWS_OUT)


if __name__ == "__main__":
    main()
