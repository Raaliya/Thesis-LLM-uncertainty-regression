import os
import re
import json
import hashlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# =========================================================
# Stress Analysis | Exp02 | LLM Feature Extraction (Ollama)
# OUTPUT ONLY 5 LLM FEATURES
# =========================================================

# =========================
# CONFIG
# =========================
CSV_PATH = "stress_analysis_normalized.csv"
OUT_DIR = "outputs_exp02_stress_ollama_5feats"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
FEATURES_OUT = os.path.join(OUT_DIR, "stress_llm_features_5.csv")
BAD_ROWS_OUT = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

MAX_CHARS = 300
TEMPERATURE = 0.0
NUM_PREDICT = 220
PRINT_EVERY = 5

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 900

# =========================
# THEMES (still used internally to compute overall stress)
# =========================
THEMES = [
    "work_school_pressure",
    "relationship_family_conflict",
    "health_mental_health",
    "financial_stress",
    "time_overload_burnout",
    "sleep_fatigue",
    "social_isolation_loneliness",
    "uncertainty_future_worry",
    "coping_support_seeking",
    "acute_crisis_distress"
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
1) work_school_pressure
2) relationship_family_conflict
3) health_mental_health
4) financial_stress
5) time_overload_burnout
6) sleep_fatigue
7) social_isolation_loneliness
8) uncertainty_future_worry
9) coping_support_seeking
10) acute_crisis_distress

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
def make_text(text) -> str:
    t = "" if pd.isna(text) else str(text)
    combined = f"TEXT: {t}".strip()
    return combined[:MAX_CHARS] + ("…" if len(combined) > MAX_CHARS else "")

def row_hash(text) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="ignore")).hexdigest()

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
# OLLAMA CALL (NO stop tokens)
# =========================
def ollama_chat(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT}
    }
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "")

def strict_code_text(text_in: str) -> Dict[str, Any]:
    user_msg = (
        "Return ONLY valid JSON for the schema given. "
        "No prose, no markdown, no backticks.\n\n"
        f"TEXT:\n{text_in}"
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

# =========================
# 5-FEATURE OUTPUT
# =========================
def compute_stress_intensity(codes: Dict[str, Any]) -> float:
    # Average intensity across the 8 stressor themes (exclude coping + crisis)
    stressor_themes = [
        "work_school_pressure",
        "relationship_family_conflict",
        "health_mental_health",
        "financial_stress",
        "time_overload_burnout",
        "sleep_fatigue",
        "social_isolation_loneliness",
        "uncertainty_future_worry",
    ]
    vals = []
    for t in stressor_themes:
        vals.append(float(codes["themes"][t]["intensity"]))
    return float(np.mean(vals)) if vals else 0.0

def flatten_5(codes: Dict[str, Any]) -> Dict[str, float]:
    return {
        "llm_overall_sentiment": float(codes["overall_sentiment"]),
        "llm_confidence": float(codes["confidence"]),
        "llm_stress_intensity": compute_stress_intensity(codes),
        "llm_coping_intensity": float(codes["themes"]["coping_support_seeking"]["intensity"]),
        "llm_crisis_intensity": float(codes["themes"]["acute_crisis_distress"]["intensity"]),
    }

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index(drop=True)
    df["row_index"] = np.arange(len(df), dtype=int)

    colmap = {c.lower(): c for c in df.columns}
    text_col = colmap.get("text", "text")

    cache = load_cache(CACHE_PATH)
    print("Loaded cache entries:", len(cache))
    print("Using model:", OLLAMA_MODEL)
    print("Using base :", OLLAMA_BASE)
    print("Text col   :", text_col)
    print("Rows       :", len(df))
    print("MAX_CHARS  :", MAX_CHARS)
    print("NUM_PREDICT:", NUM_PREDICT)

    print("\nStarting LLM qualitative coding loop...")
    print("NOTE: The FIRST request may take a few minutes due to model warm-up.\n")

    rows = []

    for i, (_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if i == 0:
            print("▶ Sending FIRST text to Ollama now (model warm-up)...")

        text_val = row.get(text_col, "")
        key = row_hash(text_val)

        if key in cache:
            codes = cache[key]
        else:
            codes = strict_code_text(make_text(text_val))
            if i == 0:
                print("✔ First response received from Ollama. Progress will now be steady.")
            append_jsonl(CACHE_PATH, {"key": key, "value": codes})
            cache[key] = codes

        rows.append({
            "row_index": int(row["row_index"]),
            **flatten_5(codes)
        })

        if (i % PRINT_EVERY) == 0:
            print(f"Processed {i+1}/{len(df)} rows")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURES_OUT, index=False)

    llm_cols = [c for c in out_df.columns if c.startswith("llm_")]
    nonzero_cells = int((out_df[llm_cols] != 0).sum().sum())
    nonzero_rows = int((out_df[llm_cols].sum(axis=1) != 0).sum())

    print("\nDONE")
    print("Rows processed:", len(out_df))
    print("Saved to:", FEATURES_OUT)
    print("Sanity check:")
    print(" - Nonzero cells:", nonzero_cells)
    print(" - Rows with any nonzero:", nonzero_rows)

    if os.path.exists(BAD_ROWS_OUT):
        print("Any JSON failures logged to:", BAD_ROWS_OUT)

if __name__ == "__main__":
    main()
