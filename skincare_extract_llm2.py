import os
import re
import json
import ast
import hashlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt


# =========================
# CONFIG (Skincare | Ollama | Highlights-only)
# =========================
CSV_PATH = "Skincare_product_info.csv"
OUT_DIR = "outputs_exp02_skincare_ollama_2000"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(OUT_DIR, "ollama_cache.jsonl")
FEATURES_OUT = os.path.join(OUT_DIR, "skincare_llm_features_2000.csv")
BAD_ROWS_OUT = os.path.join(OUT_DIR, "bad_rows_debug.jsonl")
SAMPLES_OUT = os.path.join(OUT_DIR, "first10_raw_samples.jsonl")

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

MAX_CHARS = 800
TEMPERATURE = 0.0
NUM_PREDICT = 220
N_ROWS = 2000
PRINT_EVERY = 25

# PowerShell:  $env:FORCE_RECODE="1"
# CMD:         set FORCE_RECODE=1
FORCE_RECODE = os.environ.get("FORCE_RECODE", "0") == "1"


# =========================
# THEMES (Skincare Highlights) - Deductive Codebook
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

SYSTEM_INSTRUCTIONS = f"""
You are a qualitative researcher performing DEDUCTIVE qualitative coding using a fixed codebook.

CRITICAL RULES:
- Only code what is explicitly stated in the text. Do NOT guess.
- If a theme is not mentioned: present=0, polarity=0, intensity=0.
- Polarity: -1 negative, 0 neutral/mixed/unclear, +1 positive.
  *If the text includes warnings/negatives (e.g., "may irritate"), use -1.
- Intensity: 0 none, 1 mild, 2 moderate, 3 strong.
- Output MUST be valid JSON ONLY. No extra text. No markdown. No explanations.

GUIDANCE FOR SKINCARE HIGHLIGHTS:
- "Hydrating", "Good for: Dryness", "Moisturizing", "Hyaluronic Acid" -> hydration_moisturizing
- "Retinol", "Good for: Loss of firmness", "Firming", "Anti-aging" -> anti_aging_firming
- "Good for: Dullness/Uneven Texture", "Brightening", "Even tone", "Dark spots" -> brightening_even_tone
- "Acne", "Blemish", "Oil control", "Salicylic Acid", "Good for: Pores" -> acne_blemish_oil_control
- "Sensitive skin", "Gentle", "Fragrance Free" -> sensitive_skin_gentle or fragrance_irritants_free as appropriate
- "Vegan", "Cruelty-Free", "Without Parabens", "Without Sulfates", "Clean at Sephora" -> clean_vegan_free_from
- "Clinically proven", "Dermatologist tested", "Clinical", "Award winner" -> derm_clinical_proven only if explicitly clinical/derm/proven
- "Radiant Finish", "Natural Finish", "Satin Finish", "Absorbs quickly", "Lightweight" -> texture_finish_absorption
- "Fragrance Free", "Without Fragrance", "Without Parfum" -> fragrance_irritants_free
- "Refill Available", "Clean + Planet Positive", "Eco packaging", "Sustainable" -> sustainability_eco_packaging

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
- overall_sentiment: -1 / 0 / +1  (overall tone of the highlights)
- confidence: 0.0–1.0  (how sure you are about the coding based on explicit text)

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

REPAIR_SYSTEM = """
You are a strict JSON repair tool.
Return ONLY a single valid JSON object. No markdown. No commentary.
If the input cannot be converted, output the required schema with zeros.
""".strip()


# =========================
# HELPERS
# =========================
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


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

        out["themes"][theme] = {
            "present": present,
            "polarity": polarity,
            "intensity": intensity
        }

    return out


def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_cache(path: str) -> Dict[str, Any]:
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cache[obj["key"]] = obj["value"]
                except Exception:
                    continue
    return cache


def sha_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def extract_json_block(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None

    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    m = re.search(r"({.*})", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


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


def parse_highlights_text(raw_value) -> str:
    """
    Convert raw highlights into clean readable text.

    Example:
    "['Vegan', 'Hydrating', 'Good for: Dryness']"
    ->
    "Vegan, Hydrating, Good for: Dryness"
    """
    if pd.isna(raw_value):
        return ""

    raw = str(raw_value).strip()
    if raw.lower() in {"", "nan", "none", "null"}:
        return ""

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            cleaned_items = []
            for item in parsed:
                item_s = safe_str(item)
                if item_s:
                    cleaned_items.append(item_s)
            return ", ".join(cleaned_items)
        return raw
    except Exception:
        return raw


def build_text_from_highlights_only(row: pd.Series, highlights_col: str) -> str:
    highlights = parse_highlights_text(row.get(highlights_col, ""))
    combined = f"HIGHLIGHTS: {highlights}".strip()

    if len(combined) > MAX_CHARS:
        combined = combined[:MAX_CHARS] + "…"

    return combined


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


def strict_code_text(text_in: str, debug_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    1) Ask model for JSON-only
    2) Parse
    3) Repair if needed
    4) If still invalid => zeros + debug log
    """
    if len(text_in.strip()) < 15 or text_in.strip() == "HIGHLIGHTS:":
        append_jsonl(BAD_ROWS_OUT, {"error": "empty_text", "text_in": text_in, **debug_meta})
        return zero_schema()

    user_msg = (
        "Return ONLY valid JSON for the schema given. "
        "No prose, no markdown, no backticks.\n\n"
        f"TEXT:\n{text_in}"
    )

    raw1 = ollama_chat([
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_msg}
    ])

    jb1 = extract_json_block(raw1)
    if jb1:
        try:
            return normalize(json.loads(jb1))
        except Exception:
            pass

    repair_msg = (
        "Fix this into STRICT VALID JSON that matches the required schema exactly. "
        "Return JSON only.\n\n"
        f"BAD OUTPUT:\n{raw1}\n\n"
        f"REQUIRED SCHEMA TEMPLATE:\n{json.dumps(zero_schema(), ensure_ascii=False)}"
    )

    raw2 = ollama_chat([
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": repair_msg}
    ])

    jb2 = extract_json_block(raw2)
    if jb2:
        try:
            return normalize(json.loads(jb2))
        except Exception:
            pass

    append_jsonl(BAD_ROWS_OUT, {
        "error": "json_decode_failed",
        "raw1": raw1,
        "raw2": raw2,
        "text_in": text_in,
        **debug_meta
    })
    return zero_schema()


def main():
    df = pd.read_csv(CSV_PATH)

    if len(df) >= N_ROWS:
        df = df.sample(n=N_ROWS, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    colmap = {c.lower(): c for c in df.columns}

    highlights_col = colmap.get("highlights")
    if not highlights_col:
        raise ValueError(
            "Could not find a 'highlights' column in the CSV (case-insensitive). "
            f"Available columns: {list(df.columns)}"
        )

    id_col = (
        colmap.get("product_id")
        or colmap.get("id")
        or colmap.get("listing_id")
        or colmap.get("asin")
        or "product_id"
    )

    cache = {} if FORCE_RECODE else load_cache(CACHE_PATH)

    print("Using model      :", OLLAMA_MODEL)
    print("Using base       :", OLLAMA_BASE)
    print("MAX_CHARS        :", MAX_CHARS)
    print("FORCE_RECODE     :", FORCE_RECODE)
    print("Loaded cache     :", len(cache))
    print("Highlights col   :", highlights_col)
    print("Detected id_col  :", id_col)

    rows = []
    nonzero_present_total = 0
    first10_saved = 0

    for i, (_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        if i == 0:
            print("\n▶ Sending FIRST highlights text to Ollama now (warm-up)...")

        text_in = build_text_from_highlights_only(row, highlights_col)
        key = sha_key(text_in)

        debug_meta = {"row_index": i, "product_id": row.get(id_col, np.nan)}

        if (not FORCE_RECODE) and (key in cache):
            codes = cache[key]
        else:
            codes = strict_code_text(text_in, debug_meta)
            cache[key] = codes
            append_jsonl(CACHE_PATH, {"key": key, "value": codes})

        feats = flatten(codes)
        nonzero_present = sum(int(feats[f"llm_{t}_present"]) for t in THEMES)
        nonzero_present_total += nonzero_present

        if first10_saved < 10:
            append_jsonl(SAMPLES_OUT, {
                "row_index": i,
                "product_id": row.get(id_col, np.nan),
                "text_in": text_in,
                "codes": codes
            })
            first10_saved += 1

        rows.append({
            "row_index": i,
            "product_id": row.get(id_col, np.nan),
            **feats
        })

        if (i % PRINT_EVERY) == 0:
            print(f"Processed {i+1}/{len(df)} rows | running present-sum={nonzero_present_total}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURES_OUT, index=False)

    print("\nDONE")
    print("Rows processed:", len(out_df))
    print("Saved to      :", FEATURES_OUT)
    print("First 10 samples saved to:", SAMPLES_OUT)

    if nonzero_present_total == 0:
        print("\nWARNING: All theme 'present' values are still zero across the dataset.")
        print("Check first10 samples:", SAMPLES_OUT)
        print("Check failures log   :", BAD_ROWS_OUT)
        print("Also ensure Ollama is running and the model is correct:", OLLAMA_MODEL)

    if os.path.exists(BAD_ROWS_OUT):
        print("Any JSON failures logged to:", BAD_ROWS_OUT)


if __name__ == "__main__":
    main()