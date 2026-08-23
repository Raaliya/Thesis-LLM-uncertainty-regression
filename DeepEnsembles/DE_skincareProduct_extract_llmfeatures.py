import os
import re
import json
import time
import requests
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\skincare"

INPUT_CSV = os.path.join(ROOT, "reviews_1000_1500.csv")

# features-only file
OUTPUT_FEATURES_CSV = os.path.join(ROOT, "skincare_reviewtext_llm_features_2000.csv")

# merged final file: original dataset + llm features
OUTPUT_MERGED_CSV = os.path.join(ROOT, "reviews_1000_1500_with_llm_features_2000.csv")

# raw response log
RAW_OUTPUT_JSONL = os.path.join(ROOT, "skincare_reviewtext_llm_raw_outputs_2000.jsonl")

TEXT_COLUMN = "review_text"
N_ROWS = 2000
MAX_CHARS = 1800

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"   # change if needed

REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
SLEEP_BETWEEN_ROWS = 1.0
CHECKPOINT_EVERY = 25

THEMES = [
    "packaging_dispensing_functionality",
    "ease_of_use_instructions",
    "spf_reapplication_convenience",
    "finish_oil_shine_control",
    "coverage_texture_cakey_look",
    "skin_compatibility_irritation_breakouts",
    "sun_protection_effectiveness_confidence",
    "shade_cast_complexion_match",
    "portability_travel_touchup",
    "value_repurchase_recommendation",
]

# =========================================================
# PROMPT
# =========================================================
SYSTEM_INSTRUCTIONS = f"""
You are a qualitative researcher performing DEDUCTIVE qualitative coding using a fixed codebook.

CRITICAL RULES:
- Code based on explicit OR clearly implied meaning from the text.
- You may interpret short review phrases such as "powder won’t come out", "great for touch ups", "looks cakey", "works for oily skin", "not enough as primary SPF", "too light for my skin tone", or "I returned it" as indicating relevant themes.
- Do NOT hallucinate or invent information beyond the text.
- If a theme is not mentioned directly or clearly implied: present=0, polarity=0, intensity=0.
- Polarity: -1 negative, 0 neutral/mixed/unclear, 1 positive.
- Intensity: 0 none, 1 mild, 2 moderate, 3 strong.
- Output MUST be valid JSON ONLY. No extra text. No markdown. No explanations.

THEME INTERPRETATION GUIDELINES:
- packaging_dispensing_functionality: brush/dispenser works or fails, product not coming out, leaks, spills, cracks, broken packaging, messy design
- ease_of_use_instructions: easy or hard to use, confusing instructions, learning curve, difficult to know how much product is applied
- spf_reapplication_convenience: touch ups, reapplying SPF over makeup, midday refresh, convenient sunscreen reapplication
- finish_oil_shine_control: mattifying, reduces shine, good for oily skin, too drying, natural finish, greasy vs matte appearance
- coverage_texture_cakey_look: cakey, powdery, visible on skin, sits on top of makeup, smooth/light finish, patchy texture
- skin_compatibility_irritation_breakouts: irritation, sensitive skin, breakouts, clogged pores, redness, non-irritating
- sun_protection_effectiveness_confidence: confidence or doubt about SPF protection, prevents sunburn, not enough coverage, not suitable as primary sunscreen
- shade_cast_complexion_match: too light, too dark, ashy, chalky, pale look, poor shade range, complexion mismatch
- portability_travel_touchup: purse, bag, beach, vacation, theme parks, on-the-go portability, travel convenience
- value_repurchase_recommendation: worth it, overpriced, waste of money, returned, repurchase, recommend, not recommend

IMPORTANT CODING NOTES:
- These are consumer reviews, often informal and implicit.
- Use light semantic interpretation.
- Example mappings:
  - "nothing comes out of the brush" -> packaging_dispensing_functionality present negative
  - "great for touch ups over makeup" -> spf_reapplication_convenience present positive
  - "helps mattify my oily skin" -> finish_oil_shine_control present positive
  - "looked pale and cakey" -> coverage_texture_cakey_look present negative and shade_cast_complexion_match present negative
  - "made me break out" -> skin_compatibility_irritation_breakouts present negative
  - "I’m not sure this gives enough SPF alone" -> sun_protection_effectiveness_confidence present negative
  - "fits in my bag for travel" -> portability_travel_touchup present positive
  - "I returned it / would not repurchase" -> value_repurchase_recommendation present negative

GLOBAL:
- overall_sentiment: -1 / 0 / 1
- confidence: 0.0 to 1.0

REQUIRED JSON SCHEMA (exact keys preferred):
{{
  "overall_sentiment": 0,
  "confidence": 0.9,
  "themes": {{
    "packaging_dispensing_functionality": {{"present":0,"polarity":0,"intensity":0}},
    "ease_of_use_instructions": {{"present":0,"polarity":0,"intensity":0}},
    "spf_reapplication_convenience": {{"present":0,"polarity":0,"intensity":0}},
    "finish_oil_shine_control": {{"present":0,"polarity":0,"intensity":0}},
    "coverage_texture_cakey_look": {{"present":0,"polarity":0,"intensity":0}},
    "skin_compatibility_irritation_breakouts": {{"present":0,"polarity":0,"intensity":0}},
    "sun_protection_effectiveness_confidence": {{"present":0,"polarity":0,"intensity":0}},
    "shade_cast_complexion_match": {{"present":0,"polarity":0,"intensity":0}},
    "portability_travel_touchup": {{"present":0,"polarity":0,"intensity":0}},
    "value_repurchase_recommendation": {{"present":0,"polarity":0,"intensity":0}}
  }}
}}
""".strip()

# =========================================================
# HELPERS
# =========================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]

def build_zero_result():
    return {
        "overall_sentiment": 0,
        "confidence": 0.0,
        "themes": {
            theme: {"present": 0, "polarity": 0, "intensity": 0}
            for theme in THEMES
        }
    }

def normalize_numeric_value(v, allowed=None, default=0):
    if isinstance(v, bool):
        return default

    if isinstance(v, (int, float)):
        num = int(v)
    elif isinstance(v, str):
        s = v.strip()
        s = s.replace("+", "")
        s = s.strip('"').strip("'")
        try:
            num = int(float(s))
        except Exception:
            return default
    else:
        return default

    if allowed is not None and num not in allowed:
        return default
    return num

def normalize_confidence(v):
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        try:
            x = float(s)
        except Exception:
            return 0.0
    else:
        return 0.0

    if x < 0:
        x = 0.0
    if x > 1:
        x = 1.0
    return round(x, 4)

def clean_llm_json_text(text):
    text = text.strip()

    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    text = re.sub(r'(:\s*)\+(\d+)', r'\1\2', text)
    text = re.sub(r'("polarity"\s*:\s*)"\+?(-?\d+)"', r'\1\2', text)
    text = re.sub(r'("overall_sentiment"\s*:\s*)"\+?(-?\d+)"', r'\1\2', text)
    text = re.sub(r'("present"\s*:\s*)"(\d+)"', r'\1\2', text)
    text = re.sub(r'("intensity"\s*:\s*)"(\d+)"', r'\1\2', text)
    text = re.sub(r'("confidence"\s*:\s*)"([0-9]*\.?[0-9]+)"', r'\1\2', text)

    return text

def extract_json(text):
    cleaned = clean_llm_json_text(text)
    try:
        return json.loads(cleaned), cleaned, ""
    except Exception as e:
        return None, cleaned, f"json.loads failed: {e}"

def fix_missing_themes(obj):
    if not isinstance(obj, dict):
        obj = {}

    if "themes" not in obj or not isinstance(obj["themes"], dict):
        obj["themes"] = {}

    for theme in THEMES:
        if theme not in obj["themes"] or not isinstance(obj["themes"][theme], dict):
            obj["themes"][theme] = {"present": 0, "polarity": 0, "intensity": 0}

    return obj

def normalize_result(obj):
    obj = fix_missing_themes(obj)

    normalized = {
        "overall_sentiment": normalize_numeric_value(
            obj.get("overall_sentiment", 0),
            allowed={-1, 0, 1},
            default=0
        ),
        "confidence": normalize_confidence(obj.get("confidence", 0.0)),
        "themes": {}
    }

    for theme in THEMES:
        vals = obj["themes"].get(theme, {})
        normalized["themes"][theme] = {
            "present": normalize_numeric_value(vals.get("present", 0), allowed={0, 1}, default=0),
            "polarity": normalize_numeric_value(vals.get("polarity", 0), allowed={-1, 0, 1}, default=0),
            "intensity": normalize_numeric_value(vals.get("intensity", 0), allowed={0, 1, 2, 3}, default=0),
        }

    return normalized

def flatten_result(result):
    row = {
        "overall_sentiment": result["overall_sentiment"],
        "confidence": result["confidence"],
    }

    for theme in THEMES:
        row[f"{theme}_present"] = result["themes"][theme]["present"]
        row[f"{theme}_polarity"] = result["themes"][theme]["polarity"]
        row[f"{theme}_intensity"] = result["themes"][theme]["intensity"]

    return row

def query_ollama(review_text):
    user_prompt = f'''Code the following skincare review_text using the codebook.

REVIEW_TEXT:
"""{review_text}"""
'''

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 500,
        },
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()

            if not content:
                raise ValueError("Empty response content from Ollama")

            return content

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise last_error

def get_feature_columns():
    cols = ["source_index", "review_text", "overall_sentiment", "confidence"]
    for theme in THEMES:
        cols.extend([
            f"{theme}_present",
            f"{theme}_polarity",
            f"{theme}_intensity",
        ])
    cols.extend(["parse_error", "runtime_error"])
    return cols

def build_merged_dataframe(original_subset_df, features_df):
    # remove raw review_text from features before merge to avoid duplicate column names
    features_for_merge = features_df.drop(columns=["review_text"], errors="ignore")
    merged_df = original_subset_df.merge(features_for_merge, on="source_index", how="left")
    return merged_df

def save_checkpoint(feature_rows, original_subset_df):
    if not feature_rows:
        return

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df[get_feature_columns()]
    features_df.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8-sig")

    merged_df = build_merged_dataframe(original_subset_df, features_df)
    merged_df.to_csv(OUTPUT_MERGED_CSV, index=False, encoding="utf-8-sig")

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df["__text__"] = df[TEXT_COLUMN].apply(clean_text)
    df = df[df["__text__"].str.len() > 0].head(N_ROWS).copy()
    df = df.reset_index().rename(columns={"index": "source_index"})

    original_subset_df = df.drop(columns=["__text__"]).copy()

    print(f"Rows selected: {len(df)}")

    feature_rows = []
    processed_indices = set()

    # Resume from features file if it exists
    if os.path.exists(OUTPUT_FEATURES_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_FEATURES_CSV)
            if "source_index" in existing_df.columns:
                processed_indices = set(existing_df["source_index"].dropna().astype(int).tolist())
                feature_rows = existing_df.to_dict("records")
                print(f"Resume mode: found {len(processed_indices)} already processed rows in existing features CSV.")
        except Exception as e:
            print(f"Could not load existing features CSV for resume: {e}")

    if not os.path.exists(RAW_OUTPUT_JSONL):
        with open(RAW_OUTPUT_JSONL, "w", encoding="utf-8") as f:
            pass

    processed_count_this_run = 0
    success_count = 0
    zero_fallback_count = 0

    for _, row in df.iterrows():
        idx = int(row["source_index"])

        if idx in processed_indices:
            continue

        review_text = row["__text__"]

        print("\n" + "=" * 90)
        print(f"Processing row index: {idx}")
        print(f"Processed this run: {processed_count_this_run}")
        print(f"Review preview: {review_text[:350]}")

        raw_response = ""
        cleaned_response = ""
        parse_error = ""
        runtime_error = ""
        parsed_obj = None
        final_result = build_zero_result()

        try:
            raw_response = query_ollama(review_text)
            parsed_obj, cleaned_response, parse_error = extract_json(raw_response)

            if parsed_obj is None:
                final_result = build_zero_result()
                zero_fallback_count += 1
            else:
                final_result = normalize_result(parsed_obj)
                success_count += 1

        except Exception as e:
            runtime_error = str(e)
            final_result = build_zero_result()
            zero_fallback_count += 1

        flat = flatten_result(final_result)
        flat["source_index"] = idx
        flat["review_text"] = review_text
        flat["parse_error"] = parse_error
        flat["runtime_error"] = runtime_error

        feature_rows.append(flat)
        processed_indices.add(idx)
        processed_count_this_run += 1

        raw_log_item = {
            "source_index": idx,
            "review_text": review_text,
            "raw_response": raw_response,
            "cleaned_response": cleaned_response,
            "parse_error": parse_error,
            "runtime_error": runtime_error,
            "parsed_obj": parsed_obj,
            "final_result": final_result,
        }

        with open(RAW_OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_log_item, ensure_ascii=False) + "\n")

        if processed_count_this_run % CHECKPOINT_EVERY == 0:
            save_checkpoint(feature_rows, original_subset_df)
            print(f"Checkpoint saved after {processed_count_this_run} rows in this run.")
            print(f"Successful parses this run: {success_count}")
            print(f"Zero fallbacks this run: {zero_fallback_count}")

        time.sleep(SLEEP_BETWEEN_ROWS)

    save_checkpoint(feature_rows, original_subset_df)

    print("\nDone.")
    print(f"Features-only CSV saved to: {OUTPUT_FEATURES_CSV}")
    print(f"Merged final CSV saved to: {OUTPUT_MERGED_CSV}")
    print(f"Raw JSONL saved to: {RAW_OUTPUT_JSONL}")
    print(f"Successful parses this run: {success_count}")
    print(f"Zero fallbacks this run: {zero_fallback_count}")

    features_df = pd.DataFrame(feature_rows)
    features_df = features_df[get_feature_columns()]

    preview_cols = [
        "source_index",
        "overall_sentiment",
        "confidence",
        "packaging_dispensing_functionality_present",
        "ease_of_use_instructions_present",
        "spf_reapplication_convenience_present",
        "finish_oil_shine_control_present",
        "coverage_texture_cakey_look_present",
        "skin_compatibility_irritation_breakouts_present",
        "sun_protection_effectiveness_confidence_present",
        "shade_cast_complexion_match_present",
        "portability_travel_touchup_present",
        "value_repurchase_recommendation_present",
        "parse_error",
        "runtime_error",
    ]
    print("\nPreview of important columns:")
    print(features_df.tail(15)[preview_cols].to_string(index=False))

if __name__ == "__main__":
    main()

    