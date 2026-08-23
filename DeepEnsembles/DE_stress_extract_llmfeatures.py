import os
import re
import json
import time
import requests
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\stress_analysis_normalized.csv"
OUTPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\stress_llm_features_2000.csv"
RAW_OUTPUT_JSONL = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\stress_llm_raw_outputs_2000.jsonl"

TEXT_COLUMN = "text"
N_ROWS = 2000
MAX_CHARS = 500

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_SLEEP = 2
SLEEP_BETWEEN_ROWS = 0.10
CHECKPOINT_EVERY = 25

THEMES = [
    "stress_overload",
    "anxiety_worry",
    "panic_physical_symptoms",
    "sleep_fatigue",
    "hopelessness_numbness",
    "trauma_abuse_past_harm",
    "relationship_social_stress",
    "work_school_financial_stress",
    "safety_self_harm_risk",
    "coping_help_seeking",
]

# =========================================================
# SYSTEM PROMPT
# =========================================================
SYSTEM_INSTRUCTIONS = """
You are a qualitative researcher performing deductive thematic coding using a fixed codebook.

CRITICAL RULES:
- Code only from the text provided.
- Use explicit meaning or clearly implied meaning only.
- Do not hallucinate, invent, or assume facts not supported by the text.
- If a theme is not mentioned directly or clearly implied, set:
  present = 0
  polarity = 0
  intensity = 0
- Polarity:
  -1 = negative / distressing / harmful
   0 = neutral / mixed / unclear
   1 = positive / protective / improving
- Intensity:
  0 = none
  1 = mild
  2 = moderate
  3 = strong
- Return valid JSON only.
- Do not return markdown.
- Do not return explanation.
- Do not return extra text before or after JSON.

THEME INTERPRETATION GUIDELINES:
- stress_overload: general overwhelm, pressure, overload, inability to cope, emotional burden
- anxiety_worry: anxiety, fear, worry, dread, overthinking, nervousness, unease
- panic_physical_symptoms: physical distress symptoms such as nausea, shaking, racing heart, breathlessness, floating feeling, agitation, sensory overload
- sleep_fatigue: insomnia, nightmares, poor sleep, waking distressed, exhaustion, fatigue
- hopelessness_numbness: helplessness, hopelessness, numbness, feeling stuck, worthlessness, despair
- trauma_abuse_past_harm: PTSD, abuse, assault, violence, past trauma, intrusive memories, triggers
- relationship_social_stress: conflict with partner, family, roommates, friends, loneliness, social exhaustion, interpersonal stress
- work_school_financial_stress: stress related to job, interview, work, study, school, money, bills, rent, housing, survival pressure
- safety_self_harm_risk: self-harm thoughts, suicidal ideation, unsafe conditions, fear of harm, immediate danger
- coping_help_seeking: therapy, treatment, specialist, advice-seeking, support-seeking, meditation, grounding, recovery effort

GLOBAL:
- overall_sentiment: -1 / 0 / 1
- confidence: 0.0 to 1.0

REQUIRED JSON SCHEMA:
{
  "overall_sentiment": -1,
  "confidence": 0.90,
  "themes": {
    "stress_overload": {"present":0,"polarity":0,"intensity":0},
    "anxiety_worry": {"present":0,"polarity":0,"intensity":0},
    "panic_physical_symptoms": {"present":0,"polarity":0,"intensity":0},
    "sleep_fatigue": {"present":0,"polarity":0,"intensity":0},
    "hopelessness_numbness": {"present":0,"polarity":0,"intensity":0},
    "trauma_abuse_past_harm": {"present":0,"polarity":0,"intensity":0},
    "relationship_social_stress": {"present":0,"polarity":0,"intensity":0},
    "work_school_financial_stress": {"present":0,"polarity":0,"intensity":0},
    "safety_self_harm_risk": {"present":0,"polarity":0,"intensity":0},
    "coping_help_seeking": {"present":0,"polarity":0,"intensity":0}
  }
}
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
        s = v.strip().replace("+", "").strip('"').strip("'")
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


def build_prompt(text_value):
    return f"""{SYSTEM_INSTRUCTIONS}

Code the following text using the codebook.

TEXT:
\"\"\"{text_value}\"\"\"
"""


def query_ollama(text_value):
    prompt = build_prompt(text_value)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 420,
        },
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "").strip()

            if not content:
                raise ValueError("Empty response content from Ollama")

            return content

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    raise last_error


def get_ordered_columns():
    ordered_cols = ["source_index", "text", "overall_sentiment", "confidence"]
    for theme in THEMES:
        ordered_cols.extend([
            f"{theme}_present",
            f"{theme}_polarity",
            f"{theme}_intensity",
        ])
    ordered_cols.extend(["parse_error", "runtime_error"])
    return ordered_cols


def save_checkpoint(feature_rows):
    if not feature_rows:
        return
    out_df = pd.DataFrame(feature_rows)
    out_df = out_df[get_ordered_columns()]
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


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

    print(f"Rows selected: {len(df)}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Endpoint: {OLLAMA_URL}")
    print(f"Max chars: {MAX_CHARS}")

    feature_rows = []
    processed_indices = set()

    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if "source_index" in existing_df.columns:
                processed_indices = set(existing_df["source_index"].dropna().astype(int).tolist())
                feature_rows = existing_df.to_dict("records")
                print(f"Resume mode: found {len(processed_indices)} already processed rows in existing CSV.")
        except Exception as e:
            print(f"Could not load existing output CSV for resume: {e}")

    if not os.path.exists(RAW_OUTPUT_JSONL):
        with open(RAW_OUTPUT_JSONL, "w", encoding="utf-8") as f:
            pass

    processed_count_this_run = 0
    success_count = 0
    zero_fallback_count = 0

    for idx, row in df.iterrows():
        if idx in processed_indices:
            continue

        text_value = row["__text__"]

        print("\n" + "=" * 90)
        print(f"Processing row index: {idx}")
        print(f"Processed this run: {processed_count_this_run}")
        print(f"Text preview: {text_value[:300]}")

        raw_response = ""
        cleaned_response = ""
        parse_error = ""
        runtime_error = ""
        parsed_obj = None
        final_result = build_zero_result()

        try:
            raw_response = query_ollama(text_value)
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
        flat["text"] = text_value
        flat["parse_error"] = parse_error
        flat["runtime_error"] = runtime_error

        feature_rows.append(flat)
        processed_indices.add(idx)
        processed_count_this_run += 1

        raw_log_item = {
            "source_index": idx,
            "text": text_value,
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
            save_checkpoint(feature_rows)
            print(f"Checkpoint saved after {processed_count_this_run} rows in this run.")
            print(f"Successful parses this run: {success_count}")
            print(f"Zero fallbacks this run: {zero_fallback_count}")

        time.sleep(SLEEP_BETWEEN_ROWS)

    save_checkpoint(feature_rows)

    print("\nDone.")
    print(f"Feature CSV saved to: {OUTPUT_CSV}")
    print(f"Raw JSONL saved to: {RAW_OUTPUT_JSONL}")
    print(f"Successful parses this run: {success_count}")
    print(f"Zero fallbacks this run: {zero_fallback_count}")

    out_df = pd.DataFrame(feature_rows)
    out_df = out_df[get_ordered_columns()]

    preview_cols = [
        "source_index",
        "overall_sentiment",
        "confidence",
        "stress_overload_present",
        "anxiety_worry_present",
        "panic_physical_symptoms_present",
        "sleep_fatigue_present",
        "hopelessness_numbness_present",
        "trauma_abuse_past_harm_present",
        "work_school_financial_stress_present",
        "safety_self_harm_risk_present",
        "coping_help_seeking_present",
        "parse_error",
        "runtime_error",
    ]
    print("\nPreview of important columns:")
    print(out_df.tail(15)[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()