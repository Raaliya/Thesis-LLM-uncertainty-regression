import os
import json
import re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# ============================================================
# CONFIGURATION (CHANGE ONLY THESE FOR DIFFERENT DATASETS)
# ============================================================

ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2"

INPUT_FILENAME = "skincare_2000_seed42.csv"  # CHANGE per dataset
INPUT_PATH = os.path.join(ROOT, "data_subsets", INPUT_FILENAME)

OUTPUT_FILENAME = INPUT_FILENAME.replace(".csv", "_llm.csv")
OUTPUT_PATH = os.path.join(ROOT, "data_subsets", OUTPUT_FILENAME)

RAW_LOG_PATH = os.path.join(ROOT, "data_subsets", "llm_raw_outputs_log.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

TEXT_CANDIDATES = [
    "review_text",
    "text",
    "description",
    "product_description",
    "ingredients",
    "summary",
    "name",
    "product_name"
]

MAX_CHARS = 1000
# ============================================================

FEATURES = [
    "llm_sentiment",
    "llm_product_quality",
    "llm_value_for_money",
    "llm_skin_suitability",
    "llm_ingredient_concern"
]


# ============================================================
# PROMPT (MUST REMAIN IDENTICAL FOR ALL DATASETS)
# ============================================================

def prompt_for(text):
    return f"""
You are a skincare product evaluation expert.

Based ONLY on the text below, estimate the following scores:

- llm_sentiment:
  0 = very negative
  1 = very positive

- llm_product_quality:
  0 = very poor quality
  5 = excellent quality

- llm_value_for_money:
  0 = extremely overpriced
  5 = excellent value

- llm_skin_suitability:
  0 = likely harmful or unsuitable
  5 = highly suitable and beneficial

- llm_ingredient_concern:
  0 = no concerning ingredients
  5 = highly concerning ingredients

Return ONLY valid JSON.
No explanation.

TEXT:
{text}
""".strip()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pick_text_column(df):
    lower_map = {c.lower(): c for c in df.columns}
    for c in TEXT_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        return obj_cols[0]
    raise ValueError("No text column found.")


def extract_json(text):
    match = re.search(r"\{.*\}", text.strip(), flags=re.S)
    if not match:
        raise ValueError("No JSON found in LLM response.")
    return json.loads(match.group(0))


def coerce_features(d):
    out = {}
    for k in FEATURES:
        v = d.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        out[k] = v

    # Clip to defined ranges
    out["llm_sentiment"] = float(np.clip(out["llm_sentiment"], 0.0, 1.0))

    for k in FEATURES:
        if k != "llm_sentiment":
            out[k] = float(np.clip(out[k], 0.0, 5.0))

    return out


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found:\n{INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    text_col = pick_text_column(df)
    print("✅ Using text column:", text_col)

    if os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)

    features_dict = {k: [] for k in FEATURES}
    failures = 0

    with open(RAW_LOG_PATH, "w", encoding="utf-8") as raw_file:

        for i, row in tqdm(df.iterrows(), total=len(df)):

            text = str(row.get(text_col, "")).replace("\n", " ").strip()

            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]

            prompt = prompt_for(text)

            try:
                response = call_ollama(prompt)

                raw_file.write(json.dumps({"i": int(i), "response": response}, ensure_ascii=False) + "\n")

                parsed = coerce_features(extract_json(response))

                for k in FEATURES:
                    features_dict[k].append(parsed[k])

            except Exception as e:
                failures += 1
                raw_file.write(json.dumps({"i": int(i), "error": str(e)}, ensure_ascii=False) + "\n")
                for k in FEATURES:
                    features_dict[k].append(0.0)

    for k in FEATURES:
        df[k] = features_dict[k]

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n==========================================")
    print("Extraction Completed")
    print("Output saved to:", OUTPUT_PATH)
    print("Failures:", failures, "out of", len(df))
    print("==========================================")

    print("\nFeature Summary:")
    print(df[FEATURES].describe().T[["min", "max", "mean"]])

    print("\nNon-zero counts:")
    print((df[FEATURES] != 0).sum())


if __name__ == "__main__":
    main()
