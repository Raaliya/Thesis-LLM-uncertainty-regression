import os
import json
import re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2"

IN_SUBSET = os.path.join(ROOT, "data_subsets", "flipkart_2000_seed42.csv")
OUT_LLM = os.path.join(ROOT, "data_subsets", "flipkart_2000_seed42_llm.csv")
RAW_JSONL = os.path.join(ROOT, "data_subsets", "flipkart_llm_raw_outputs.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

MAX_CHARS = 1000  # your chosen standard for ALL datasets

TEXT_CANDIDATES = ["review", "summary", "review_text", "text", "product_title", "name"]

FEATURES = [
    "llm_sentiment",
    "llm_product_quality",
    "llm_value_for_money",
    "llm_skin_suitability",
    "llm_ingredient_concern"
]

# ============================================================
# SAME PROMPT (DO NOT CHANGE ACROSS DATASETS)
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

def pick_text_column(df):
    lower_map = {c.lower(): c for c in df.columns}
    for c in TEXT_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        return obj_cols[0]
    raise ValueError("No suitable text column found.")


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


def extract_json(text):
    m = re.search(r"\{.*\}", text.strip(), flags=re.S)
    if not m:
        raise ValueError("No JSON object found in output.")
    return json.loads(m.group(0))


def coerce(d):
    out = {}
    for k in FEATURES:
        v = d.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        out[k] = v

    out["llm_sentiment"] = float(np.clip(out["llm_sentiment"], 0.0, 1.0))
    for k in FEATURES:
        if k != "llm_sentiment":
            out[k] = float(np.clip(out[k], 0.0, 5.0))
    return out


def main():
    if not os.path.exists(IN_SUBSET):
        raise FileNotFoundError(f"Missing subset:\n{IN_SUBSET}\nRun bnn_make_flipkart_2000_sample_and_split.py first.")

    df = pd.read_csv(IN_SUBSET)
    text_col = pick_text_column(df)
    print("✅ Using text column:", text_col)

    if os.path.exists(RAW_JSONL):
        os.remove(RAW_JSONL)

    feats = {k: [] for k in FEATURES}
    failures = 0

    with open(RAW_JSONL, "w", encoding="utf-8") as fraw:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            txt = str(row.get(text_col, "")).replace("\n", " ").strip()
            if len(txt) > MAX_CHARS:
                txt = txt[:MAX_CHARS]

            try:
                resp = call_ollama(prompt_for(txt))
                fraw.write(json.dumps({"i": int(i), "response": resp}, ensure_ascii=False) + "\n")
                parsed = coerce(extract_json(resp))
                for k in FEATURES:
                    feats[k].append(parsed[k])
            except Exception as e:
                failures += 1
                fraw.write(json.dumps({"i": int(i), "error": str(e)}, ensure_ascii=False) + "\n")
                for k in FEATURES:
                    feats[k].append(0.0)

    for k in FEATURES:
        df[k] = feats[k]

    df.to_csv(OUT_LLM, index=False)
    print("\n✅ Saved:", OUT_LLM)
    print("✅ Raw log:", RAW_JSONL)
    print("Failures:", failures, "out of", len(df))

    print("\nLLM feature summary:")
    print(df[FEATURES].describe().T[["min", "max", "mean"]])

    print("\nNon-zero counts:")
    print((df[FEATURES] != 0).sum())


if __name__ == "__main__":
    main()
