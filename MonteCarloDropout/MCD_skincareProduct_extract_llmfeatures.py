import os
import json
import re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# =========================
# CONFIG
# =========================
ROOT = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2"
IN_CSV = os.path.join(ROOT, "skincare_products.csv")
OUT_CSV = os.path.join(ROOT, "skincare_products_llm_2000_seed42.csv")
RAW_JSONL = os.path.join(ROOT, "skincare_llm_raw_outputs.jsonl")

TARGET = "rating"

N_ROWS = 2000
SEED = 42

# Pick the best text column available in skincare_products.csv
TEXT_CANDIDATES = ["description", "product_description", "product_name", "name", "ingredients", "brand", "category"]

MAX_CHARS = 1200

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.latest"   # <-- CHANGE THIS to your installed model

# LLM features we want (numeric 0-1 or 0-5 style)
FEATURES = [
    "llm_sentiment",           # 0..1 (negative->0, positive->1)
    "llm_product_quality",     # 0..5
    "llm_value_for_money",     # 0..5
    "llm_skin_suitability",    # 0..5
    "llm_ingredient_concern"   # 0..5 (higher = more concerns)
]
# =========================


def pick_text_column(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    for c in TEXT_CANDIDATES:
        if c.lower() in cols:
            # find actual case-sensitive column name
            return [x for x in df.columns if x.lower() == c.lower()][0]
    # fallback: first object column
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        return obj_cols[0]
    raise ValueError("No suitable text column found for LLM extraction.")


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")


def extract_json(text: str) -> dict:
    """
    Extract JSON object from model output robustly.
    """
    # try direct json
    text = text.strip()
    # find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    blob = m.group(0)
    return json.loads(blob)


def build_prompt(product_text: str) -> str:
    return f"""
You are extracting numeric features from a skincare product description for research.

Return ONLY a valid JSON object with these keys:
- llm_sentiment: number between 0 and 1
- llm_product_quality: integer 0 to 5
- llm_value_for_money: integer 0 to 5
- llm_skin_suitability: integer 0 to 5
- llm_ingredient_concern: integer 0 to 5

Rules:
- If information is missing, make your best guess from the text.
- Output JSON only. No extra text.

TEXT:
{product_text}
""".strip()


def coerce_features(d: dict) -> dict:
    out = {}
    for k in FEATURES:
        v = d.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        out[k] = v
    # clamp sentiment
    out["llm_sentiment"] = float(np.clip(out["llm_sentiment"], 0.0, 1.0))
    # clamp 0..5 for others
    for k in FEATURES:
        if k != "llm_sentiment":
            out[k] = float(np.clip(out[k], 0.0, 5.0))
    return out


def main():
    df = pd.read_csv(IN_CSV)

    # sample 2000 rows reproducibly
    if len(df) < N_ROWS:
        raise ValueError(f"Dataset has {len(df)} rows; cannot sample {N_ROWS}.")
    df = df.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)

    text_col = pick_text_column(df)
    print("Using text column:", text_col)

    # Ensure output file fresh
    if os.path.exists(RAW_JSONL):
        os.remove(RAW_JSONL)

    feats = {k: [] for k in FEATURES}
    raw_failures = 0

    with open(RAW_JSONL, "w", encoding="utf-8") as fraw:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            txt = str(row.get(text_col, ""))
            txt = txt.replace("\n", " ").strip()
            if len(txt) > MAX_CHARS:
                txt = txt[:MAX_CHARS]

            prompt = build_prompt(txt)

            try:
                resp = call_ollama(prompt)
                # save raw
                fraw.write(json.dumps({"i": int(i), "response": resp}, ensure_ascii=False) + "\n")

                parsed = extract_json(resp)
                parsed = coerce_features(parsed)

                for k in FEATURES:
                    feats[k].append(parsed[k])

            except Exception as e:
                raw_failures += 1
                # write failure with error
                fraw.write(json.dumps({"i": int(i), "error": str(e)}, ensure_ascii=False) + "\n")
                for k in FEATURES:
                    feats[k].append(0.0)

    # attach features
    for k in FEATURES:
        df[k] = feats[k]

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Raw outputs saved:", RAW_JSONL)
    print("Failures:", raw_failures, "out of", len(df))

    # quick check
    print("\nLLM feature summary (min/max/mean):")
    print(df[FEATURES].describe().T[["min", "max", "mean"]])
    print("\nNon-zero counts:")
    print((df[FEATURES] != 0).sum())


if __name__ == "__main__":
    main()
