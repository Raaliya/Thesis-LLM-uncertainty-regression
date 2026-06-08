import os
import json
import time
import csv
import pandas as pd
import requests

# ==========================================================
# 🔧 EDIT ONLY THIS SECTION IF NEEDED
# ==========================================================

#INPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\data_subsets\women_2000_seed42.csv"
INPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\women_clothing_reviews_normalized.csv"

# IMPORTANT: overwrite the same Exp-02 file path you train on
OUTPUT_CSV = r"D:\MAT-UNI\SEMESTER 02\THESIS\VS Code Implementation 1.2\Deep Ensembles\women_exp02_with_llm.csv"

CHECKPOINT_FILE = "women_thematic_checkpoint_safe.json"

MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

REVIEW_COLUMN = "Review Text"   # change if your CSV uses different name
TITLE_COLUMN = "Title"          # change if your CSV uses different name

N_ROWS = 2000
MAX_CHARS = 1000

SAVE_EVERY = 25          # writes a FULL safe CSV every 25 rows
SLEEP_BETWEEN = 0.2      # reduce if you want faster

# ==========================================================


CODEBOOK = {
    "T1_PRODUCT_QUALITY": "Mentions fabric/material quality, stitching, durability, comfort.",
    "T2_FIT_SIZING": "Mentions fit, sizing accuracy, too small/large, tight/loose.",
    "T3_STYLE_APPEARANCE": "Mentions look, style, color, design, flattering/unflattering.",
    "T4_VALUE_FOR_MONEY": "Mentions price fairness or value.",
    "T5_DELIVERY_PACKAGING": "Mentions shipping, delivery speed, packaging condition.",
    "T6_DEFECT_DAMAGE": "Mentions damage, defect, stain, tear, poor finishing.",
    "T7_RETURN_REFUND": "Mentions return, exchange, refund process.",
    "T8_RECOMMENDATION": "Explicitly recommends or not recommends.",
    "T9_EXPECTATION_GAP": "Mentions mismatch vs description/photos.",
    "T10_EMOTIONAL_INTENSITY": "Strong emotional tone."
}


def build_prompt(text: str) -> str:
    return f"""
You are a qualitative research assistant performing deductive thematic coding.

Use the fixed codebook below.
For each theme return:
0 = Absent
1 = Present

Return ONLY valid JSON with these exact keys.

Codebook:
{json.dumps(CODEBOOK, indent=2)}

Review:
{text}
""".strip()


def call_ollama(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )
    r.raise_for_status()
    return r.json()["response"]


def parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Invalid JSON output")


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return int(json.load(f).get("next_row", 0))
    return 0


def save_checkpoint(row: int):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"next_row": int(row)}, f)


def safe_save_csv(df: pd.DataFrame, path: str):
    """
    Full safe overwrite save.
    QUOTE_ALL + escapechar prevents the “Expected X fields, saw Y” corruption forever.
    """
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def main():
    print("Loading input CSV...")
    df = pd.read_csv(INPUT_CSV)

    if len(df) > N_ROWS:
        df = df.iloc[:N_ROWS].copy()

    # Ensure thematic columns exist
    for k in CODEBOOK.keys():
        if k not in df.columns:
            df[k] = pd.NA

    start_row = load_checkpoint()
    print(f"Resuming from row: {start_row}")
    print(f"Rows to process  : {len(df)}")
    print(f"Model            : {MODEL_NAME}")

    failures = 0

    for i in range(start_row, len(df)):

        title = ""
        review = ""

        if TITLE_COLUMN in df.columns and pd.notna(df.loc[i, TITLE_COLUMN]):
            title = str(df.loc[i, TITLE_COLUMN])

        if REVIEW_COLUMN in df.columns and pd.notna(df.loc[i, REVIEW_COLUMN]):
            review = str(df.loc[i, REVIEW_COLUMN])

        text = f"Title: {title} | Review: {review}".strip()
        text = text[:MAX_CHARS]

        if not text or text == "Title:  | Review:":
            themes = {k: 0 for k in CODEBOOK.keys()}
        else:
            try:
                response = call_ollama(build_prompt(text))
                raw = parse_json(response)

                themes = {}
                for k in CODEBOOK.keys():
                    try:
                        themes[k] = 1 if int(raw.get(k, 0)) == 1 else 0
                    except Exception:
                        themes[k] = 0

            except Exception as e:
                failures += 1
                print(f"Row {i} failed: {e}")
                themes = {k: 0 for k in CODEBOOK.keys()}

        # Write themes into df
        for k, v in themes.items():
            df.loc[i, k] = v

        # Save checkpoint + safe CSV overwrite periodically
        save_checkpoint(i + 1)

        if (i + 1) % SAVE_EVERY == 0:
            safe_save_csv(df, OUTPUT_CSV)
            print(f"Saved @ {i+1}/{len(df)} | failures={failures}")

        time.sleep(SLEEP_BETWEEN)

    # Final save
    safe_save_csv(df, OUTPUT_CSV)

    print("\n✅ Thematic coding completed (SAFE CSV).")
    print("Output file:", OUTPUT_CSV)
    print("Rows       :", len(df))
    print("Failures   :", failures)


if __name__ == "__main__":
    main()