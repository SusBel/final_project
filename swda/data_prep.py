"""
=============================================================================
  data_preparation.py — SwDA Data Ingestion & Preprocessing
  Dual-Head Chatbot | Machine 1: Intent Classification

  Scans ALL subfolders for SwDA CSV files (sw00utt/, sw01utt/, ...),
  maps raw act_tags → simplified intents, cleans text, deduplicates,
  and writes  processed_intents.csv  ready for model_training.py.

  Run:
      python data_preparation.py
  Output:
      processed_intents.csv   — final training-ready dataset
      prep_report.txt         — data quality summary
=============================================================================
"""

import os
import re
import pandas as pd
from collections import defaultdict
from config import INTENT_MAPPING, CFG, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Normalise a raw SwDA utterance:
      1. Strip SwDA annotation noise  { } # [ ] < >
      2. Lowercase
      3. Remove everything except letters and whitespace
      4. Collapse multiple spaces
    """
    # Step 1 – remove SwDA-specific markup characters
    text = re.sub(r"[{}\[\]<>#]", " ", raw)

    # Step 2 – lowercase
    text = text.lower()

    # Step 3 – keep only alphabet + whitespace (removes punctuation, digits)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 – collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCANNER  (directly inspired by your os.walk approach)
# ─────────────────────────────────────────────────────────────────────────────

def process_all_swda_folders(base_path: str = ".") -> pd.DataFrame:
    """
    Recursively walks base_path, reads every CSV that looks like a SwDA file
    (must contain 'act_tag' and 'text' columns), maps tags → intents,
    cleans text, and returns a raw combined DataFrame.
    """
    all_data      = []
    stats         = defaultdict(int)   # for the prep report

    print(f"\n[SCAN] Starting in: {os.path.abspath(base_path)}")
    print("[SCAN] Walking all subfolders for SwDA CSV files...\n")

    for root, _dirs, files in os.walk(base_path):
        for filename in files:
            if not filename.endswith(".csv"):
                continue

            full_path = os.path.join(root, filename)

            try:
                df = pd.read_csv(full_path, dtype=str)   # read all as str → safer

                # ── Normalise column names (your idea — kept it) ─────────────
                df.columns = [c.lower().strip() for c in df.columns]

                # ── Skip if this CSV is not a SwDA file ──────────────────────
                if "act_tag" not in df.columns or "text" not in df.columns:
                    stats["skipped_non_swda"] += 1
                    continue

                stats["files_processed"] += 1

                # ── Row-level processing ──────────────────────────────────────
                for _, row in df.iterrows():
                    raw_tag  = str(row["act_tag"]).strip().lower()
                    raw_text = str(row["text"])

                    # Only keep tags we've explicitly mapped
                    if raw_tag not in INTENT_MAPPING:
                        stats["rows_unknown_tag"] += 1
                        continue

                    cleaned = clean_text(raw_text)

                    # Enforce minimum text length (stricter than your > 1)
                    if len(cleaned) < CFG["min_text_len"]:
                        stats["rows_too_short"] += 1
                        continue

                    all_data.append({
                        "text"  : cleaned,
                        "label" : INTENT_MAPPING[raw_tag],   # 'label' matches model_training.py
                    })
                    stats["rows_collected"] += 1

                # Progress pulse every 50 files (your idea — kept it)
                if stats["files_processed"] % 50 == 0:
                    print(f"  ↳ Processed {stats['files_processed']} files, "
                          f"{stats['rows_collected']} rows so far...", end="\r")

            except Exception as exc:
                # Log the failure but keep going (your resilient approach)
                stats["files_failed"] += 1
                stats[f"error::{filename}"] = str(exc)

    print()  # newline after the \r progress line
    return pd.DataFrame(all_data), dict(stats)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop exact duplicates  (SwDA repeats utterances across files)
    2. Shuffle
    3. Reset index
    """
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    after  = len(df)
    print(f"[POST] Removed {before - after} duplicate rows.")

    df = df.sample(frac=1, random_state=CFG["random_state"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_report(df: pd.DataFrame, stats: dict, path: str) -> None:
    """Write a human-readable data quality summary to a text file."""
    dist = df["label"].value_counts()

    lines = [
        "=" * 60,
        "  DATA PREPARATION REPORT — Machine 1 (Intent)",
        "=" * 60,
        "",
        f"Files processed      : {stats.get('files_processed', 0)}",
        f"Files failed         : {stats.get('files_failed', 0)}",
        f"Files skipped (non-SwDA) : {stats.get('skipped_non_swda', 0)}",
        f"Rows – unknown tag   : {stats.get('rows_unknown_tag', 0)}",
        f"Rows – too short     : {stats.get('rows_too_short', 0)}",
        f"Rows collected       : {stats.get('rows_collected', 0)}",
        f"Rows after dedup     : {len(df)}",
        "",
        "Label distribution:",
        dist.to_string(),
        "",
        f"Min text length (chars) : {df['text'].str.len().min()}",
        f"Max text length (chars) : {df['text'].str.len().max()}",
        f"Mean text length (chars): {df['text'].str.len().mean():.1f}",
        "",
        "Sample rows (first 5):",
        df.head(5).to_string(index=False),
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DATA PREPARATION — Machine 1: Intent Classification")
    print("=" * 60)

    # ── 1. Scan & collect ─────────────────────────────────────────────────────
    raw_df, stats = process_all_swda_folders(base_path=PATHS["swda_root"])

    if raw_df.empty:
        print("\n[ERROR] No valid SwDA data found.")
        print("  Make sure the CSV files are inside subfolders (sw00utt/, etc.)")
        print("  and that each CSV has 'act_tag' and 'text' columns.")
        return

    # ── 2. Post-process ───────────────────────────────────────────────────────
    clean_df = postprocess(raw_df)

    # ── 3. Save processed dataset ─────────────────────────────────────────────
    clean_df.to_csv(PATHS["processed_csv"], index=False)
    print(f"\n[SAVE] Dataset saved → {PATHS['processed_csv']} ({len(clean_df)} rows)")

    # ── 4. Save report ────────────────────────────────────────────────────────
    save_report(clean_df, stats, PATHS["prep_report"])

    print("\n[DONE] data_preparation.py finished. Run model_training.py next.\n")


if __name__ == "__main__":
    main()