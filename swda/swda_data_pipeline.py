"""
=============================================================================
  data_pipeline.py — SwDA Ingestion, Cleaning & Balancing
  Dual-Head Chatbot | Machine 1: Intent Classification

  Run:  python data_pipeline.py
  Out:  processed_intents_AUGMENTED.csv
=============================================================================
"""

import os
import re
import random
import pandas as pd
import nltk
from collections import defaultdict
from nltk.corpus import wordnet
from swda_config import INTENT_MAPPING, CFG, PATHS

TARGET_COUNT = 3000

# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    text = re.sub(r"[{}\[\]<>#]", " ", raw)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ─── Scan & Collect ───────────────────────────────────────────────────────────

def scan_swda_folders(base_path: str) -> pd.DataFrame:
    all_data = []

    for root, _dirs, files in os.walk(base_path):
        for filename in files:
            if not filename.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(root, filename), dtype=str)
                df.columns = [c.lower().strip() for c in df.columns]

                if "act_tag" not in df.columns or "text" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    raw_tag = str(row["act_tag"]).strip().lower()
                    if raw_tag not in INTENT_MAPPING:
                        continue
                    cleaned = clean_text(str(row["text"]))
                    if len(cleaned) < CFG["min_text_len"]:
                        continue
                    all_data.append({"text": cleaned, "label": INTENT_MAPPING[raw_tag]})

            except Exception:
                continue

    return pd.DataFrame(all_data)

# ─── Augmentation ─────────────────────────────────────────────────────────────

def get_synonyms(word: str) -> list[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").replace("-", " ").lower()
            if candidate != word and candidate.isalpha():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(text: str, n: int = 1) -> str:
    words      = text.split()
    if len(words) < 2:
        return text
    new_words  = words.copy()
    candidates = [w for w in set(words) if w.isalpha()]
    random.shuffle(candidates)
    replaced   = 0
    for word in candidates:
        syns = get_synonyms(word)
        if not syns:
            continue
        chosen = random.choice(syns)
        if chosen == word:
            continue
        new_words = [chosen if w == word else w for w in new_words]
        replaced += 1
        if replaced >= n:
            break
    augmented = " ".join(new_words)
    return augmented if augmented != text else text

# ─── Balance ──────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    parts = []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        count  = len(subset)

        if count > target_count:
            parts.append(subset.sample(n=target_count, random_state=CFG["random_state"]))
            print(f"  [DOWN]    '{label}': {count} -> {target_count}")

        elif count < target_count:
            missing      = target_count - count
            source_texts = subset["text"].tolist()
            new_rows     = []
            attempts     = 0
            while len(new_rows) < missing and attempts < missing * 5:
                aug = synonym_replacement(random.choice(source_texts), n=random.randint(1, 2))
                attempts += 1
                if aug != random.choice(source_texts):
                    new_rows.append({"text": aug, "label": label})
            while len(new_rows) < missing:
                new_rows.append({"text": random.choice(source_texts), "label": label})
            parts.append(subset)
            parts.append(pd.DataFrame(new_rows))
            print(f"  [AUGMENT] '{label}': {count} -> {target_count}")

        else:
            parts.append(subset)
            print(f"  [KEEP]    '{label}': {count}")

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=CFG["random_state"])
        .reset_index(drop=True)
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Scanning SwDA folders...")
    df = scan_swda_folders(PATHS["swda_root"])
    if df.empty:
        print("[ERROR] No valid SwDA data found.")
        return

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Collected {before} rows, {len(df)} after dedup.")

    print("Balancing classes:")
    final_df = balance_dataset(df, TARGET_COUNT)

    os.makedirs(os.path.dirname(PATHS["processed_csv"]), exist_ok=True)
    final_df.to_csv(PATHS["processed_csv"], index=False)
    print(f"\n[DONE] {len(final_df)} rows -> '{PATHS['processed_csv']}'")


if __name__ == "__main__":
    main()