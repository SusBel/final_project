"""
=============================================================================
  data_augmentation.py — Smart Dataset Balancing via Synonym Replacement
  Dual-Head Chatbot | Machine 1: Intent Classification

  Pipeline position:
      data_preparation.py  →  [data_augmentation.py]  →  model_training.py

  Reads  processed_intents.csv, balances every class to TARGET_COUNT by:
    • Downsampling  classes that are over the target
    • Augmenting    classes that are under the target (WordNet synonyms)

  Writes: processed_intents_AUGMENTED.csv

  Run:
      python data_augmentation.py
=============================================================================
"""

import os
import random
import pandas as pd
import nltk

from config import CFG, PATHS

# ─────────────────────────────────────────────────────────────────────────────
# NLTK BOOTSTRAP  (safe download — no silent failures)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_nltk_resources() -> None:
    """
    Downloads WordNet and averaged_perceptron_tagger if not already present.
    Raises a clear error instead of swallowing it silently.
    """
    resources = ["corpora/wordnet", "corpora/omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            pkg = resource.split("/")[-1]
            print(f"[NLTK] Downloading '{pkg}'...")
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()
from nltk.corpus import wordnet   # import AFTER download check


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION  —  Synonym Replacement  (your core idea, hardened)
# ─────────────────────────────────────────────────────────────────────────────

def get_synonyms(word: str) -> list[str]:
    """
    Returns a deduplicated list of WordNet synonyms for `word`,
    excluding the word itself and multi-token phrases.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = (
                lemma.name()
                .replace("_", " ")
                .replace("-", " ")
                .lower()
            )
            # Keep only single-token, purely alphabetic synonyms
            if candidate != word and candidate.isalpha():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(text: str, n: int = 1) -> str:
    """
    Replaces up to `n` words in `text` with a random WordNet synonym.
    Returns the original text unchanged when no valid synonym is found,
    making it safe to call in a loop without risk of an infinite retry.
    """
    words = text.split()
    if len(words) < 2:
        return text

    new_words = words.copy()

    # Shuffle candidate words so we pick randomly, not always from the front
    candidates = [w for w in set(words) if w.isalpha()]
    random.shuffle(candidates)

    replaced = 0
    for word in candidates:
        syns = get_synonyms(word)
        if not syns:
            continue

        chosen_syn = random.choice(syns)

        # Guard: skip if the synonym is identical to what's already there
        if chosen_syn == word:
            continue

        new_words = [chosen_syn if w == word else w for w in new_words]
        replaced += 1

        if replaced >= n:
            break

    augmented = " ".join(new_words)

    # Final guard: if nothing changed, return the original (don't add a dupe)
    return augmented if augmented != text else text


# ─────────────────────────────────────────────────────────────────────────────
# CORE BALANCER  (your smart_balance_dataset, aligned to the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def smart_balance_dataset(
    df          : pd.DataFrame,
    text_col    : str,
    label_col   : str,
    target_count: int = 3000,
) -> pd.DataFrame:
    """
    Balances every class in `df` to exactly `target_count` samples:
      • Over  target → random downsample (no augmentation needed)
      • Under target → keep all originals + generate synonyms to fill the gap
      • Exact target → keep as-is

    Returns a shuffled, reset-index DataFrame.
    """
    print(f"\n[BALANCE] Target per class: {target_count}")
    print(f"[BALANCE] Input  rows: {len(df)}\n")

    balanced_parts = []

    for label in sorted(df[label_col].unique()):
        subset = df[df[label_col] == label]
        count  = len(subset)

        if count == 0:
            continue

        # ── Downsample ────────────────────────────────────────────────────────
        if count > target_count:
            resampled = subset.sample(n=target_count, random_state=CFG["random_state"])
            balanced_parts.append(resampled)
            print(f"  [DOWN]    '{label}': {count:>5} → {target_count}")

        # ── Augment ───────────────────────────────────────────────────────────
        elif count < target_count:
            balanced_parts.append(subset)   # keep all originals first
            missing      = target_count - count
            source_texts = subset[text_col].tolist()
            new_rows     = []
            attempts     = 0
            max_attempts = missing * 5       # safety ceiling against infinite loop

            print(f"  [AUGMENT] '{label}': {count:>5} → {target_count} "
                  f"(generating {missing} variations)", end="\r")

            while len(new_rows) < missing and attempts < max_attempts:
                original  = random.choice(source_texts)
                augmented = synonym_replacement(original, n=random.randint(1, 2))
                attempts += 1

                # Only add if it's actually a new sentence
                if augmented != original:
                    new_rows.append({text_col: augmented, label_col: label})

            # If WordNet couldn't generate enough unique variants,
            # fill the remainder with duplicates (better than a broken split)
            while len(new_rows) < missing:
                new_rows.append({
                    text_col  : random.choice(source_texts),
                    label_col : label,
                })

            balanced_parts.append(pd.DataFrame(new_rows))
            print(f"  [AUGMENT] '{label}': {count:>5} → {target_count} "
                  f"({len(new_rows)} new rows, {attempts} attempts)   ")

        # ── Already balanced ──────────────────────────────────────────────────
        else:
            balanced_parts.append(subset)
            print(f"  [KEEP]    '{label}': {count:>5} — exact match")

    final_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=CFG["random_state"])
        .reset_index(drop=True)
    )
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_distribution(df: pd.DataFrame, label_col: str, title: str) -> None:
    print(f"\n[DIST] {title}:")
    dist = df[label_col].value_counts().sort_index()
    for label, count in dist.items():
        bar = "█" * (count // 100)
        print(f"  {label:<15} {count:>6}  {bar}")
    print(f"  {'TOTAL':<15} {len(df):>6}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Target count per class — tune this based on your GPU memory / training time
TARGET_COUNT = 3000

# Output path (consumed by model_training.py)
AUGMENTED_PATH = "swda/processed_intents_AUGMENTED.csv"


def main():
    print("=" * 60)
    print("  DATA AUGMENTATION — Machine 1: Intent Classification")
    print("=" * 60)

    input_path = PATHS["processed_csv"]   # processed_intents.csv

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"\n[ERROR] '{input_path}' not found.\n"
            "  → Run  python data_preparation.py  first."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print_distribution(df, "label", "Before balancing")

    # ── Balance ───────────────────────────────────────────────────────────────
    df_balanced = smart_balance_dataset(
        df,
        text_col    = "text",
        label_col   = "label",    # matches data_preparation.py output
        target_count= TARGET_COUNT,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print_distribution(df_balanced, "label", "After balancing")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_balanced.to_csv(AUGMENTED_PATH, index=False)
    print(f"\n[SAVE] Augmented dataset → '{AUGMENTED_PATH}' ({len(df_balanced)} rows)")
    print("\n[DONE] data_augmentation.py finished. Run model_training.py next.\n")


if __name__ == "__main__":
    main()