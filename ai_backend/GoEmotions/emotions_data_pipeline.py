"""
=============================================================================
  emotion_data.py — GoEmotions Processing & Balancing
  Dual-Head Chatbot | Machine 2: Emotion Classification

  Run:  python emotion_data.py
  Out:  processed_emotions_train.csv (balanced)
        processed_emotions_dev.csv
        processed_emotions_test.csv
=============================================================================
"""

import pandas as pd
import json
import re
import os
import random
from nltk.corpus import wordnet

TARGET_COUNT = 10000

# ─── Config ───────────────────────────────────────────────────────────────────

try:
    with open('C:\\Finals_Project\\ai_backend\\GoEmotions\\emotion_data\\emotion_mapping.json', 'r') as f:
        PROJECT_EMOTION_MAPPING = json.load(f)
except FileNotFoundError:
    print("CRITICAL ERROR: 'emotion_mapping.json' not found.")
    exit()

try:
    with open('C:\\Finals_Project\\ai_backend\\GoEmotions\\emotion_data\\emotions.txt', 'r') as f:
        idtolabel = {i: label for i, label in enumerate(f.read().splitlines())}
except FileNotFoundError:
    print("CRITICAL ERROR: 'emotions.txt' not found.")
    exit()

# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s?!.,]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ─── Label Mapping ────────────────────────────────────────────────────────────

def get_project_emotion(label_str):
    if pd.isna(label_str):
        return 'neutral'
    ids            = [int(x) for x in label_str.split(',')]
    original_names = [idtolabel[i] for i in ids if i in idtolabel]
    mapped_labels  = [PROJECT_EMOTION_MAPPING.get(name, 'neutral') for name in original_names]
    non_neutral    = [label for label in mapped_labels if label != 'neutral']
    return non_neutral[0] if non_neutral else 'neutral'

# ─── Process TSV ─────────────────────────────────────────────────────────────

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return None
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label_ids', 'id'])
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['emotion']      = df['label_ids'].apply(get_project_emotion)
    return df[['cleaned_text', 'emotion']]

# ─── Augmentation ─────────────────────────────────────────────────────────────

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").replace("-", " ").lower()
            if candidate != word and candidate.isalpha():
                synonyms.add(candidate)
    return list(synonyms)

def synonym_replacement(text, n=1):
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

def balance_dataset(df, target_count):
    parts = []
    for label in sorted(df["emotion"].unique()):
        subset = df[df["emotion"] == label]
        count  = len(subset)

        if count > target_count:
            parts.append(subset.sample(n=target_count, random_state=42))
            print(f"  [DOWN]    '{label}': {count} -> {target_count}")

        elif count < target_count:
            missing      = target_count - count
            source_texts = subset["cleaned_text"].tolist()
            new_rows     = []
            attempts     = 0
            while len(new_rows) < missing and attempts < missing * 5:
                aug = synonym_replacement(random.choice(source_texts), n=random.randint(1, 2))
                attempts += 1
                if aug != random.choice(source_texts):
                    new_rows.append({"cleaned_text": aug, "emotion": label})
            while len(new_rows) < missing:
                new_rows.append({"cleaned_text": random.choice(source_texts), "emotion": label})
            parts.append(subset)
            parts.append(pd.DataFrame(new_rows))
            print(f"  [AUGMENT] '{label}': {count} -> {target_count}")

        else:
            parts.append(subset)
            print(f"  [KEEP]    '{label}': {count}")

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    files = {
        'train': 'ai_backend/GoEmotions/emotion_data/train.tsv',
        'dev'  : 'ai_backend/GoEmotions/emotion_data/dev.tsv',
        'test' : 'ai_backend/GoEmotions/emotion_data/test.tsv',
    }

    for split, path in files.items():
        print(f"\n[{split.upper()}] Processing...")
        df = process_file(path)
        if df is None:
            continue

        # Balance only the training set
        if split == 'train':
            print(f"  Balancing classes:")
            df = balance_dataset(df, TARGET_COUNT)

        out = f'ai_backend/GoEmotions/emotion_data/processed_emotions_{split}.csv'
        df.to_csv(out, index=False)
        print(f"  Saved -> '{out}' ({len(df)} rows)")

    print("\n[DONE] emotion_data.py finished. Run model_training.py next.")


if __name__ == "__main__":
    main()