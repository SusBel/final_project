"""
=============================================================================
  model_training.py — Intent Classification Model Training
  Dual-Head Chatbot | Machine 1: Intent Classification

  Reads processed_intents.csv produced by data_preparation.py,
  builds a BiLSTM classifier, trains it with Keras callbacks,
  evaluates on the held-out test set, and saves all artefacts
  needed by the inference server.

  Run:
      python model_training.py
  Requires:
      processed_intents.csv  (output of data_preparation.py)
  Outputs:
      intent_model.keras     — best model weights
      tokenizer.pkl          — fitted Keras Tokenizer
      label_encoder.pkl      — fitted LabelEncoder
      confusion_matrix.png
      training_history.png
=============================================================================
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models                 import Sequential
from tensorflow.keras.layers                 import (
    Embedding, LSTM, Bidirectional,
    GlobalAveragePooling1D, Dense, Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from config import CFG, PATHS, INTENT_LABELS

warnings.filterwarnings("ignore")
tf.random.set_seed(CFG["random_state"])
np.random.seed(CFG["random_state"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 · LOAD  (reads the CSV that data_preparation.py produced)
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the cleaned dataset.  Expects columns: ['text', 'label'].

    Priority:
      1. processed_intents_AUGMENTED.csv  (if data_augmentation.py was run)
      2. processed_intents.csv            (raw output of data_preparation.py)

    Raises a clear error if neither file exists.
    """
    import os
    augmented_path = "processed_intents_AUGMENTED.csv"
    if os.path.exists(augmented_path):
        print(f"[DATA] Augmented dataset found — loading '{augmented_path}'")
        csv_path = augmented_path
    elif not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Neither '{augmented_path}' nor '{csv_path}' found.\n"
            "  → Run  python data_preparation.py  first,\n"
            "  → then optionally python data_augmentation.py  for balancing."
        )

    df = pd.read_csv(csv_path)

    # Validate required columns
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"[ERROR] Missing column '{col}' in {csv_path}.")

    df = df.dropna(subset=["text", "label"])
    print(f"[DATA] Loaded {len(df)} rows from '{csv_path}'")
    print(f"[DATA] Label distribution:\n{df['label'].value_counts()}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 · PREPROCESSING  (tokenisation + label encoding + split)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_train, X_val, X_test   — padded integer sequences
        y_train, y_val, y_test   — integer class indices
        tokenizer                — fitted Keras Tokenizer  (save → pkl)
        le                       — fitted LabelEncoder     (save → pkl)
        num_classes              — number of unique intents
    """
    # ── 2a. Tokenise text ────────────────────────────────────────────────────
    tokenizer = Tokenizer(
        num_words=CFG["max_vocab_size"],
        oov_token="<OOV>",
    )
    tokenizer.fit_on_texts(df["text"])

    sequences = tokenizer.texts_to_sequences(df["text"])
    X = pad_sequences(
        sequences,
        maxlen    = CFG["max_seq_len"],
        padding   = "post",
        truncating= "post",
    )

    # ── 2b. Encode labels ─────────────────────────────────────────────────────
    le  = LabelEncoder()
    y   = le.fit_transform(df["label"])
    num_classes = len(le.classes_)

    print(f"[PREP] {num_classes} classes: {list(le.classes_)}")
    print(f"[PREP] Vocabulary size: {len(tokenizer.word_index)}\n")

    # ── 2c. Stratified train / val / test split ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CFG["test_size"],
        random_state = CFG["random_state"],
        stratify     = y,
    )
    val_frac = CFG["val_size"] / (1 - CFG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size    = val_frac,
        random_state = CFG["random_state"],
        stratify     = y_train,
    )

    print(f"[PREP] Split → train: {len(X_train)} | "
          f"val: {len(X_val)} | test: {len(X_test)}\n")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, le, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 · MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, num_classes: int) -> tf.keras.Model:
    """
    Embedding  →  BiLSTM (or GlobalAveragePooling1D)
               →  Dense(ReLU)  →  Dropout  →  Dense(Softmax)

    Toggle CFG['use_bilstm'] to switch the sequence encoder.
    """
    model = Sequential(name="IntentClassifier_Machine1")

    model.add(Embedding(
        input_dim   = vocab_size + 1,   # +1 for the 0 padding index
        output_dim  = CFG["embedding_dim"],
        input_length= CFG["max_seq_len"],
        name        = "token_embedding",
    ))

    if CFG["use_bilstm"]:
        model.add(Bidirectional(
            LSTM(CFG["lstm_units"], return_sequences=False),
            name="bi_lstm",
        ))
    else:
        model.add(GlobalAveragePooling1D(name="global_avg_pool"))

    model.add(Dense(CFG["dense_units"], activation="relu",    name="dense_relu"))
    model.add(Dropout(CFG["dropout_rate"],                    name="dropout"))
    model.add(Dense(num_classes,         activation="softmax", name="output"))

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG["learning_rate"]),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 · TRAINING LOOP WITH CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks() -> list:
    """
    Three callbacks as specified in the project document:
      • EarlyStopping      – stops when val_loss stagnates; restores best weights
      • ModelCheckpoint    – saves the model only when val_loss improves
      • ReduceLROnPlateau  – halves LR after patience epochs of no improvement
    """
    return [
        EarlyStopping(
            monitor             = "val_loss",
            patience            = CFG["patience_es"],
            restore_best_weights= True,
            verbose             = 1,
        ),
        ModelCheckpoint(
            filepath      = PATHS["model"],
            monitor       = "val_loss",
            save_best_only= True,
            verbose       = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = CFG["lr_factor"],
            patience = CFG["patience_lr"],
            min_lr   = CFG["min_lr"],
            verbose  = 1,
        ),
    ]


def train(model, X_train, y_train, X_val, y_val):
    print("\n[TRAIN] Starting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = CFG["epochs"],
        batch_size      = CFG["batch_size"],
        callbacks       = get_callbacks(),
        verbose         = 1,
    )
    print("\n[TRAIN] Training complete.\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 · EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le: LabelEncoder) -> None:
    """
    Prints test loss/accuracy, a full classification report,
    and saves a seaborn confusion matrix PNG.
    """
    # ── 5a. Keras metrics ─────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[EVAL] Test Loss     : {loss:.4f}")
    print(f"[EVAL] Test Accuracy : {acc:.4f}\n")

    # ── 5b. Classification report ─────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("[EVAL] Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

   
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 · SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(model, tokenizer, le: LabelEncoder) -> None:
    """
    Persists the three artefacts the inference server needs to load:
      1. Model      → .keras  (already written by ModelCheckpoint; re-saved here)
      2. Tokenizer  → .pkl
      3. LabelEncoder → .pkl
    """
    model.save(PATHS["model"])
    print(f"[SAVE] Model        → {PATHS['model']}")

    with open(PATHS["tokenizer"], "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[SAVE] Tokenizer    → {PATHS['tokenizer']}")

    with open(PATHS["label_encoder"], "wb") as f:
        pickle.dump(le, f)
    print(f"[SAVE] LabelEncoder → {PATHS['label_encoder']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE SMOKE-TEST  (verifies the saved artefacts work end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """
    Loads artefacts from disk and runs a quick prediction on 5 sample sentences.
    Demonstrates the exact API the inference server will use.
    """
    import re

    def _clean(text: str) -> str:
        text = re.sub(r"[{}\[\]<>#]", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    model = tf.keras.models.load_model(PATHS["model"])
    with open(PATHS["tokenizer"],    "rb") as f: tok = pickle.load(f)
    with open(PATHS["label_encoder"],"rb") as f: enc = pickle.load(f)

    samples = [
        "What time does the store open?",
        "Please send me the updated report.",
        "I think the current approach is inefficient.",
        "Alright, thanks for your help, goodbye.",
        "Uh huh, I see.",
    ]

    cleaned  = [_clean(s) for s in samples]
    seqs     = tok.texts_to_sequences(cleaned)
    padded   = pad_sequences(seqs, maxlen=CFG["max_seq_len"],
                             padding="post", truncating="post")
    proba    = model.predict(padded, verbose=0)
    labels   = enc.inverse_transform(np.argmax(proba, axis=1))
    confs    = np.max(proba, axis=1)

    results = pd.DataFrame({
        "text"            : samples,
        "predicted_intent": labels,
        "confidence"      : confs.round(4),
    })
    print("[SMOKE TEST] Results on 5 unseen sentences:\n")
    print(results.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MACHINE 1 — Intent Classification | Training")
    print("=" * 60, "\n")

    # 1. Load
    df = load_processed_data(PATHS["processed_csv"])

    # 2. Preprocess
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     tokenizer, le, num_classes) = preprocess(df)

    # 3. Build
    vocab_size = min(CFG["max_vocab_size"], len(tokenizer.word_index))
    model      = build_model(vocab_size, num_classes)

    # 4. Train
    history = train(model, X_train, y_train, X_val, y_val)
    plot_history(history)

    # 5. Evaluate
    evaluate(model, X_test, y_test, le)

    # 6. Save
    save_artefacts(model, tokenizer, le)

    # 7. Smoke test
    smoke_test()

    print("\n[DONE] model_training.py finished successfully.\n")


if __name__ == "__main__":
    main()