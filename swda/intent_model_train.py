"""
train_intent_model.py
Trains the intent classifier and saves fresh artefacts

Outputs:
    intent_model.keras
    tokenizer.pkl
    label_encoder.pkl
"""

import os, pickle, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models                 import Sequential
from tensorflow.keras.layers                 import (
    Embedding, Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

# ── Config (mirrors your original config.py) ─────────────────────────────────
MAX_VOCAB   = 10_000
MAX_SEQ_LEN = 50
EMB_DIM     = 64
LSTM_UNITS  = 128
DENSE_UNITS = 64
DROPOUT     = 0.4
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15

DATA_PATH   = "swda\processed_intents_AUGMENTED.csv"
MODEL_OUT   = "intent_model.keras"
TOK_OUT     = "tokenizer.pkl"
LE_OUT      = "label_encoder.pkl"

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\n[1] Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
print(f"    {len(df)} samples | labels: {sorted(df['label'].unique())}")

# ── Tokenise ──────────────────────────────────────────────────────────────────
print("\n[2] Tokenising ...")
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
seqs = tokenizer.texts_to_sequences(df["text"])
X    = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

# ── Encode labels ─────────────────────────────────────────────────────────────
le  = LabelEncoder()
y   = le.fit_transform(df["label"])
num_classes = len(le.classes_)
print(f"    vocab={len(tokenizer.word_index)} | classes={list(le.classes_)}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                           random_state=42, stratify=y)
val_frac = VAL_SIZE / (1 - TEST_SIZE)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=val_frac,
                                             random_state=42, stratify=y_tr)
print(f"    train={len(X_tr)} | val={len(X_val)} | test={len(X_te)}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\n[3] Building model ...")
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index))

model = Sequential([
    Embedding(input_dim=vocab_size + 1, output_dim=EMB_DIM,
              input_length=MAX_SEQ_LEN, name="embedding"),
    Bidirectional(LSTM(LSTM_UNITS), name="bilstm"),
    Dense(DENSE_UNITS, activation="relu", name="dense"),
    Dropout(DROPOUT, name="dropout"),
    Dense(num_classes, activation="softmax", name="output"),
], name="IntentModel")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[4] Training ...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_loss",
                    save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]
model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=callbacks, verbose=1)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[5] Evaluating ...")
loss, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"    Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")
y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
print(classification_report(y_te, y_pred, target_names=le.classes_))

# ── Save ──────────────────────────────────────────────────────────────────────
print("\n[6] Saving artefacts ...")
model.save(MODEL_OUT)
print(f"    ✓ {MODEL_OUT}")

with open(TOK_OUT, "wb") as f: pickle.dump(tokenizer, f)
print(f"    ✓ {TOK_OUT}")

with open(LE_OUT, "wb") as f: pickle.dump(le, f)
print(f"    ✓ {LE_OUT}")

print("\n✓ Intent model ready.\n")
