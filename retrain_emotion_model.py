"""
retrain_emotion_model.py
Retrains the emotion classifier and saves fresh artefacts
compatible with your current Keras/TF installation.

Run:
    python retrain_emotion_model.py

Outputs:
    emotion_model.keras
    emotion_label_encoder.pkl
"""

import os, pickle, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics        import classification_report

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Config (mirrors your original train_emotion_model.py) ────────────────────
TRAIN_FILE  = "GoEmotions/processed_emotions_train_BALANCED.csv"
DEV_FILE    = "GoEmotions/processed_emotions_dev.csv"
TEST_FILE   = "GoEmotions/processed_emotions_test.csv"

MODEL_OUT   = "emotion_model"            # SavedModel folder (no extension)
LE_OUT      = "emotion_label_encoder.pkl"

VOCAB_SIZE  = 5000
EPOCHS      = 60
BATCH_SIZE  = 128

# ── Load data ─────────────────────────────────────────────────────────────────
def load_split(path):
    print(f"    Loading {path} ...")
    df = pd.read_csv(path)
    return df["cleaned_text"].astype(str).values, df["emotion"].values

print("\n[1] Loading data ...")
X_train, y_train = load_split(TRAIN_FILE)
X_val,   y_val   = load_split(DEV_FILE)
X_test,  y_test  = load_split(TEST_FILE)

# ── Encode labels ─────────────────────────────────────────────────────────────
print("\n[2] Encoding labels ...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)
num_classes = len(le.classes_)
print(f"    {num_classes} classes: {list(le.classes_)}")

y_train_cat = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val_enc,   num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test_enc,  num_classes)

with open(LE_OUT, "wb") as f: pickle.dump(le, f)
print(f"    ✓ label encoder saved → {LE_OUT}")

# ── Build TF-IDF vectoriser ───────────────────────────────────────────────────
print("\n[3] Adapting TextVectorization (TF-IDF) ...")
tfidf_vec = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="tf_idf",
    name="tfidf_vec",
)
tfidf_vec.adapt(X_train)
print(f"    Vocabulary size: {len(tfidf_vec.get_vocabulary())}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\n[4] Building model ...")
text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
x = tfidf_vec(text_input)
x = layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(0.01), name="dense_relu")(x)
x = layers.Dropout(0.5, name="dropout")(x)
output = layers.Dense(num_classes, activation="softmax", name="output")(x)

model = tf.keras.Model(text_input, output, name="EmotionModel")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)
print(f"    Total params: {model.count_params():,}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[5] Training ...")
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                      patience=4, min_lr=1e-6, mode="max", verbose=1),
]
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, verbose=1,
)
best_val = max(history.history["val_accuracy"])
print(f"\n    Best val_accuracy: {best_val:.2%}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[6] Evaluating ...")
_, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"    Test accuracy: {acc:.2%}")
y_pred = le.inverse_transform(np.argmax(model.predict(X_test, verbose=0), axis=1))
print(classification_report(y_test, y_pred))

# ── Save ──────────────────────────────────────────────────────────────────────
print("\n[7] Saving model ...")
model.save(MODEL_OUT, save_format="tf")   # SavedModel format — required for TextVectorization
print(f"    ✓ {MODEL_OUT}/  (SavedModel folder)")
print("\n✓ Emotion model ready.\n")