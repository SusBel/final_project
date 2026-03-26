"""
test_load.py — Run this before starting the server.
    python test_load.py
Prints PASS/FAIL for every component. Fix any failures before running uvicorn.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("=" * 50)
print("  PersonaAI — Pre-flight check")
print("=" * 50)

errors = []

# ── 1. Files ──────────────────────────────────────────────────
REQUIRED = [
    "intent_model.keras",
    "tokenizer.pkl",
    "label_encoder.pkl",
    "emotion_model",                  # SavedModel folder
    "emotion_label_encoder.pkl",
    "golden_set_logic_stateful.csv",
]
print("\n[1] Checking files...")
for f in REQUIRED:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) / 1024 if exists and os.path.isfile(f) else 0
    label  = f"({size:.0f} KB)" if size else "(folder)" if os.path.isdir(f) else "MISSING"
    mark   = "✓" if exists else "✗"
    print(f"    {mark}  {f}  {label}")
    if not exists:
        errors.append(f"Missing: {f}")

# ── 2. TF import ──────────────────────────────────────────────
print("\n[2] Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"    ✓  tensorflow {tf.__version__}")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print(f"    ✓  pad_sequences")
except Exception as e:
    errors.append(f"TF import failed: {e}")
    print(f"    ✗  {e}")

# ── 3. Load models ────────────────────────────────────────────
print("\n[3] Loading models...")
if not errors:
    try:
        intent_model = tf.keras.models.load_model("intent_model.keras", compile=False)
        print(f"    ✓  intent_model  — output: {intent_model.output_shape}")
    except Exception as e:
        errors.append(f"intent_model failed: {e}")
        print(f"    ✗  intent_model: {e}")

    try:
        emotion_model = tf.keras.models.load_model("emotion_model", compile=False)
        print(f"    ✓  emotion_model — output: {emotion_model.output_shape}")
    except Exception as e:
        errors.append(f"emotion_model failed: {e}")
        print(f"    ✗  emotion_model: {e}")

# ── 4. Load pkl files ─────────────────────────────────────────
print("\n[4] Loading pkl artefacts...")
import pickle
for fname in ["tokenizer.pkl", "label_encoder.pkl", "emotion_label_encoder.pkl"]:
    if os.path.exists(fname):
        try:
            with open(fname, "rb") as f:
                obj = pickle.load(f)
            extra = f"— classes: {list(obj.classes_)}" if hasattr(obj, "classes_") else ""
            print(f"    ✓  {fname}  {extra}")
        except Exception as e:
            errors.append(f"{fname}: {e}")
            print(f"    ✗  {fname}: {e}")

# ── 5. Logic CSV ──────────────────────────────────────────────
print("\n[5] Loading logic CSV...")
if os.path.exists("golden_set_logic_stateful.csv"):
    try:
        import pandas as pd
        df = pd.read_csv("golden_set_logic_stateful.csv")
        print(f"    ✓  {len(df)} rows | actions: {sorted(df['expected_response'].unique())}")
    except Exception as e:
        errors.append(f"CSV: {e}")
        print(f"    ✗  {e}")

# ── 6. Quick end-to-end prediction ───────────────────────────
print("\n[6] Quick prediction test...")
if not errors:
    try:
        import numpy as np
        # Intent
        with open("tokenizer.pkl", "rb") as f: tok = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f: le_i = pickle.load(f)
        seq = tok.texts_to_sequences(["my internet is not working"])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50, padding="post")
        intent = le_i.inverse_transform([np.argmax(intent_model.predict(pad, verbose=0))])[0]
        print(f"    ✓  Intent prediction: '{intent}'")

        # Emotion
        with open("emotion_label_encoder.pkl", "rb") as f: le_e = pickle.load(f)
        proba  = emotion_model.predict(np.array([["my internet is not working"]]), verbose=0)
        emotion = le_e.inverse_transform([np.argmax(proba)])[0]
        print(f"    ✓  Emotion prediction: '{emotion}'")
    except Exception as e:
        errors.append(f"Prediction test: {e}")
        print(f"    ✗  {e}")

# ── Result ────────────────────────────────────────────────────
print("\n" + "=" * 50)
if errors:
    print(f"  ✗  {len(errors)} ISSUE(S):")
    for e in errors: print(f"     • {e}")
    sys.exit(1)
else:
    print("  ✓  ALL CHECKS PASSED — ready to start the server!")
    print("=" * 50)
    print("\n  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000\n")
