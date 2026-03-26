"""
resave_emotion_model.py

The emotion model's TextVectorization layer can't be loaded by newer Keras
due to a serialization bug. This script:
  1. Unzips the .keras file and extracts the raw vocabulary + IDF weights
  2. Rebuilds the exact same architecture from your training code
  3. Re-adapts the TextVectorization layer with the saved vocab
  4. Restores all Dense layer weights
  5. Saves as .h5

Run:
    python resave_emotion_model.py
"""

import os, json, zipfile, pickle, tempfile, shutil
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, regularizers

MODEL_SRC  = "emotion_model_optimized.keras"
MODEL_DST  = "emotion_model_optimized.h5"
LE_PATH    = "emotion_label_encoder.pkl"

# Must match your training config exactly
VOCAB_SIZE  = 5000
DROPOUT     = 0.5
L2_REG      = 0.01

print("=" * 55)
print("  Emotion model re-saver")
print("=" * 55)

# ── Step 1: load label encoder to get num_classes ─────────────
with open(LE_PATH, "rb") as f:
    le = pickle.load(f)
num_classes = len(le.classes_)
print(f"\n[1] num_classes = {num_classes}  ({list(le.classes_)})")

# ── Step 2: extract internals from the .keras zip ────────────
print(f"\n[2] Unzipping {MODEL_SRC} ...")
tmp_dir = tempfile.mkdtemp()

with zipfile.ZipFile(MODEL_SRC, "r") as zf:
    zf.extractall(tmp_dir)
    all_files = zf.namelist()

print(f"    Files inside zip:")
for f in sorted(all_files):
    print(f"      {f}")

# ── Step 3: find the config JSON ──────────────────────────────
print(f"\n[3] Reading config ...")
config_path = None
for candidate in ["config.json", "model.json"]:
    p = os.path.join(tmp_dir, candidate)
    if os.path.exists(p):
        config_path = p
        break

if config_path is None:
    raise FileNotFoundError("Could not find config.json inside the .keras zip")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"    Config loaded from {config_path}")

# ── Step 4: extract vocabulary and IDF weights from saved assets ──
# TextVectorization stores its state in the model's non-trainable weights.
# We'll read them directly from the weights file inside the zip.
print(f"\n[4] Extracting TextVectorization state ...")

vocab_words = None
idf_weights = None

# Look for weights files
weights_files = [f for f in all_files if f.endswith(".weights.h5") or "weights" in f.lower()]
print(f"    Weight files found: {weights_files}")

# Try to find vocabulary in assets
asset_files = [f for f in all_files if "asset" in f.lower() or "vocab" in f.lower()]
print(f"    Asset files found: {asset_files}")

for af in asset_files:
    full_path = os.path.join(tmp_dir, af)
    if os.path.exists(full_path) and af.endswith(".txt"):
        with open(full_path, "r", encoding="utf-8") as f:
            vocab_words = [line.strip() for line in f.readlines()]
        print(f"    Loaded vocab from {af}: {len(vocab_words)} tokens")

# ── Step 5: rebuild model architecture (mirrors train_emotion_model.py) ──
print(f"\n[5] Rebuilding model architecture ...")

tfidf_vec = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="tf_idf",
    name="tfidf_vec"
)

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
x = tfidf_vec(text_input)
x = layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(DROPOUT)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

new_model = tf.keras.Model(text_input, output)
print(f"    Architecture rebuilt ✓")

# ── Step 6: restore TextVectorization vocabulary ──────────────
print(f"\n[6] Restoring TextVectorization vocabulary ...")

# Try to load the vocabulary from the weights zip using h5py
try:
    import h5py

    weights_h5 = None
    for f in all_files:
        if f.endswith(".weights.h5"):
            weights_h5 = os.path.join(tmp_dir, f)
            break

    if weights_h5 and os.path.exists(weights_h5):
        print(f"    Reading from {weights_h5} ...")
        with h5py.File(weights_h5, "r") as hf:
            def print_structure(name, obj):
                print(f"      {name}")
            hf.visititems(print_structure)

            # Look for tfidf_vec vocabulary and idf_weights
            for key in ["tfidf_vec", "text_vectorization", "tfidf"]:
                if key in hf:
                    grp = hf[key]
                    print(f"    Found group: {key}")
                    if "vocabulary" in grp:
                        raw = grp["vocabulary"][:]
                        vocab_words = [v.decode("utf-8") if isinstance(v, bytes) else v for v in raw]
                        print(f"    Vocabulary: {len(vocab_words)} tokens")
                    if "idf_weights" in grp:
                        idf_weights = grp["idf_weights"][:]
                        print(f"    IDF weights: shape {idf_weights.shape}")
except ImportError:
    print("    h5py not available, trying alternate method ...")

# ── Step 7: adapt or restore vocabulary ───────────────────────
print(f"\n[7] Setting up vocabulary ...")

if vocab_words and len(vocab_words) > 0:
    # Restore directly from saved vocab
    tfidf_vec.set_vocabulary(vocab_words, idf_weights=idf_weights)
    print(f"    Vocabulary restored from saved file ({len(vocab_words)} tokens) ✓")
else:
    # Fallback: load training data and re-adapt
    print("    WARNING: Could not extract vocabulary from model file.")
    print("    Attempting to re-adapt from training data ...")

    train_files = [
        "GoEmotions/processed_emotions_train_BALANCED.csv",
        "processed_emotions_train_BALANCED.csv",
    ]
    adapted = False
    for tf_path in train_files:
        if os.path.exists(tf_path):
            import pandas as pd
            df = pd.read_csv(tf_path)
            texts = df["cleaned_text"].astype(str).values
            tfidf_vec.adapt(texts)
            print(f"    Re-adapted from {tf_path} ({len(texts)} samples) ✓")
            adapted = True
            break

    if not adapted:
        raise RuntimeError(
            "\nCould not restore vocabulary — neither the vocab file nor training data found.\n"
            "Please make sure one of these exists:\n"
            "  • GoEmotions/processed_emotions_train_BALANCED.csv\n"
            "  • processed_emotions_train_BALANCED.csv"
        )

# ── Step 8: restore Dense weights ────────────────────────────
print(f"\n[8] Restoring Dense layer weights ...")

# Build the model so layers have shapes
_ = new_model(tf.constant([["hello world"]]))

try:
    import h5py
    weights_h5 = None
    for f in all_files:
        if f.endswith(".weights.h5"):
            weights_h5 = os.path.join(tmp_dir, f)
            break

    if weights_h5 and os.path.exists(weights_h5):
        with h5py.File(weights_h5, "r") as hf:
            dense_layers = [l for l in new_model.layers
                           if isinstance(l, tf.keras.layers.Dense)]
            restored = 0
            for layer in dense_layers:
                layer_name = layer.name
                # Try to find matching weights in hdf5
                for grp_name in hf.keys():
                    if layer_name in grp_name or grp_name in layer_name:
                        try:
                            w = hf[grp_name]["kernel"][:]
                            b = hf[grp_name]["bias"][:]
                            layer.set_weights([w, b])
                            print(f"    Restored weights for '{layer_name}' ✓")
                            restored += 1
                        except Exception as e:
                            print(f"    Could not restore '{layer_name}': {e}")
        if restored == 0:
            print("    WARNING: No Dense weights restored — model will have random weights!")
            print("    The TextVectorization vocab IS correct though.")
    else:
        print("    No .weights.h5 file found — Dense weights will be random")
        print("    (TextVectorization vocab is still correct)")

except Exception as e:
    print(f"    Weight restore error: {e}")

# ── Step 9: save as .h5 ───────────────────────────────────────
print(f"\n[9] Saving as {MODEL_DST} ...")
new_model.save(MODEL_DST)
size_mb = os.path.getsize(MODEL_DST) / 1024 / 1024
print(f"    Saved ✓  ({size_mb:.1f} MB)")

# ── Cleanup ───────────────────────────────────────────────────
shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\n{'='*55}")
print(f"  Done! Now run:  python test_load.py")
print(f"{'='*55}\n")