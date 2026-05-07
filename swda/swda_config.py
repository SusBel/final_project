# swda/config.py — thin wrapper, just re-exports from root config
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, THRESHOLDS, INTENT_CFG

# ─── SwDA training paths (only relevant to the SwDA pipeline) ────────────────
SWDA_DATA_DIR = r"C:\Finals_Project\swda\swda_data"

PATHS = {
    **PATHS,   # keep all the root paths
    "swda_root"    : r"C:\Finals_Project\swda",
    "processed_csv": os.path.join(SWDA_DATA_DIR, "processed_intents.csv"),
    "prep_report"  : os.path.join(SWDA_DATA_DIR, "prep_report.txt"),
    "augmented_csv": os.path.join(SWDA_DATA_DIR, "processed_intents_AUGMENTED.csv"),
}

# Training-only settings that don't belong in the runtime config
INTENT_MAPPING = {
    "qy": "question", "qw": "question", "qo": "question",
    "qh": "question", "qr": "question", "qrr": "question",
    "ad": "request",
    "sd": "statement", "sv": "statement",
    "fp": "general",  "fc": "general",
    "b":  "general",  "bk": "general",
}

INTENT_LABELS = sorted(set(INTENT_MAPPING.values()))

CFG = {
    "max_vocab_size": 10_000,
    "max_seq_len":    50,
    "min_text_len":   5,
    "test_size":      0.15,
    "val_size":       0.15,
    "random_state":   42,
    "embedding_dim":  64,
    "lstm_units":     128,
    "dense_units":    64,
    "dropout_rate":   0.4,
    "use_bilstm":     True,
    "epochs":         50,
    "batch_size":     32,
    "learning_rate":  1e-3,
    "patience_es":    7,
    "patience_lr":    3,
    "lr_factor":      0.5,
    "min_lr":         1e-6,
}