"""
=============================================================================
  config.py — Shared Configuration
  Dual-Head Chatbot | Machine 1: Intent Classification
  
  Single source of truth for intent mapping, paths, and model hyperparams.
  Both data_preparation.py and model_training.py import from here.
=============================================================================
"""

# ── SwDA act_tag  →  simplified intent label ─────────────────────────────────
# Source: Switchboard Dialog Act Corpus tag set
# https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
INTENT_MAPPING = {
    # Questions
    "qy"  : "question",   # Yes/No question
    "qw"  : "question",   # Wh-question
    "qo"  : "question",   # Open-ended question
    "qh"  : "question",   # Rhetorical question
    "qr"  : "question",   # Or-clause following y/n question
    "qrr" : "question",   # Or-clause following rhetorical question

    # Requests / directives
    "ad"  : "request",    # Action directive

    # Statements
    "sd"  : "statement",  # Statement – non-opinion
    "sv"  : "statement",  # Statement – opinion/point of view

    # General / social acts
    "fp"  : "general",    # Conventional opening
    "fc"  : "general",    # Conventional closing
    "b"   : "general",    # Acknowledge / back-channel
    "bk"  : "general",    # Response acknowledgement
}

# Derived list of final class labels (used by model_training.py)
INTENT_LABELS = sorted(set(INTENT_MAPPING.values()))   # ['general','question','request','statement']

# ── File / folder paths ───────────────────────────────────────────────────────
PATHS = {
    "swda_root"       : ".",                       # root dir to scan for SwDA CSVs
    "processed_csv"   : "processed_intents.csv",   # output of data_preparation.py
    "prep_report"     : "prep_report.txt",         # data stats saved after prep
    "model"           : "intent_model.keras",      # best model checkpoint
    "tokenizer"       : "tokenizer.pkl",
    "label_encoder"   : "label_encoder.pkl",
    "confusion_matrix": "confusion_matrix.png",
    "training_curves" : "training_history.png",
}

# ── Model / training hyperparameters ─────────────────────────────────────────
CFG = {
    # Preprocessing
    "max_vocab_size" : 10_000,
    "max_seq_len"    : 50,
    "min_text_len"   : 5,       # minimum char length to keep a sample

    # Train / val / test split
    "test_size"      : 0.15,
    "val_size"       : 0.15,
    "random_state"   : 42,

    # Architecture
    "embedding_dim"  : 64,
    "lstm_units"     : 128,
    "dense_units"    : 64,
    "dropout_rate"   : 0.4,
    "use_bilstm"     : True,    # False → GlobalAveragePooling1D (faster, less accurate)

    # Training
    "epochs"         : 50,
    "batch_size"     : 32,
    "learning_rate"  : 1e-3,

    # Callbacks
    "patience_es"    : 7,       # EarlyStopping patience (epochs)
    "patience_lr"    : 3,       # ReduceLROnPlateau patience
    "lr_factor"      : 0.5,
    "min_lr"         : 1e-6,
}