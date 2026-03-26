"""
config.py — Paths and settings for the Dual-Head Chatbot backend.
Adjust paths here to match where your model files live on disk.
"""

import os

# ─── Base directory (default: same folder as this file) ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Model artefact paths ────────────────────────────────────────────────────
PATHS = {
    # Machine 1 — Intent Classification
    "intent_model":         os.path.join(BASE_DIR, "intent_model.keras"),
    "intent_tokenizer":     os.path.join(BASE_DIR, "tokenizer.pkl"),
    "intent_label_encoder": os.path.join(BASE_DIR, "label_encoder.pkl"),

    # Machine 2 — Emotion Classification
    "emotion_model":         os.path.join(BASE_DIR, "emotion_model"),
    "emotion_label_encoder": os.path.join(BASE_DIR, "emotion_label_encoder.pkl"),

    # Machine 3 — Logic Module (Golden Set CSV)
    "logic_csv": os.path.join(BASE_DIR, "golden_set_logic_stateful.csv"),
}

# ─── Intent model preprocessing settings (must match training config) ────────
INTENT_CFG = {
    "max_seq_len": 50,      # same value as CFG["max_seq_len"] in training
}

# ─── Confidence thresholds ────────────────────────────────────────────────────
THRESHOLDS = {
    "intent_min_confidence":  0.40,   # below → fallback to "general"
    "emotion_min_confidence": 0.35,   # below → fallback to "neutral"
}

# ─── History-state transitions ────────────────────────────────────────────────
# Maps (current_history_state, bot_action) → next_history_state
# This drives stateful multi-turn behaviour.
STATE_TRANSITIONS = {
    ("start",               "apology_empathy"):                  "bot_apologized",
    ("start",               "apology_quality_assurance"):        "bot_apologized",
    ("start",               "apology_rephrase"):                 "bot_rephrased",
    ("start",               "security_reassurance"):             "bot_reassured",
    ("start",               "urgent_assistance"):                "bot_assisted",
    ("start",               "provide_information"):              "bot_provided_info",
    ("start",               "execute_action"):                   "service_completed",
    ("start",               "process_order_enthusiastic"):       "service_completed",
    ("start",               "check_account_status"):             "bot_provided_info",
    ("start",               "empathy_retention_offer"):          "bot_provided_info",
    ("start",               "empathetic_explanation"):           "bot_provided_info",
    ("start",               "technical_solution_with_apology"):  "bot_provided_info",
    ("bot_provided_info",   "apology_rephrase"):                 "bot_rephrased",
    ("bot_provided_info",   "close_interaction"):                "start",
    ("bot_reassured",       "provide_detailed_policy"):          "bot_provided_info",
    ("bot_apologized",      "escalate_to_human"):                "start",
    ("bot_apologized",      "offer_compensation"):               "service_completed",
    ("bot_solved_problem",  "polite_closing"):                   "start",
    ("bot_rephrased",       "happy_closing"):                    "start",
    ("service_completed",   "thank_and_close"):                  "start",
    ("service_completed",   "polite_closing"):                   "start",
}
