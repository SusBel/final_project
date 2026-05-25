"""
config.py — Paths and settings for the Dual-Head Chatbot backend.
"""

import os

INTENT_MODEL_PATH         = r"C:\Finals_Project\ai_backend\swda\intent_model.keras"  
INTENT_TOKENIZER_PATH     = r"C:\Finals_Project\ai_backend\swda\intent_tokenizer.pkl"   
INTENT_LABEL_ENCODER_PATH = r"C:\Finals_Project\ai_backend\swda\intent_label_encoder.pkl"   

EMOTION_MODEL_PATH        = r"C:\Finals_Project\ai_backend\GoEmotions\emotion_model"
EMOTION_LABEL_ENCODER_PATH= r"C:\Finals_Project\ai_backend\GoEmotions\emotion_label_encoder.pkl"  

LOGIC_CSV_PATH            = r"C:\Finals_Project\ai_backend\GoldenSet\golden_set_logic_stateful.csv"   

def _p(override, filename):
    """Use the override path if set, otherwise look next to this file."""
    if override is not None:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


PATHS = {
    # Machine 1 — Intent Classification
    "intent_model":         _p(INTENT_MODEL_PATH,          "intent_model.keras"),
    "intent_tokenizer":     _p(INTENT_TOKENIZER_PATH,      "tokenizer.pkl"),
    "intent_label_encoder": _p(INTENT_LABEL_ENCODER_PATH,  "label_encoder.pkl"),

    # Machine 2 — Emotion Classification
    "emotion_model":         _p(EMOTION_MODEL_PATH,         "emotion_model_v5.keras"),
    "emotion_label_encoder": _p(EMOTION_LABEL_ENCODER_PATH, "emotion_label_encoder.pkl"),

    # Machine 3 — Logic Module (Golden Set CSV)
    "logic_csv": _p(LOGIC_CSV_PATH, "golden_set_logic_stateful.csv"),
}

# ─── Intent model preprocessing settings (must match training config) ────────
INTENT_CFG = {"max_seq_len": 50}

# ─── Confidence thresholds ────────────────────────────────────────────────────
THRESHOLDS = {
    "intent_min_confidence":  0.40,   # below → fallback to "general"
    "emotion_min_confidence": 0.35,   # below → fallback to "neutral"
}

# ─── History-state transitions ────────────────────────────────────────────────
# Maps (current_history_state, bot_action) → next_history_state
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