"""
config.py — Backend settings.
All paths and tuning knobs in one place.
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "intent_model":          os.path.join(BASE_DIR, "intent_model.keras"),
    "intent_tokenizer":      os.path.join(BASE_DIR, "tokenizer.pkl"),
    "intent_label_encoder":  os.path.join(BASE_DIR, "label_encoder.pkl"),
    "emotion_model":         os.path.join(BASE_DIR, "emotion_model"),
    "emotion_label_encoder": os.path.join(BASE_DIR, "emotion_label_encoder.pkl"),
    "logic_csv":             os.path.join(BASE_DIR, "golden_set_logic_stateful.csv"),
}

# Must match max_seq_len used in intent model training
INTENT_CFG = {"max_seq_len": 50}

THRESHOLDS = {
    "intent_min_confidence":  0.40,
    "emotion_min_confidence": 0.35,
}

STATE_TRANSITIONS = {
    ("start",              "apology_empathy"):                 "bot_apologized",
    ("start",              "apology_quality_assurance"):       "bot_apologized",
    ("start",              "apology_rephrase"):                "bot_rephrased",
    ("start",              "security_reassurance"):            "bot_reassured",
    ("start",              "urgent_assistance"):               "bot_assisted",
    ("start",              "provide_information"):             "bot_provided_info",
    ("start",              "execute_action"):                  "service_completed",
    ("start",              "process_order_enthusiastic"):      "service_completed",
    ("start",              "check_account_status"):            "bot_provided_info",
    ("start",              "empathy_retention_offer"):         "bot_provided_info",
    ("start",              "empathetic_explanation"):          "bot_provided_info",
    ("start",              "technical_solution_with_apology"): "bot_provided_info",
    ("bot_provided_info",  "apology_rephrase"):                "bot_rephrased",
    ("bot_provided_info",  "close_interaction"):               "start",
    ("bot_reassured",      "provide_detailed_policy"):         "bot_provided_info",
    ("bot_apologized",     "escalate_to_human"):               "start",
    ("bot_apologized",     "offer_compensation"):              "service_completed",
    ("bot_solved_problem", "polite_closing"):                  "start",
    ("bot_rephrased",      "happy_closing"):                   "start",
    ("service_completed",  "thank_and_close"):                 "start",
    ("service_completed",  "polite_closing"):                  "start",
}