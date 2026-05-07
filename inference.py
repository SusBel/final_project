"""
inference.py — Prediction pipeline for the Dual-Head Chatbot.

Pipeline:
    raw text
      ├──► Intent Model  (BiLSTM)       → intent + confidence
      └──► Emotion Model (TF-IDF MLP)   → emotion + confidence
                  │
            Logic Module (CSV)
            (intent, emotion, state) → action
"""

import os
import re
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import PATHS, INTENT_CFG, THRESHOLDS, STATE_TRANSITIONS

logger = logging.getLogger(__name__)

INTENT_MODEL_PATH         = r"C:\Finals_Project\swda\intent_model.keras"  
INTENT_TOKENIZER_PATH     = r"C:\Finals_Project\swda\intent_tokenizer.pkl"   
INTENT_LABEL_ENCODER_PATH = r"C:\Finals_Project\swda\intent_label_encoder.pkl"   

EMOTION_MODEL_PATH        = r"C:\Finals_Project\GoEmotions\emotion_model"
EMOTION_LABEL_ENCODER_PATH= r"C:\Finals_Project\GoEmotions\emotion_label_encoder.pkl"  

LOGIC_CSV_PATH            = r"C:\Finals_Project\GoldenSet\golden_set_logic.csv"   

def _resolve(override, config_key: str) -> str:
    """Returns the override path if set, otherwise falls back to config.py."""
    return override if override is not None else PATHS[config_key]


def _clean(text: str) -> str:
    """Normalise raw user text before feeding it to the intent model."""
    text = re.sub(r"[{}\[\]<>#]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_emotion_model(path: str):
    """
    Load the emotion model.
    Accepts either a SavedModel folder or a .keras / .h5 file.
    """
    clean_path = path.rstrip("/\\")

    # Strip accidental extension so we can check for the folder first
    for ext in (".keras", ".h5"):
        if clean_path.endswith(ext):
            clean_path = clean_path[: -len(ext)]
            break

    if os.path.isdir(clean_path):
        logger.info(f"Loading emotion model from folder: {clean_path}")
        return tf.keras.models.load_model(clean_path, compile=False)

    if os.path.exists(path):
        logger.info(f"Loading emotion model from file: {path}")
        return tf.keras.models.load_model(path, compile=False)

    raise FileNotFoundError(
        f"Emotion model not found.\n"
        f"  Tried folder : {clean_path}\n"
        f"  Tried file   : {path}\n"
        f"Fix: set EMOTION_MODEL_PATH at the top of inference.py, "
        f"or run retrain_emotion_model.py to rebuild the model."
    )


# =============================================================================
#  MODEL BUNDLE  —  loads every artefact once at startup
# =============================================================================

class ModelBundle:
    """Holds all loaded models and encoders. Created once in main.py lifespan."""

    def __init__(self):
        # ── Intent model ──────────────────────────────────────────────────────
        intent_model_path = _resolve(INTENT_MODEL_PATH, "intent_model")
        logger.info(f"Loading intent model from: {intent_model_path}")
        self.intent_model = tf.keras.models.load_model(intent_model_path, compile=False)

        tokenizer_path = _resolve(INTENT_TOKENIZER_PATH, "intent_tokenizer")
        with open(tokenizer_path, "rb") as f:
            self.intent_tokenizer = pickle.load(f)

        intent_le_path = _resolve(INTENT_LABEL_ENCODER_PATH, "intent_label_encoder")
        with open(intent_le_path, "rb") as f:
            self.intent_le = pickle.load(f)

        # ── Emotion model ─────────────────────────────────────────────────────
        emotion_model_path = _resolve(EMOTION_MODEL_PATH, "emotion_model")
        logger.info(f"Loading emotion model from: {emotion_model_path}")
        self.emotion_model = _load_emotion_model(emotion_model_path)

        emotion_le_path = _resolve(EMOTION_LABEL_ENCODER_PATH, "emotion_label_encoder")
        with open(emotion_le_path, "rb") as f:
            self.emotion_le = pickle.load(f)

        # ── Logic module (Golden Set CSV) ─────────────────────────────────────
        logic_csv_path = _resolve(LOGIC_CSV_PATH, "logic_csv")
        logger.info(f"Loading logic CSV from: {logic_csv_path}")
        df = pd.read_csv(logic_csv_path)
        self.logic_map = {
            (r["input_intent"], r["input_emotion"], r["history_state"]): r["expected_response"]
            for _, r in df.iterrows()
        }
        # ── KNN Fallback Loading ──────────────────────────────────────────────
        knn_model_path = r"C:\Finals_Project\GoldenSet\logic_knn_model.pkl"
        logger.info(f"Loading Logic KNN model from: {knn_model_path}")
        with open(knn_model_path, 'rb') as f:
            self.knn_model = pickle.load(f)
            
        knn_encoder_path = r"C:\Finals_Project\GoldenSet\logic_encoder.pkl"
        logger.info(f"Loading Logic KNN encoder from: {knn_encoder_path}")
        with open(knn_encoder_path, 'rb') as f:
            self.knn_encoder = pickle.load(f)

        logger.info("All models loaded successfully")


# =============================================================================
#  PREDICTION HELPERS
# =============================================================================

def _predict_intent(bundle: ModelBundle, text: str) -> tuple[str, float]:
    seq    = bundle.intent_tokenizer.texts_to_sequences([_clean(text)])
    padded = pad_sequences(
        seq,
        maxlen    = INTENT_CFG["max_seq_len"],
        padding   = "post",
        truncating= "post",
    )
    proba = bundle.intent_model.predict(padded, verbose=0)[0]
    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])

    if conf < THRESHOLDS["intent_min_confidence"]:
        return "general", conf

    return bundle.intent_le.inverse_transform([idx])[0], conf


def _predict_emotion(bundle: ModelBundle, text: str) -> tuple[str, float]:
    proba = bundle.emotion_model.predict(np.array([[text]]), verbose=0)[0]
    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])

    if conf < THRESHOLDS["emotion_min_confidence"]:
        return "neutral", conf

    return bundle.emotion_le.inverse_transform([idx])[0], conf


def _lookup_action(bundle: ModelBundle, intent: str, emotion: str, state: str) -> str:
    """
    Look up the best matching action.
    Uses Exact Match (Dictionary) first. If not found, falls back to KNN.
    """
    # 1. חיפוש התאמה מדויקת במילון
    if (intent, emotion, state) in bundle.logic_map:
        return bundle.logic_map[(intent, emotion, state)]
    
    # שלב 2': מנגנון גיבוי מבוסס KNN!    
    logger.info(f"No exact match for ({intent}, {emotion}, {state}). Using KNN Fallback...")
    
    X_new = pd.DataFrame([[intent, emotion, state]], 
                         columns=['input_intent', 'input_emotion', 'history_state'])
    X_new_encoded = bundle.knn_encoder.transform(X_new)
    predicted_action = bundle.knn_model.predict(X_new_encoded)[0]
    
    return predicted_action


# =============================================================================
#  PUBLIC API
# =============================================================================

def predict(bundle: ModelBundle, user_message: str, history_state: str = "start") -> dict:
    """
    Run the full pipeline on a single user message.

    Returns a dict with keys:
        action, intent, intent_conf, emotion, emotion_conf, prev_state, next_state
    """
    intent,  intent_conf  = _predict_intent(bundle,  user_message)
    emotion, emotion_conf = _predict_emotion(bundle, user_message)
    action     = _lookup_action(bundle, intent, emotion, history_state)
    next_state = STATE_TRANSITIONS.get((history_state, action), history_state)

    logger.info(
        f"intent={intent}({intent_conf:.2f}) | "
        f"emotion={emotion}({emotion_conf:.2f}) | "
        f"action={action} | {history_state} → {next_state}"
    )

    return {
        "action":       action,
        "intent":       intent,
        "intent_conf":  round(intent_conf,  4),
        "emotion":      emotion,
        "emotion_conf": round(emotion_conf, 4),
        "prev_state":   history_state,
        "next_state":   next_state,
    }