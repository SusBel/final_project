"""
inference.py — Loads all three model artefacts once at startup
               and exposes a single predict() function for the API.

Pipeline:
  raw text  →  clean()
            →  Intent Model   (BiLSTM)     → intent label + confidence
            →  Emotion Model  (TF-IDF+MLP) → emotion label + confidence
            →  Logic Module   (CSV lookup) → action code
"""

import re
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.preprocessing.sequence import pad_sequences

from config import PATHS, INTENT_CFG, THRESHOLDS, STATE_TRANSITIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING  (mirrors the preprocessing used during training)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove special chars — same logic used in both training scripts."""
    text = re.sub(r"[{}\[\]<>#]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER  (called once at app startup — not on every request)
# ─────────────────────────────────────────────────────────────────────────────

def _load_keras_model(path: str) -> tf.keras.Model:
    """Load a .keras model. Run fix_models.py first if you see config errors."""
    return tf.keras.models.load_model(path, compile=False)


class ModelBundle:
    """Holds all loaded artefacts to avoid reloading per request."""

    def __init__(self):
        logger.info("Loading Intent model …")
        self.intent_model = _load_keras_model(PATHS["intent_model"])

        with open(PATHS["intent_tokenizer"], "rb") as f:
            self.intent_tokenizer = pickle.load(f)

        with open(PATHS["intent_label_encoder"], "rb") as f:
            self.intent_le = pickle.load(f)

        logger.info("Loading Emotion model …")
        self.emotion_model = _load_keras_model(PATHS["emotion_model"])

        with open(PATHS["emotion_label_encoder"], "rb") as f:
            self.emotion_le = pickle.load(f)

        logger.info("Loading Logic Module (Golden Set CSV) …")
        self.logic_df = pd.read_csv(PATHS["logic_csv"])

        # Build a lookup dict for O(1) access:
        # key = (intent, emotion, history_state) → action
        self.logic_map: dict[tuple, str] = {
            (row["input_intent"], row["input_emotion"], row["history_state"]): row["expected_response"]
            for _, row in self.logic_df.iterrows()
        }

        logger.info("All models loaded successfully ✓")


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_intent(bundle: ModelBundle, text: str) -> tuple[str, float]:
    """Returns (intent_label, confidence)."""
    cleaned = clean_text(text)
    seq = bundle.intent_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=INTENT_CFG["max_seq_len"],
                           padding="post", truncating="post")
    proba = bundle.intent_model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])

    if confidence < THRESHOLDS["intent_min_confidence"]:
        return "general", confidence          # low-confidence fallback

    label = bundle.intent_le.inverse_transform([idx])[0]
    return label, confidence


def predict_emotion(bundle: ModelBundle, text: str) -> tuple[str, float]:
    """Returns (emotion_label, confidence).
    The emotion model uses a TextVectorization layer inside the Keras graph,
    so we feed the raw string directly (no external tokenisation needed).
    """
    # The emotion model expects shape (batch, 1) of raw strings
    input_array = np.array([[text]])          # shape (1, 1)
    proba = bundle.emotion_model.predict(input_array, verbose=0)[0]
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])

    if confidence < THRESHOLDS["emotion_min_confidence"]:
        return "neutral", confidence          # low-confidence fallback

    label = bundle.emotion_le.inverse_transform([idx])[0]
    return label, confidence


# ─────────────────────────────────────────────────────────────────────────────
# LOGIC MODULE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def lookup_action(bundle: ModelBundle,
                  intent: str,
                  emotion: str,
                  history_state: str) -> str:
    """
    Looks up the action in the Golden Set.
    Fallback cascade:
      1. Exact match  (intent, emotion, history_state)
      2. Relax history → 'start'
      3. Relax emotion → 'neutral'
      4. Default fallback action
    """
    key = (intent, emotion, history_state)
    if key in bundle.logic_map:
        return bundle.logic_map[key]

    # Fallback 1: try with history_state = 'start'
    key2 = (intent, emotion, "start")
    if key2 in bundle.logic_map:
        logger.debug("Logic fallback: relaxed history_state → 'start'")
        return bundle.logic_map[key2]

    # Fallback 2: try with emotion = 'neutral'
    key3 = (intent, "neutral", "start")
    if key3 in bundle.logic_map:
        logger.debug("Logic fallback: relaxed emotion → 'neutral'")
        return bundle.logic_map[key3]

    logger.warning(f"No logic match for {key} — using 'fallback'")
    return "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY STATE TRANSITION
# ─────────────────────────────────────────────────────────────────────────────

def advance_state(current_state: str, action: str) -> str:
    """Returns the next history_state given the current state and bot action."""
    return STATE_TRANSITIONS.get((current_state, action), current_state)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICT FUNCTION  (called by the API endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def predict(bundle: ModelBundle,
            user_message: str,
            history_state: str = "start") -> dict:
    """
    Full pipeline:  text + state  →  structured response dict.

    Returns:
    {
        "response":        str,    # bot reply text
        "action":          str,    # action code from Logic Module
        "intent":          str,    # predicted intent label
        "intent_conf":     float,
        "emotion":         str,    # predicted emotion label
        "emotion_conf":    float,
        "prev_state":      str,    # history state coming in
        "next_state":      str,    # history state after this turn
    }
    """
    # 1. Classify intent
    intent, intent_conf = predict_intent(bundle, user_message)

    # 2. Classify emotion
    emotion, emotion_conf = predict_emotion(bundle, user_message)

    # 3. Logic lookup
    action = lookup_action(bundle, intent, emotion, history_state)

    # 4. Advance conversation state
    next_state = advance_state(history_state, action)

    return {
        "action":       action,
        "intent":       intent,
        "intent_conf":  round(intent_conf, 4),
        "emotion":      emotion,
        "emotion_conf": round(emotion_conf, 4),
        "prev_state":   history_state,
        "next_state":   next_state,
    }