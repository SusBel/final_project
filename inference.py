"""
inference.py — Clean version for TF 2.10
Pipeline:
    raw text
      ├──► Intent Model  (BiLSTM)       → intent + confidence
      └──► Emotion Model (TF-IDF MLP)   → emotion + confidence
                  │
            Logic Module (CSV)
            (intent, emotion, state) → action
"""
import os, re, pickle, logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import PATHS, INTENT_CFG, THRESHOLDS, STATE_TRANSITIONS

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    text = re.sub(r"[{}\[\]<>#]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_emotion_model(path: str):
    """
    Emotion model is saved as a SavedModel folder (not .h5/.keras).
    This strips any accidental extension and ensures we load the folder.
    """
    # Strip extension if someone accidentally added one
    clean_path = path.rstrip("/\\")
    for ext in (".keras", ".h5"):
        if clean_path.endswith(ext):
            clean_path = clean_path[: -len(ext)]
            break

    # Prefer the folder; fall back to exact path
    if os.path.isdir(clean_path):
        logger.info(f"Loading emotion model from folder: {clean_path}")
        return tf.keras.models.load_model(clean_path, compile=False)
    elif os.path.exists(path):
        logger.info(f"Loading emotion model from file: {path}")
        return tf.keras.models.load_model(path, compile=False)
    else:
        raise FileNotFoundError(
            f"Emotion model not found at '{path}' or '{clean_path}'.\n"
            f"Expected a SavedModel folder named 'emotion_model' in your project folder.\n"
            f"Run: python retrain_emotion_model.py"
        )


class ModelBundle:
    def __init__(self):
        logger.info("Loading Intent model ...")
        self.intent_model = tf.keras.models.load_model(
            PATHS["intent_model"], compile=False
        )
        with open(PATHS["intent_tokenizer"],     "rb") as f:
            self.intent_tokenizer = pickle.load(f)
        with open(PATHS["intent_label_encoder"], "rb") as f:
            self.intent_le = pickle.load(f)

        logger.info("Loading Emotion model ...")
        self.emotion_model = _load_emotion_model(PATHS["emotion_model"])
        with open(PATHS["emotion_label_encoder"], "rb") as f:
            self.emotion_le = pickle.load(f)

        logger.info("Loading Logic Module ...")
        df = pd.read_csv(PATHS["logic_csv"])
        self.logic_map = {
            (r["input_intent"], r["input_emotion"], r["history_state"]): r["expected_response"]
            for _, r in df.iterrows()
        }
        logger.info("All models loaded ✓")


def _predict_intent(bundle: ModelBundle, text: str):
    seq    = bundle.intent_tokenizer.texts_to_sequences([_clean(text)])
    padded = pad_sequences(seq, maxlen=INTENT_CFG["max_seq_len"],
                           padding="post", truncating="post")
    proba  = bundle.intent_model.predict(padded, verbose=0)[0]
    idx    = int(np.argmax(proba))
    conf   = float(proba[idx])
    if conf < THRESHOLDS["intent_min_confidence"]:
        return "general", conf
    return bundle.intent_le.inverse_transform([idx])[0], conf


def _predict_emotion(bundle: ModelBundle, text: str):
    proba = bundle.emotion_model.predict(np.array([[text]]), verbose=0)[0]
    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])
    if conf < THRESHOLDS["emotion_min_confidence"]:
        return "neutral", conf
    return bundle.emotion_le.inverse_transform([idx])[0], conf


def _lookup_action(bundle: ModelBundle, intent: str, emotion: str, state: str) -> str:
    if (k := (intent, emotion, state))     in bundle.logic_map: return bundle.logic_map[k]
    if (k := (intent, emotion, "start"))   in bundle.logic_map: return bundle.logic_map[k]
    if (k := (intent, "neutral", "start")) in bundle.logic_map: return bundle.logic_map[k]
    logger.warning(f"No logic match for ({intent}, {emotion}, {state})")
    return "fallback"


def predict(bundle: ModelBundle, user_message: str, history_state: str = "start") -> dict:
    intent,  intent_conf  = _predict_intent(bundle,  user_message)
    emotion, emotion_conf = _predict_emotion(bundle, user_message)
    action     = _lookup_action(bundle, intent, emotion, history_state)
    next_state = STATE_TRANSITIONS.get((history_state, action), history_state)

    logger.info(
        f"intent={intent}({intent_conf:.2f}) | "
        f"emotion={emotion}({emotion_conf:.2f}) | "
        f"action={action} | {history_state}→{next_state}"
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