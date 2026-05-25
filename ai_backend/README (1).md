# Dual-Head Chatbot — FastAPI Backend

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────┐
│  Machine 1: Intent Model    │  BiLSTM  →  intent label + confidence
│  (intent_model.keras)       │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Machine 2: Emotion Model   │  TF-IDF + MLP  →  emotion label + confidence
│  (emotion_model_optimized)  │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Machine 3: Logic Module    │  (intent, emotion, history_state) → action code
│  (golden_set_logic.csv)     │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Response Templates         │  action code → bot reply text
└─────────────────────────────┘
```

## File Structure

```
chatbot_backend/
├── main.py                         ← FastAPI app + endpoints
├── inference.py                    ← Full prediction pipeline
├── config.py                       ← Paths + settings
├── response_templates.py           ← Action code → bot reply text
├── requirements.txt
│
│   ── Place your model files here ──
├── intent_model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── emotion_model_optimized.keras
├── emotion_label_encoder.pkl
└── golden_set_logic_stateful.csv
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your model files into this folder (see structure above)

# 3. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/chat` | Send a message, get a bot reply |
| `GET` | `/session/{id}` | Inspect a session's current state |
| `DELETE` | `/session/{id}` | Reset a session to 'start' |
| `GET` | `/health` | Liveness probe |
| `GET` | `/logic/actions` | List all action codes |
| `GET` | `/docs` | Interactive Swagger UI |

## Example Usage

### Start a new conversation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Nothing is working and I am furious!"}'
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "response": "I'm truly sorry to hear you're having this experience...",
  "action": "apology_empathy",
  "intent": "statement",
  "intent_conf": 0.87,
  "emotion": "anger",
  "emotion_conf": 0.92,
  "prev_state": "start",
  "next_state": "bot_apologized"
}
```

### Continue the same conversation (still angry)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to speak to a manager!", "session_id": "a1b2c3d4-..."}'
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "response": "I understand this situation requires more personalised attention...",
  "action": "escalate_to_human",
  ...
  "prev_state": "bot_apologized",
  "next_state": "start"
}
```

## Customisation

- **Change model file locations**: edit `PATHS` in `config.py`
- **Tune confidence thresholds**: edit `THRESHOLDS` in `config.py`
- **Change what the bot says**: edit `RESPONSE_TEMPLATES` in `response_templates.py`
- **Add new state transitions**: edit `STATE_TRANSITIONS` in `config.py`
