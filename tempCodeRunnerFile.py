"""
main.py — FastAPI server for the Dual-Head Chatbot.

Endpoints:
  POST   /chat              — main chat turn
  DELETE /session/{id}      — reset a session's history state
  GET    /session/{id}      — inspect a session's current state
  GET    /health            — liveness probe
  GET    /logic/actions     — list all valid action codes (useful for debugging)

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
  http://localhost:8000/docs
"""

import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import ModelBundle, predict

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE  (models loaded once, sessions stored in-memory)
# ─────────────────────────────────────────────────────────────────────────────

# Global model bundle — populated in lifespan()
model_bundle: Optional[ModelBundle] = None

# In-memory session store  { session_id: history_state }
sessions: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artefacts before the first request."""
    global model_bundle
    logger.info("=== Chatbot Backend starting up — loading models ===")
    model_bundle = ModelBundle()
    logger.info("=== Models ready. Server accepting requests. ===")
    yield
    logger.info("=== Chatbot Backend shutting down. ===")


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dual-Head Chatbot API",
    description=(
        "Intent classification + Emotion detection + Stateful logic module. "
        "Each conversation is tracked by a session_id."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any origin (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         json_schema_extra={"example": "Hello World"})  
    session_id: Optional[str] = Field(
        default=None,
        description="Omit to start a new session; include to continue an existing one."
    )

class ChatResponse(BaseModel):
    session_id:   str
    action:       str           # logic module action code
    intent:       str
    intent_conf:  float
    emotion:      str
    emotion_conf: float
    prev_state:   str           # history state before this turn
    next_state:   str           # history state after this turn

class SessionInfo(BaseModel):
    session_id:    str
    history_state: str

class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    active_sessions: int


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat(req: ChatRequest):
    """
    Main chat endpoint.

    - If `session_id` is omitted, a new session is created and its ID is returned.
    - The session's `history_state` is updated automatically after each turn.
    - Pass the returned `session_id` in subsequent requests to continue the conversation.
    """
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    # Resolve / create session
    session_id = req.session_id or str(uuid.uuid4())
    history_state = sessions.get(session_id, "start")

    # Run the full pipeline
    result = predict(model_bundle, req.message, history_state)

    # Persist new state
    sessions[session_id] = result["next_state"]

    logger.info(
        f"[{session_id[:8]}] "
        f"intent={result['intent']}({result['intent_conf']:.2f}) | "
        f"emotion={result['emotion']}({result['emotion_conf']:.2f}) | "
        f"action={result['action']} | "
        f"state: {result['prev_state']} → {result['next_state']}"
    )

    return ChatResponse(session_id=session_id, **result)


@app.get("/session/{session_id}", response_model=SessionInfo,
         summary="Inspect a session's current state")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionInfo(session_id=session_id, history_state=sessions[session_id])


@app.delete("/session/{session_id}", summary="Reset a session (start fresh)")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions[session_id] = "start"
    return {"detail": f"Session '{session_id}' reset to 'start'."}


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    return HealthResponse(
        status="ok" if model_bundle is not None else "loading",
        models_loaded=model_bundle is not None,
        active_sessions=len(sessions),
    )


@app.get("/logic/actions", summary="List all valid action codes")
async def list_actions():
    """Returns every action code defined in the Golden Set logic CSV."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    actions = sorted(model_bundle.logic_map.values())
    return {"actions": list(dict.fromkeys(actions))}  # deduplicated, order preserved