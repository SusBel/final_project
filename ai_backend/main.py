"""
main.py — FastAPI server for the Dual-Head Chatbot.

Endpoints:
  POST   /chat              — main chat turn
  DELETE /session/{id}      — reset a session's history state
  GET    /session/{id}      — inspect a session's current state
  GET    /health            — liveness probe
  GET    /logic/actions     — list all valid action codes (useful for debugging)

Run:
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
  http://localhost:8000/docs
"""

import uuid
import logging
import mysql.connector
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import ModelBundle, predict

from fastapi.staticfiles import StaticFiles  # <-- ייבוא מנגנון הקבצים הסטטיים
from fastapi.responses import FileResponse    # <-- ייבוא רכיב החזרת קבצים

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE & DATABASE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Global model bundle — populated in lifespan()
model_bundle: Optional[ModelBundle] = None

# Database connection configuration 
DB_CONFIG = {
    "host": "localhost",
    "port": 3308,             
    "user": "root",
    "password": "yaronsql", 
    "database": "personaai_auth"         
}

def get_db_connection():
    """
    Establishes and returns a new connection to the MySQL database.
    Ensures that each API request operates on a fresh, thread-safe connection.
    """
    return mysql.connector.connect(**DB_CONFIG)

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

# 1. הגדרת התיקייה static כתיקיית קבצים סטטיים ציבורית
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. יצירת נקודת קצה (Endpoint) שמחזירה את קובץ ה-HTML כאשר ניגשים ל-index.html
@app.get("/index.html")
async def get_index_page():
    return FileResponse("static/index.html")

# 3. אופציונלי: ניתוב גם של דף הבית הראשי לקובץ הצ'אט
@app.get("/")
async def get_home_page():
    return FileResponse("static/index.html")
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
                         example="I'm so angry, nothing is working!")
    session_id: Optional[str] = Field(
        default=None,
        description="Omit to start a new session; include to continue an existing one."
    )
    user_id: Optional[int] = Field(
        default=None, 
        description="Optional Java User ID to link this session to a registered user."
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
    Main chat endpoint with Database Persistence.
    - Resolves or creates a session in the MySQL 'sessions' table.
    - Processes text via the AI Inference Pipeline.
    - Logs the full interaction into the MySQL 'logs' table.
    """
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    # Initialize database connection
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    session_id = req.session_id or str(uuid.uuid4())
    history_state = "start"

    try:
        # Step 1: Retrieve the current session state from the database
        cursor.execute("SELECT current_state FROM sessions WHERE session_id = %s", (session_id,))
        row = cursor.fetchone()
        
        if row:
            history_state = row['current_state']
        else:
            # Create a new session entry if it does not exist
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id, current_state, status) VALUES (%s, %s, %s, %s)",
                (session_id, req.user_id, history_state, "active")
            )
            conn.commit()

        # Step 2: Run the complete dual-head AI and logic pipeline
        result = predict(model_bundle, req.message, history_state)

        # Step 3: Persist the updated history state back to the sessions table
        cursor.execute(
            "UPDATE sessions SET current_state = %s, last_updated_at = NOW() WHERE session_id = %s",
            (result["next_state"], session_id)
        )

        # Step 4: Archive the interaction details into the logs table
        cursor.execute("""
            INSERT INTO logs (session_id, message, intent, emotion, action, prev_state, next_state, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            session_id, req.message, result["intent"], result["emotion"], 
            result["action"], result["prev_state"], result["next_state"]
        ))
        
        # Commit all transaction changes securely
        conn.commit() 
        
    except Exception as e:
        # Rollback prevents partial data corruption in case of unexpected failures
        conn.rollback() 
        logger.error(f"Database transaction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Database Error")
    finally:
        # Ensure resources are released immediately
        cursor.close()
        conn.close() 

    logger.info(
        f"[{session_id[:8]}] "
        f"intent={result['intent']}({result['intent_conf']:.2f}) | "
        f"emotion={result['emotion']}({result['emotion_conf']:.2f}) | "
        f"action={result['action']}"
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
    """
    Resets the session state back to 'start' and marks it as 'closed' in the database.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Perform soft-reset by updating status rather than deleting the historical record
        cursor.execute(
            "UPDATE sessions SET current_state = 'start', status = 'closed', last_updated_at = NOW() WHERE session_id = %s", 
            (session_id,)
        )
        conn.commit()
        
    except Exception as e:
        logger.error(f"Failed to reset session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Database Error during session reset")
    finally:
        cursor.close()
        conn.close()

    return {"detail": f"Session '{session_id}' has been reset to 'start' and closed."}

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