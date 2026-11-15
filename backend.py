# backend.py
# FastAPI Asynchronous Backend for SAKEC AI, ML, DL, GenAI, RAG & Agentic Simulation Platform
# Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur

import sqlite3
import hashlib
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import requests

app = FastAPI(
    title="SAKEC AI & GenAI Virtual Simulation Platform API",
    description="Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "training_platform.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            department TEXT DEFAULT 'EXTC - SAKEC',
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            module_id TEXT NOT NULL,
            module_name TEXT NOT NULL,
            task_completed INTEGER DEFAULT 0,
            quiz_score INTEGER DEFAULT 0,
            max_score INTEGER DEFAULT 3,
            updated_at TEXT NOT NULL,
            UNIQUE(username, module_id)
        )
    """)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
    student_hash = hashlib.sha256("student123".encode()).hexdigest()
    c.execute("""
        INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("admin", "Dr. Rohan Borgalli (Admin)", "EXTC - SAKEC Chembur", admin_hash, "admin", now))
    c.execute("""
        INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("student1", "Demo Engineering Participant", "EXTC - SAKEC Chembur", student_hash, "user", now))
    conn.commit()
    conn.close()

init_db()

# Optional Local SentenceTransformer & ChromaDB Loading
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedder = None

try:
    import chromadb
    chroma_client = chromadb.Client()
except Exception:
    chroma_client = None

def encode_texts(texts: List[str]) -> np.ndarray:
    if embedder is not None:
        return np.array(embedder.encode(texts, normalize_embeddings=True))
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t.lower())) % (2**32))
        v = rng.normal(0, 0.1, 384)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(vecs)

class AuthRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    full_name: str
    department: str = "EXTC - SAKEC"
    password: str
    role: str = "user"

class ProgressPayload(BaseModel):
    username: str
    module_id: str
    module_name: str
    task_completed: bool
    quiz_score: int
    max_score: int = 3

class EmbedPayload(BaseModel):
    texts: List[str]

class LLMPayload(BaseModel):
    prompt: str
    system_prompt: str = "You are an expert AI/ML Engineering Professor."
    temperature: float = 0.3
    top_p: float = 0.9
    engine: str = "built_in_simulator"
    endpoint_url: str = "http://localhost:11434"
    model_name: str = "llama3.2:1b"

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "developer": "Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur",
        "embedder_loaded": embedder is not None,
        "chromadb_loaded": chroma_client is not None
    }

@app.post("/auth/login")
def api_login(req: AuthRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    c.execute("SELECT username, full_name, department, role FROM users WHERE username=? AND password_hash=?",
              (req.username, pw_hash))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": row[0], "full_name": row[1], "department": row[2], "role": row[3]}

@app.post("/auth/register")
def api_register(req: RegisterRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("""
            INSERT INTO users (username, full_name, department, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (req.username.strip(), req.full_name.strip(), req.department.strip(), pw_hash, req.role, now))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"status": "registered", "username": req.username}

@app.post("/progress/update")
def api_update_progress(req: ProgressPayload):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO progress (username, module_id, module_name, task_completed, quiz_score, max_score, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username, module_id) DO UPDATE SET
            task_completed = MAX(task_completed, excluded.task_completed),
            quiz_score = MAX(quiz_score, excluded.quiz_score),
            updated_at = excluded.updated_at
    """, (req.username, req.module_id, req.module_name, int(req.task_completed), req.quiz_score, req.max_score, now))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/admin/overview")
def api_admin_overview():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, full_name, department, role, created_at FROM users")
    users = c.fetchall()
    c.execute("SELECT username, module_id, module_name, task_completed, quiz_score, max_score, updated_at FROM progress")
    prog = c.fetchall()
    conn.close()
    return {
        "users": [{"username": u[0], "full_name": u[1], "department": u[2], "role": u[3], "created_at": u[4]} for u in users],
        "progress": [{"username": p[0], "module_id": p[1], "module_name": p[2], "task_completed": bool(p[3]), "quiz_score": p[4], "max_score": p[5], "updated_at": p[6]} for p in prog]
    }

@app.post("/embed")
def api_embed(req: EmbedPayload):
    vecs = encode_texts(req.texts)
    return {"shape": list(vecs.shape), "embeddings": vecs.tolist()}

@app.post("/llm/generate")
def api_llm_generate(req: LLMPayload):
    if req.engine == "local_ollama":
        try:
            resp = requests.post(
                f"{req.endpoint_url.rstrip('/')}/api/generate",
                json={
                    "model": req.model_name,
                    "system": req.system_prompt,
                    "prompt": req.prompt,
                    "stream": False,
                    "options": {"temperature": req.temperature, "top_p": req.top_p}
                },
                timeout=25
            )
            if resp.status_code == 200:
                return {"response": resp.json().get("response", ""), "source": f"Local Ollama ({req.model_name})"}
        except Exception:
            pass
    return {
        "response": (
            f"[Built-in Zero-API-Key LLM Engine | Temp={req.temperature} | Top-P={req.top_p}]
"
            f"System Context: {req.system_prompt}
"
            f"Response synthesized for prompt: {req.prompt[:180]}"
        ),
        "source": "Built-in Pedagogical LLM Simulator"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
