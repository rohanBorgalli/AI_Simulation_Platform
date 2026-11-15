# app.py
# End-to-End AI, ML, DL, GenAI, LLM, NLP, Embeddings, Vector DB, RAG & Agentic AI Simulation Platform
# Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur

import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from content_data import MODULES, REFERENCE_PLATFORMS
except ImportError:
    REFERENCE_PLATFORMS = [
        {"name": "TensorFlow Neural Network Playground", "category": "ML & Deep Learning", "url": "https://playground.tensorflow.org/", "desc": "Interactive browser simulation of neural network layers, activations, learning rates, and non-linear decision boundaries."},
        {"name": "Machine Learning Playground", "category": "Classical ML Classifiers", "url": "https://ml-playground.com/", "desc": "Visual experimentation with SVM, KNN, Decision Trees, and MLP hyperparameters on synthetic 2D datasets."},
        {"name": "Seeing Theory (Brown University)", "category": "Probability & Statistics", "url": "https://seeing-theory.brown.edu/", "desc": "Visual intuition for probability distributions, Bayesian inference, and ordinary least squares regression."},
        {"name": "SPIT Virtual Lab for AI & Deep Learning", "category": "Academic AI/DL Virtual Lab", "url": "https://vlab.spit.ac.in/", "desc": "Curriculum-aligned engineering virtual laboratory with step-by-step pre-tests, simulations, and post-tests."},
        {"name": "Google TensorFlow Embedding Projector", "category": "Embeddings & Dimensionality Reduction", "url": "https://projector.tensorflow.org/", "desc": "Interactive 2D/3D exploration of high-dimensional word and sentence embeddings via PCA, t-SNE, and UMAP."},
        {"name": "Cohere Embed & Rerank Playground", "category": "Semantic Search & Ranking", "url": "https://dashboard.cohere.com/", "desc": "Visual comparison of dense semantic embeddings, cosine similarity clustering, and cross-encoder reranking."},
        {"name": "Dify.ai / Pinecone RAG Workbench", "category": "RAG & Agentic Workflows", "url": "https://dify.ai/", "desc": "Visual pipeline orchestration showing document chunking, vector database retrieval, and ReAct agent tool calling."}
    ]
    MODULES = {}

# ==========================================
# 1. PAGE CONFIG & ACADEMIC HEADER/FOOTER CSS
# ==========================================
st.set_page_config(
    page_title="AI & GenAI Virtual Simulation Lab | Dr. Rohan Borgalli - SAKEC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BRAND_ATTRIBUTION = "Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur"

st.markdown(f"""
<style>
    .sakec-header {{
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 52%, #0f172a 100%);
        color: #f8fafc;
        padding: 15px 24px;
        border-radius: 10px;
        border-left: 6px solid #38bdf8;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.16);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .sakec-header h2 {{
        margin: 0;
        font-size: 1.32rem;
        font-weight: 700;
        color: #ffffff;
    }}
    .sakec-header p {{
        margin: 4px 0 0 0;
        font-size: 0.92rem;
        color: #bae6fd;
        font-weight: 600;
    }}
    .sakec-footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0f172a;
        color: #f1f5f9;
        text-align: center;
        padding: 9px 16px;
        font-size: 0.86rem;
        font-weight: 600;
        border-top: 2px solid #38bdf8;
        z-index: 9999;
    }}
    .block-container {{
        padding-bottom: 80px !important;
    }}
    .step-card {{
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #0f172a;
    }}
</style>

<div class="sakec-header">
    <div>
        <h2>🧠 End-to-End AI, ML, DL, GenAI, LLM, NLP, Vector DB, RAG & Agentic AI Virtual Simulation Lab</h2>
        <p>🏛️ {BRAND_ATTRIBUTION}</p>
    </div>
    <div style="text-align: right; font-size: 0.82rem; color: #e2e8f0;">
        <b>Multi-Engine Architecture</b><br>
        <span style="color:#38bdf8;">Google Gemini API • Local Ollama • Built-in Simulator</span>
    </div>
</div>

<div class="sakec-footer">
    🎓 {BRAND_ATTRIBUTION} | Interactive AI, Deep Learning, GenAI, RAG & Agentic Training Platform
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. BUILT-IN FALLBACK DATASETS & DOCUMENTS
# ==========================================
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

DEFAULT_PARAM_SHAVAK_DOC = (
    "SAKEC High-Performance AI & PARAM Shavak Laboratory Charter (2026-27)\n"
    "Department of Electronics & Telecommunication Engineering | Shah & Anchor Kutchhi Engineering College, Chembur\n\n"
    "1. Infrastructure Overview: The PARAM Shavak supercomputing facility at Shah & Anchor Kutchhi Engineering College (SAKEC), Chembur, "
    "provides high-density multi-core CPU and GPU accelerator compute dedicated to Deep Learning, Vision-Language Models, and "
    "Retrieval-Augmented Generation (RAG) research.\n\n"
    "2. Multi-Engine LLM & Zero-API-Key Policy: During student workshops and Faculty Development Programs (FDPs), training workloads "
    "can execute on Google Gemini Flash/Pro API, local quantized models via Ollama (llama3.2:1b, qwen2.5:1.5b, deepseek-r1), or "
    "built-in deterministic simulators alongside 384-dimensional embeddings (all-MiniLM-L6-v2).\n\n"
    "3. Standard RAG Ingestion Protocol: When indexing technical manuals or OBE curriculum documents into ChromaDB or FAISS, "
    "practitioners configure a sliding window chunk size between 180 and 512 characters/tokens with a 15% to 20% chunk overlap "
    "(typically 35 to 64 characters). This prevents semantic loss across section boundaries.\n\n"
    "4. Vector Similarity & Retrieval Thresholds: All document chunks are normalized to unit L2 length prior to indexing. "
    "During semantic retrieval, Top-K (typically K=3) chunks are ranked using Cosine Similarity. Any chunk scoring below 0.30 "
    "Cosine Similarity is flagged as low-confidence context.\n\n"
    "5. Agentic AI Safety & Governance: Autonomous ReAct agents deployed in academic workflows restrict tool execution to "
    "whitelisted deterministic sandboxes (Python mathematical evaluator, EV battery range calculator, and verified curriculum repositories) "
    "and log every Thought-Action-Observation trace."
)

DEFAULT_EV_COURSE_DOC = (
    "Course Manual: Introduction to Electric Vehicles (Course Code: T26104.3)\n"
    "First-Year Engineering Curriculum | SAKEC EXTC Department | Academic Year 2026-27\n\n"
    "Module 1 - EV Powertrain Architecture: An Electric Vehicle powertrain integrates a High-Voltage Energy Storage System "
    "(Lithium-Ion or LFP battery pack), a Battery Management System (BMS), a DC-DC Buck/Boost converter, a Three-Phase Sinusoidal PWM "
    "(SPWM) Inverter, and a traction motor (BLDC or PMSM).\n\n"
    "Module 2 - Battery Equivalent Circuit & State of Charge (SoC): The Thevenin Equivalent Circuit Model (ECM) represents a battery cell "
    "using an Open-Circuit Voltage source Voc(SoC) in series with an ohmic resistance R0 and an RC polarization network (R1, C1). "
    "Coulomb counting integrates load current over time to track SoC.\n\n"
    "Module 3 - Passive & Active Cell Balancing: Manufacturing variations cause individual series cells to drift in State of Charge. "
    "Passive BMS balancing dissipates excess energy from higher-voltage cells through bleed resistors, whereas active balancing "
    "transfers charge inductively or capacitively between cells.\n\n"
    "Module 4 - Range Estimation & Regenerative Braking: Vehicle energy consumption is typically 130 to 160 Wh/km for compact passenger EVs. "
    "For a 96-series cell pack (3.7V nominal, 50Ah capacity), total pack voltage is 355.2V and nominal capacity is 17.76 kWh, "
    "yielding an estimated driving range of 118.4 km at 150 Wh/km."
)

# ==========================================
# 3. SQLITE DATABASE & RBAC PROGRESS ENGINE
# ==========================================
DB_PATH = "training_platform.db"

def init_database():
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
    c.execute(
        "INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("admin", "Dr. Rohan Borgalli (Admin)", "EXTC - SAKEC Chembur", admin_hash, "admin", now)
    )
    c.execute(
        "INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("student1", "Demo Participant (Trainee)", "EXTC - SAKEC Chembur", student_hash, "user", now)
    )
    conn.commit()
    conn.close()

init_database()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute(
        "SELECT username, full_name, department, role FROM users WHERE username=? AND password_hash=?",
        (username.strip(), pw_hash)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {"username": row[0], "full_name": row[1], "department": row[2], "role": row[3]}
    return None

def register_new_user(username, full_name, department, password, role="user"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute(
            "INSERT INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username.strip(), full_name.strip(), department.strip(), pw_hash, role, now)
        )
        conn.commit()
        conn.close()
        return True, "Account registered successfully! Please sign in."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists. Please choose a different username."

def save_module_progress(username, module_id, module_name, task_done=False, quiz_score=0, max_score=3):
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
    """, (username, module_id, module_name, int(task_done), int(quiz_score), int(max_score), now))
    conn.commit()
    conn.close()

def fetch_user_progress(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT module_id, task_completed, quiz_score, max_score, updated_at FROM progress WHERE username=?", (username,))
    rows = c.fetchall()
    conn.close()
    return {r[0]: {"task_completed": bool(r[1]), "quiz_score": r[2], "max_score": r[3], "updated_at": r[4]} for r in rows}

# ==========================================
# 4. EMBEDDING & MULTI-ENGINE LLM ROUTER (GEMINI + OLLAMA + BUILT-IN)
# ==========================================
@st.cache_resource(show_spinner=False)
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

def compute_embeddings(texts):
    use_st = st.session_state.get("use_local_st", False)
    if use_st:
        model = get_sentence_transformer()
        if model is not None:
            return np.array(model.encode(texts, normalize_embeddings=True)), "SentenceTransformer (all-MiniLM-L6-v2 | 384-D)"

    domain_anchors = {
        "electric": np.array([1.2, 0.1, -0.6, 0.2]),
        "ev": np.array([1.2, 0.15, -0.55, 0.25]),
        "battery": np.array([1.15, 0.2, -0.5, 0.3]),
        "lithium": np.array([1.1, 0.22, -0.48, 0.28]),
        "bldc": np.array([1.05, 0.18, -0.52, 0.22]),
        "motor": np.array([1.0, 0.12, -0.5, 0.2]),
        "regenerative": np.array([1.08, 0.16, -0.45, 0.24]),
        "voltage": np.array([1.12, 0.19, -0.49, 0.27]),
        "range": np.array([1.06, 0.17, -0.46, 0.23]),
        "neural": np.array([-0.5, 1.2, 0.3, -0.2]),
        "deep": np.array([-0.48, 1.15, 0.32, -0.18]),
        "transformer": np.array([-0.45, 1.18, 0.4, -0.15]),
        "attention": np.array([-0.42, 1.12, 0.38, -0.12]),
        "rag": np.array([-0.35, 1.1, 0.55, 0.1]),
        "vector": np.array([-0.38, 1.08, 0.52, 0.12]),
        "chromadb": np.array([-0.36, 1.05, 0.58, 0.15]),
        "embedding": np.array([-0.4, 1.1, 0.5, 0.08]),
        "param": np.array([-0.32, 1.02, 0.6, 0.18]),
        "shavak": np.array([-0.30, 1.00, 0.62, 0.20]),
        "chunk": np.array([-0.34, 1.04, 0.56, 0.14]),
        "ollama": np.array([-0.33, 1.03, 0.57, 0.16]),
        "food": np.array([-0.4, -0.7, -1.1, 0.8]),
        "biryani": np.array([-0.45, -0.75, -1.15, 0.85]),
        "vada": np.array([-0.42, -0.72, -1.12, 0.82]),
        "masala": np.array([-0.38, -0.68, -1.08, 0.78]),
        "spices": np.array([-0.44, -0.74, -1.10, 0.80]),
        "puran": np.array([-0.41, -0.71, -1.09, 0.81]),
    }
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t.lower())) % (2**32))
        v = rng.normal(0, 0.05, 384)
        tokens = re.findall(r"[a-z0-9]+", t.lower())
        for tok in tokens:
            for kw, anchor in domain_anchors.items():
                if kw in tok:
                    v[:4] += anchor * 0.95
            idx = 4 + (abs(hash(tok)) % 60)
            v[idx] += 0.45
        v = v / (np.linalg.norm(v) + 1e-9)
        vecs.append(v)
    return np.array(vecs), "Built-in Deterministic 384-D Semantic Vector Engine (Zero-API-Key)"

def call_google_gemini_api(prompt, system_prompt, temperature, top_p, api_key, model_name="gemini-1.5-flash"):
    """Calls Google Gemini REST API directly via requests (Zero extra SDK dependency required)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "topP": float(top_p),
            "maxOutputTokens": 1024
        }
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_out = "".join(p.get("text", "") for p in parts)
            if text_out:
                return text_out, f"Google Gemini API ({model_name})"
        return "Gemini returned an empty response.", f"Google Gemini API ({model_name})"
    else:
        err_msg = resp.text[:300]
        raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {err_msg}")

def execute_llm(prompt, system_prompt, temperature, top_p):
    mode = st.session_state.get("exec_mode", "Built-in Interactive Simulator (Zero Setup)")

    # 1. Google Gemini API Mode (or automatic if key provided)
    gemini_key = st.session_state.get("gemini_api_key", "").strip()
    if not gemini_key:
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            gemini_key = ""

    gemini_model = st.session_state.get("gemini_model", "gemini-1.5-flash")

    if "Gemini" in mode:
        if gemini_key:
            try:
                return call_google_gemini_api(prompt, system_prompt, temperature, top_p, gemini_key, gemini_model)
            except Exception as e:
                st.warning(f"Google Gemini API call failed ({e}). Falling back to Built-in Pedagogical LLM Simulator.")
        else:
            st.warning("Please enter your Google Gemini API Key in the sidebar (or switch to Built-in Simulator). Using Built-in Simulator for now.")

    # 2. Local Ollama / Cluster Mode
    ollama_url = st.session_state.get("ollama_url", "http://localhost:11434")
    ollama_model = st.session_state.get("ollama_model", "llama3.2:1b")
    if "Ollama" in mode:
        try:
            resp = requests.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={
                    "model": ollama_model,
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": float(temperature), "top_p": float(top_p)}
                },
                timeout=20
            )
            if resp.status_code == 200:
                return resp.json().get("response", ""), f"Live Ollama / Cluster ({ollama_model})"
        except Exception as e:
            st.warning(f"Local Ollama endpoint `{ollama_url}` unreachable ({e}). Falling back to Built-in Pedagogical LLM Simulator.")

    # 3. Context-Aware Built-in Pedagogical LLM Simulator (Works 100% offline/free)
    if "RETRIEVED CONTEXT:" in prompt:
        ctx_part = prompt.split("RETRIEVED CONTEXT:")[1].split("USER QUESTION:")[0].strip()
        q_part = prompt.split("USER QUESTION:")[-1].replace("GROUNDED ANSWER:", "").strip()
        return (
            f"**Grounded RAG Synthesis (Temp={temperature}, Top-P={top_p}):**\n\n"
            f"Based strictly on the retrieved ChromaDB context chunks for your question (*\"{q_part}\"*):\n\n"
            f"{ctx_part}\n\n"
            f"* **Grounded Verification:** Every technical parameter above is extracted directly from the highest-scoring Cosine Similarity chunks in Stage 4, preventing hallucination."
        ), "Built-in Zero-API-Key RAG Synthesizer"

    if "96 cells" in prompt or "3.7V" in prompt:
        return (
            f"**Step-by-Step Chain-of-Thought Engineering Derivation (Temp={temperature}):**\n\n"
            f"1. **Total Pack Nominal Voltage:** $V_{{\\text{{pack}}}} = 96 \\times 3.7\\text{{ V}} = 355.2\\text{{ V}}$\n"
            f"2. **Total Pack Energy Capacity:** $E_{{\\text{{pack}}}} = \\frac{{355.2\\text{{ V}} \\times 50\\text{{ Ah}}}}{{1000}} = 17.76\\text{{ kWh}}$\n"
            f"3. **Estimated Driving Range (at 150 Wh/km):** $\\text{{Range}} = \\frac{{17,760\\text{{ Wh}}}}{{150\\text{{ Wh/km}}}} = 118.4\\text{{ km}}$"
        ), "Built-in Zero-API-Key LLM Engine"

    return (
        f"**[Pedagogical LLM Output | Temp={temperature} | Top-P={top_p}]**\n\n"
        f"* **System Persona Active:** *{system_prompt}*\n"
        f"* **Synthesized Response:** For the input prompt (`{prompt[:140]}...`), the autoregressive Transformer scales raw vocabulary logits by $1/T = 1/{max(0.05, temperature):.2f}$, filters the top {int(top_p*100)}% nucleus probability mass, and generates context-aligned tokens."
    ), "Built-in Zero-API-Key LLM Engine"

def compute_dynamic_next_token_probs(prompt_text, temperature):
    """Dynamically computes candidate next tokens and Softmax probabilities based on user prompt content."""
    p_low = prompt_text.lower()
    if any(k in p_low for k in ["ev", "battery", "voltage", "cell", "motor", "96"]):
        tokens = ["355.2V", "battery", "kWh", "powertrain", "thermal"]
        logits = np.array([3.5, 2.7, 2.1, 1.4, 0.6])
    elif any(k in p_low for k in ["rag", "chunk", "vector", "chromadb", "embedding"]):
        tokens = ["Cosine", "ChromaDB", "embeddings", "retrieval", "context"]
        logits = np.array([3.4, 2.8, 2.2, 1.5, 0.7])
    elif any(k in p_low for k in ["facial", "cnn", "convolutional", "vision", "image"]):
        tokens = ["Computer_Vision", "Deep_Learning", "CNN", "features", "accuracy"]
        logits = np.array([3.6, 2.6, 1.9, 1.2, 0.5])
    else:
        tokens = ["model", "neural", "learning", "system", "analysis"]
        logits = np.array([3.1, 2.4, 1.8, 1.1, 0.4])

    scaled = np.exp(logits / max(0.05, float(temperature)))
    probs = scaled / scaled.sum()
    return tokens, logits, probs

# ==========================================
# 5. LOGIN & REGISTRATION VIEW
# ==========================================
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.markdown("### 🎯 Welcome to the SAKEC End-to-End AI & GenAI Virtual Simulation Lab")
        st.write(
            "Designed for interactive training sessions, workshops, and Faculty Development Programs (FDPs), "
            "this platform provides live mathematical and visual sandboxes across **Machine Learning, Deep Learning, "
            "NLP, Dense Embeddings, Vector Databases, Prompt Engineering, RAG, and Agentic AI**."
        )
        st.info(
            "**🔑 Pre-Configured Demo Accounts:**\n"
            "- **Admin / Instructor Access:** Username: `admin` | Password: `admin123`\n"
            "- **Participant / Student Access:** Username: `student1` | Password: `student123`"
        )
        st.markdown("#### 🌟 Integrated Reference Pedagogies")
        for ref in REFERENCE_PLATFORMS[:5]:
            st.markdown(f"- **[{ref['name']}]({ref['url']})** (*{ref['category']}*): {ref['desc']}")

    with right_col:
        t_login, t_reg = st.tabs(["🔐 Login to Platform", "📝 Register New Participant"])
        with t_login:
            with st.form("login_form"):
                u_in = st.text_input("Username", value="admin")
                p_in = st.text_input("Password", type="password", value="admin123")
                btn_login = st.form_submit_button("Sign In to Simulation Lab", use_container_width=True)
                if btn_login:
                    auth_u = authenticate_user(u_in, p_in)
                    if auth_u:
                        st.session_state.user = auth_u
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        with t_reg:
            with st.form("reg_form"):
                r_user = st.text_input("Choose Username")
                r_name = st.text_input("Full Name (For Progress Certificate)")
                r_dept = st.text_input("Department / College", value="EXTC - SAKEC Chembur")
                r_pass = st.text_input("Password", type="password")
                r_role = st.selectbox("Select Role", ["user", "admin"], format_func=lambda r: "Participant / Trainee" if r == "user" else "Instructor / Admin")
                btn_reg = st.form_submit_button("Register Account", use_container_width=True)
                if btn_reg:
                    if not r_user or not r_name or not r_pass:
                        st.warning("Please complete all required fields.")
                    else:
                        ok, msg = register_new_user(r_user, r_name, r_dept, r_pass, r_role)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
    st.stop()

# ==========================================
# 6. SIDEBAR: USER PROFILE, PROGRESS & GEMINI / OLLAMA ENGINE ROUTER
# ==========================================
user = st.session_state.user
user_prog = fetch_user_progress(user["username"])

with st.sidebar:
    st.markdown(f"### 👤 {user['full_name']}")
    st.caption(f"Role: **{user['role'].upper()}** | {user['department']}")

    done_mods = sum(1 for m_k in MODULES if user_prog.get(m_k, {}).get("task_completed") or user_prog.get(m_k, {}).get("quiz_score", 0) > 0)
    pct_done = int((done_mods / max(1, len(MODULES))) * 100)
    st.progress(pct_done / 100.0, text=f"Completion: {done_mods}/{len(MODULES)} Modules ({pct_done}%)")

    st.divider()
    nav_keys = list(MODULES.keys())
    nav_titles = {k: MODULES[k]["title"] for k in nav_keys}
    nav_keys.append("ref_platforms")
    nav_titles["ref_platforms"] = "🌐 Reference Platforms & Architecture"
    if user["role"] == "admin":
        nav_keys.insert(0, "admin_panel")
        nav_titles["admin_panel"] = "🛡️ Admin Analytics & Cohort Tracker"

    active_page = st.radio("📚 Training Curriculum Modules", nav_keys, format_func=lambda k: nav_titles[k])

    st.divider()
    st.markdown("### ⚙️ LLM & Embedding Engine Router")
    st.session_state.exec_mode = st.selectbox(
        "Select LLM Execution Backend",
        [
            "Google Gemini API (Live Cloud LLM)",
            "Built-in Interactive Simulator (Zero API Key)",
            "Local Ollama / PARAM Shavak Cluster Endpoint"
        ],
        index=0
    )

    if "Gemini" in st.session_state.exec_mode:
        default_gem_key = st.session_state.get("gemini_api_key", "")
        st.session_state.gemini_api_key = st.text_input(
            "🔑 Google Gemini API Key",
            value=default_gem_key,
            type="password",
            help="Paste your Google AI Studio API Key (AIza...). If left blank, the platform automatically falls back to the Built-in Simulator."
        )
        st.session_state.gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
            index=0
        )
        if st.session_state.gemini_api_key:
            st.success("✅ Gemini API Key Configured")
        else:
            st.caption("ℹ️ Enter a Gemini key above or leave blank to use the automatic Built-in Simulator fallback.")

    elif "Ollama" in st.session_state.exec_mode:
        st.session_state.ollama_url = st.text_input("Ollama / Cluster Endpoint URL", value="http://localhost:11434")
        st.session_state.ollama_model = st.text_input("Model Name", value="llama3.2:1b")

    st.session_state.use_local_st = st.checkbox("Use Local HuggingFace SentenceTransformer (all-MiniLM-L6-v2)", value=False)

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.user = None
        st.rerun()

# ==========================================
# 7. ADMIN ANALYTICS & PARTICIPANT TRACKER
# ==========================================
if active_page == "admin_panel":
    st.subheader("🛡️ Admin Analytics & Participant Progress Tracker")
    conn = sqlite3.connect(DB_PATH)
    df_users = pd.read_sql_query("SELECT username, full_name, department, role, created_at FROM users", conn)
    df_prog = pd.read_sql_query("SELECT username, module_id, module_name, task_completed, quiz_score, max_score, updated_at FROM progress", conn)
    conn.close()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Registered Users", len(df_users))
    k2.metric("Participants (Trainees)", len(df_users[df_users["role"] == "user"]))
    k3.metric("Logged Module Completions", len(df_prog))
    cohort_acc = f"{(df_prog['quiz_score'].sum() / max(1, df_prog['max_score'].sum()) * 100):.1f}%" if not df_prog.empty else "0.0%"
    k4.metric("Average Cohort Quiz Score", cohort_acc)

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("#### 👥 Registered Participants Roster")
        st.dataframe(df_users, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown("#### 📊 Module-Wise Quiz Performance")
        if not df_prog.empty:
            fig_bar = px.bar(
                df_prog.groupby("module_name", as_index=False)["quiz_score"].mean(),
                x="module_name", y="quiz_score", color="quiz_score",
                title="Mean Quiz Score by Module (Out of 3)", color_continuous_scale="Teal"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No quiz submissions recorded yet. Complete a module quiz to populate cohort charts.")

    st.markdown("#### 📋 Detailed Participant Progress Log")
    if not df_prog.empty:
        st.dataframe(df_prog, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Cohort Progress Report (CSV)",
            data=df_prog.to_csv(index=False).encode("utf-8"),
            file_name="SAKEC_AI_Simulation_Cohort_Progress.csv",
            mime="text/csv"
        )
    st.stop()

# ==========================================
# 8. REFERENCE PLATFORMS HUB
# ==========================================
if active_page == "ref_platforms":
    st.subheader("🌐 Reference Simulation Platforms & Pedagogical Benchmarks")
    st.write("This platform integrates simulation paradigms from the following benchmark AI/ML educational tools:")
    st.dataframe(pd.DataFrame(REFERENCE_PLATFORMS), use_container_width=True, hide_index=True)
    st.stop()

# ==========================================
# 9. 4-PILLAR MODULE WORKSPACE
# ==========================================
mod = MODULES[active_page]
st.subheader(mod["title"])
st.caption(mod["subtitle"])

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 1. Theory & Mathematical Concepts",
    "🧪 2. Live Interactive Simulation Sandbox",
    "✅ 3. Knowledge Check & Hands-On Task",
    "🔗 4. Reference Platforms & Further Reading"
])

with tab1:
    st.markdown(mod["theory"])

with tab4:
    st.markdown("#### Benchmark Platforms & Further Reading")
    for r in REFERENCE_PLATFORMS:
        st.markdown(f"- **[{r['name']}]({r['url']})** (*{r['category']}*): {r['desc']}")

# ==========================================
# TAB 2: LIVE SIMULATIONS FOR ALL 6 MODULES
# ==========================================
with tab2:
    # ------------------------------------------
    # MODULE 1: AI, ML & DEEP LEARNING SANDBOX
    # ------------------------------------------
    if active_page == "mod1_ml_dl":
        col_ctl, col_viz = st.columns([1, 2.2])
        with col_ctl:
            st.markdown("##### ⚙️ Dataset & Neural Network Controls")
            ds_opt = st.selectbox(
                "Select Live Dataset",
                [
                    "Preloaded CSV: EV Battery Diagnostics (SAKEC Lab)",
                    "Preloaded CSV: Student AI Readiness",
                    "Synthetic Non-Linear Moons",
                    "Synthetic Concentric Circles",
                    "Upload Custom CSV (2 Numeric Columns + 1 Target)"
                ]
            )
            custom_csv = None
            if "Upload" in ds_opt:
                custom_csv = st.file_uploader("Upload CSV File", type=["csv"])

            lr = st.select_slider("Learning Rate (η)", options=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2], value=0.03)
            epochs = st.slider("Training Epochs", 20, 300, 120, step=10)
            hidden_n = st.slider("Hidden Neurons per Layer (2 Layers)", 2, 32, 10)
            act_fn = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
            reg_alpha = st.select_slider("L2 Regularization (α)", options=[0.0, 0.0001, 0.001, 0.01, 0.1], value=0.001)

        with col_viz:
            if custom_csv is not None:
                df_d = pd.read_csv(custom_csv).dropna()
                X = df_d.iloc[:, :2].values.astype(float)
                y = pd.factorize(df_d.iloc[:, 2])[0]
                feat_names = [df_d.columns[0], df_d.columns[1]]
            elif "EV Battery" in ds_opt and os.path.exists(os.path.join(SAMPLE_DIR, "ev_battery_diagnostics.csv")):
                df_d = pd.read_csv(os.path.join(SAMPLE_DIR, "ev_battery_diagnostics.csv"))
                X = df_d.iloc[:, :2].values.astype(float)
                y = pd.factorize(df_d.iloc[:, 2])[0]
                feat_names = ["Cell_Temp_C", "Internal_Resistance_mOhm"]
            elif "Student AI" in ds_opt and os.path.exists(os.path.join(SAMPLE_DIR, "student_ai_readiness.csv")):
                df_d = pd.read_csv(os.path.join(SAMPLE_DIR, "student_ai_readiness.csv"))
                X = df_d.iloc[:, :2].values.astype(float)
                y = pd.factorize(df_d.iloc[:, 2])[0]
                feat_names = ["Weekly_Practice_Hours", "Lab_Coding_Index"]
            elif "Circles" in ds_opt:
                X, y = make_circles(n_samples=220, noise=0.15, factor=0.5, random_state=42)
                feat_names = ["Feature X1", "Feature X2"]
            else:
                X, y = make_moons(n_samples=220, noise=0.18, random_state=42)
                feat_names = ["Feature X1", "Feature X2"]

            X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std

            clf = MLPClassifier(
                hidden_layer_sizes=(hidden_n, hidden_n),
                activation=act_fn,
                learning_rate_init=lr,
                alpha=reg_alpha,
                max_iter=epochs,
                random_state=42
            )
            clf.fit(X_norm, y)
            acc = clf.score(X_norm, y) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Classification Accuracy", f"{acc:.1f}%")
            m2.metric("Final Cross-Entropy Loss", f"{clf.loss_:.4f}")
            m3.metric("Trainable Weights", f"{(2*hidden_n) + (hidden_n*hidden_n) + hidden_n}")

            p1, p2 = st.columns(2)
            with p1:
                fig_l = px.line(y=clf.loss_curve_, labels={"index": "Epoch", "y": "Cross-Entropy Loss"}, title="Gradient Descent Convergence Curve")
                st.plotly_chart(fig_l, use_container_width=True)
            with p2:
                x_min, x_max = X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5
                y_min, y_max = X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 75), np.linspace(y_min, y_max, 75))
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

                fig_b = go.Figure()
                fig_b.add_trace(go.Contour(x=np.linspace(x_min, x_max, 75), y=np.linspace(y_min, y_max, 75), z=Z,
                                           colorscale="RdBu", opacity=0.45, showscale=False))
                fig_b.add_trace(go.Scatter(x=X_norm[:, 0], y=X_norm[:, 1], mode="markers",
                                           marker=dict(color=y, colorscale="RdBu", size=8, line=dict(width=1, color="black"))))
                fig_b.update_layout(title="Live Neural Decision Boundary", xaxis_title=f"{feat_names[0]} (Normalized)", yaxis_title=f"{feat_names[1]} (Normalized)")
                st.plotly_chart(fig_b, use_container_width=True)

    # ------------------------------------------
    # MODULE 2: NLP & EMBEDDINGS SANDBOX
    # ------------------------------------------
    elif active_page == "mod2_nlp_embed":
        corpus_path = os.path.join(SAMPLE_DIR, "multidomain_embedding_corpus.txt")
        default_txt = open(corpus_path, "r", encoding="utf-8").read() if os.path.exists(corpus_path) else (
            "Electric vehicles use lithium-ion battery packs and BLDC motors.\n"
            "Regenerative braking recovers kinetic energy during urban deceleration in Mumbai traffic.\n"
            "Deep learning neural networks learn hierarchical representations.\n"
            "Transformer architectures apply multi-head self-attention mechanisms to tokens.\n"
            "Spicy Mumbai vada pav and masala dosa are iconic Indian street foods.\n"
            "Hyderabadi dum biryani is slow-cooked with basmati rice and aromatic spices."
        )
        raw_inp = st.text_area("Enter Sentences to Tokenize & Embed (One sentence per line):", value=default_txt, height=180)
        sents = [s.strip() for s in raw_inp.splitlines() if s.strip()]

        if len(sents) >= 3:
            vecs, emb_name = compute_embeddings(sents)
            st.caption(f"Active Embedding Engine: **{emb_name}** | Vector Tensor Shape: `{vecs.shape}`")

            c_left, c_right = st.columns([1.35, 1])
            with c_left:
                proj_dim = st.radio("Select PCA Projection Mode", ["2D PCA Cluster Map", "3D Interactive PCA Cube"], horizontal=True)
                k_dim = 3 if "3D" in proj_dim else 2
                pca_coords = PCA(n_components=min(k_dim, len(sents))).fit_transform(vecs)
                df_pca = pd.DataFrame({"Sentence": sents, "Short_Label": [s[:36] + "..." for s in sents], "PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1]})
                if k_dim == 3 and pca_coords.shape[1] == 3:
                    df_pca["PC3"] = pca_coords[:, 2]
                    fig_pca = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", text="Short_Label", color="Short_Label", hover_data=["Sentence"], title="3D PCA Semantic Embedding Space")
                else:
                    fig_pca = px.scatter(df_pca, x="PC1", y="PC2", text="Short_Label", color="Short_Label", hover_data=["Sentence"], title="2D PCA Semantic Embedding Space")
                    fig_pca.update_traces(textposition="top center", marker=dict(size=12))
                st.plotly_chart(fig_pca, use_container_width=True)

            with c_right:
                st.markdown("##### 🔢 Classical TF-IDF vs. 384-D Dense Vector Inspector")
                tfidf_mat = TfidfVectorizer(stop_words="english").fit_transform(sents).toarray()
                st.write(f"Sparse TF-IDF Matrix Shape: `{tfidf_mat.shape}` | Dense Embedding Matrix Shape: `{vecs.shape}`")
                df_preview = pd.DataFrame(np.round(vecs[:, :6], 4), columns=[f"Dim_{i}" for i in range(6)])
                df_preview.insert(0, "Sentence", [s[:28] + "..." for s in sents])
                st.dataframe(df_preview, use_container_width=True, hide_index=True)

    # ------------------------------------------
    # MODULE 3: VECTOR DATABASE MECHANICS
    # ------------------------------------------
    elif active_page == "mod3_vectordb":
        default_kb = [
            "SAKEC PARAM Shavak supercomputing facility accelerates Deep Learning and RAG training workloads.",
            "Lithium-Ion Battery Management Systems prevent thermal runaway using cell balancing.",
            "ChromaDB indexes 384-dimensional embeddings using HNSW graphs for fast Cosine Similarity search.",
            "Backpropagation computes gradients of the Cross-Entropy Loss function across neural layers.",
            "Authentic Mumbai vada pav and Hyderabadi biryani are popular Indian culinary dishes."
        ]
        docs_raw = st.text_area("Vector Database Stored Documents (One per line):", value="\n".join(default_kb), height=130)
        docs = [d.strip() for d in docs_raw.splitlines() if d.strip()]
        query_str = st.text_input("🔍 Enter Search Query (Converted to Query Vector q):", value="How does ChromaDB search embeddings for RAG?")

        if docs and query_str:
            all_v, _ = compute_embeddings(docs + [query_str])
            d_vecs, q_vec = all_v[:-1], all_v[-1]

            cos_sim = np.dot(d_vecs, q_vec) / (np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-9)
            dot_prod = np.dot(d_vecs, q_vec)
            l2_dist = np.linalg.norm(d_vecs - q_vec, axis=1)

            df_rank = pd.DataFrame({
                "Stored Document": docs,
                "Cosine Similarity (↑ Best)": np.round(cos_sim, 4),
                "Dot Product": np.round(dot_prod, 4),
                "Euclidean L2 Distance (↓ Best)": np.round(l2_dist, 4),
                "Vector Slice [d0..d3]": [str(np.round(v[:4], 3).tolist()) for v in d_vecs]
            }).sort_values(by="Cosine Similarity (↑ Best)", ascending=False).reset_index(drop=True)
            df_rank.index = df_rank.index + 1
            df_rank.index.name = "Rank"

            st.markdown(f"**Query Vector $q$ Slice `[q0..q5]`:** `{np.round(q_vec[:6], 4).tolist()}`")
            st.dataframe(df_rank, use_container_width=True)

            sim_matrix = np.dot(d_vecs, d_vecs.T)
            short_labels = [f"Doc {i+1}" for i in range(len(docs))]
            fig_hm = px.imshow(np.round(sim_matrix, 2), x=short_labels, y=short_labels, text_auto=True,
                               color_continuous_scale="Blues", title="Pairwise Document Cosine Similarity Matrix")
            st.plotly_chart(fig_hm, use_container_width=True)

    # ------------------------------------------
    # MODULE 4: LLMs & PROMPT ENGINEERING (WITH SUB-TABS 4A, 4B, 4C)
    # ------------------------------------------
    elif active_page == "mod4_llm_prompt":
        sub_4a, sub_4b, sub_4c = st.tabs([
            "🎛️ 4A. Prompting Paradigms & Softmax Lab",
            "💬 4B. Multi-Turn Context Window Chat Simulator",
            "🔍 4C. Live API JSON Payload & Token Cost Inspector"
        ])

        with sub_4a:
            c_p1, c_p2 = st.columns([1, 1.75])
            with c_p1:
                sys_prompt = st.text_area(
                    "System Prompt (Role & Guardrails)",
                    value="You are an Engineering Professor at SAKEC Mumbai. Provide rigorous, step-by-step technical explanations.",
                    key="m4_sys"
                )
                temp_val = st.slider("Temperature (T)", 0.05, 1.5, 0.25, 0.05, key="m4_temp")
                top_p_val = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05, key="m4_topp")
                p_mode = st.selectbox(
                    "Prompting Paradigm Template",
                    ["Zero-Shot Prompt", "Few-Shot (In-Context Learning)", "Chain-of-Thought (Step-by-Step Engineering Math)"],
                    key="m4_mode"
                )

            with c_p2:
                prompt_presets = {
                    "Zero-Shot Prompt": "Classify the technical domain of this project title: 'Real-Time Facial Expression Recognition using Deep Convolutional Networks'.",
                    "Few-Shot (In-Context Learning)": (
                        "Title: 'Design of Solar Buck-Boost Converter' -> Domain: Power Electronics | Lab: EV & Power Lab\n"
                        "Title: 'RAG Chatbot on PARAM Shavak Cluster' -> Domain: Generative AI & NLP | Lab: High-Performance Computing Lab\n"
                        "Title: 'Real-Time Facial Expression Recognition using Deep Convolutional Networks' ->"
                    ),
                    "Chain-of-Thought (Step-by-Step Engineering Math)": (
                        "An EV battery pack has 96 cells in series, each with nominal voltage 3.7V and capacity 50Ah. "
                        "Think step-by-step: 1) Calculate total pack voltage, 2) Calculate total energy in kWh, 3) Estimate driving range at 150 Wh/km."
                    )
                }
                live_p = st.text_area("Live Prompt Input:", value=prompt_presets[p_mode], height=145, key="m4_prompt")

                cand_tokens, raw_logits, probs = compute_dynamic_next_token_probs(live_p, temp_val)
                fig_tok = px.bar(
                    x=cand_tokens, y=probs,
                    labels={"x": "Candidate Next Token", "y": "Softmax Probability P(w_t)"},
                    title=f"Dynamic Next-Token Softmax Distribution (T={temp_val})",
                    color=probs, color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_tok, use_container_width=True)

                if st.button("🚀 Run Prompt on Configured LLM Engine (Gemini / Ollama / Simulator)", use_container_width=True, key="m4_run"):
                    with st.spinner("Generating LLM response..."):
                        resp_txt, resp_src = execute_llm(live_p, sys_prompt, temp_val, top_p_val)
                    st.success(f"Executed via: **{resp_src}**")
                    st.markdown(resp_txt)

        with sub_4b:
            st.markdown("##### 💬 Multi-Turn Context Memory & Conversation History Lab")
            st.caption("Observe how LLMs maintain state by re-sending the entire conversation history array on every turn.")
            if "m4_chat_history" not in st.session_state:
                st.session_state.m4_chat_history = [
                    {"role": "user", "content": "Hello! We are designing a 48V 30Ah EV battery pack."},
                    {"role": "assistant", "content": "Great! A 48V 30Ah pack stores 1.44 kWh of nominal energy. How can I help with yourBMS or cell configuration?"}
                ]

            col_chat1, col_chat2 = st.columns([1.4, 1])
            with col_chat1:
                for msg in st.session_state.m4_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                new_chat_msg = st.text_input("Send a follow-up message in this conversation:", value="How many 3.7V Li-ion cells do we need in series for this pack?", key="m4_chat_in")
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("📤 Send Turn to LLM", use_container_width=True):
                    if new_chat_msg.strip():
                        st.session_state.m4_chat_history.append({"role": "user", "content": new_chat_msg.strip()})
                        compiled_ctx = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.m4_chat_history])
                        reply, src = execute_llm(compiled_ctx, sys_prompt, temp_val, top_p_val)
                        st.session_state.m4_chat_history.append({"role": "assistant", "content": f"*({src})*\n\n{reply}"})
                        st.rerun()
                if c_btn2.button("🗑️ Reset Conversation Memory", use_container_width=True):
                    st.session_state.m4_chat_history = []
                    st.rerun()

            with col_chat2:
                tot_chars = sum(len(m["content"]) for m in st.session_state.m4_chat_history)
                est_tokens = max(1, int(tot_chars / 4))
                st.metric("Active Context Window Usage", f"~{est_tokens} Tokens", f"{len(st.session_state.m4_chat_history)} Messages in Memory")
                st.markdown("**Context Window Array Sent on Next Turn:**")
                st.json(st.session_state.m4_chat_history)

        with sub_4c:
            st.markdown("##### 🔍 4C. Live LLM API Payload, Tokenization & Latency/Cost Inspector")
            st.write(
                "Inspect the exact JSON HTTP request body dispatched to **Google Gemini (`generateContent`)**, "
                "**OpenAI / vLLM (`chat/completions`)**, and **Local Ollama (`/api/generate`)**, along with token budget metrics."
            )
            insp_sys = st.session_state.get("m4_sys", "You are an Engineering Professor at SAKEC Mumbai.")
            insp_prompt = st.session_state.get("m4_prompt", "Explain how temperature scales logits before Softmax.")
            insp_temp = st.session_state.get("m4_temp", 0.25)
            insp_topp = st.session_state.get("m4_topp", 0.90)
            insp_gem_model = st.session_state.get("gemini_model", "gemini-1.5-flash")

            words_list = re.findall(r"\S+", insp_prompt)
            est_sys_tokens = max(1, int(len(insp_sys) / 3.8))
            est_usr_tokens = max(1, int(len(insp_prompt) / 3.8))
            total_in_tokens = est_sys_tokens + est_usr_tokens

            tk1, tk2, tk3, tk4 = st.columns(4)
            tk1.metric("System Prompt Tokens", est_sys_tokens)
            tk2.metric("User Prompt Tokens", est_usr_tokens)
            tk3.metric("Total Input Tokens", total_in_tokens)
            tk4.metric("Est. Gemini Flash Cost", f"${(total_in_tokens * 0.00000015):.7f}")

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**1. Google Gemini API Payload (`{insp_gem_model}`)**")
                gemini_json_preview = {
                    "endpoint": f"https://generativelanguage.googleapis.com/v1beta/models/{insp_gem_model}:generateContent",
                    "systemInstruction": {"parts": [{"text": insp_sys}]},
                    "contents": [{"role": "user", "parts": [{"text": insp_prompt}]}],
                    "generationConfig": {
                        "temperature": insp_temp,
                        "topP": insp_topp,
                        "maxOutputTokens": 1024
                    }
                }
                st.code(json.dumps(gemini_json_preview, indent=2), language="json")

            with p_col2:
                st.markdown("**2. Local Ollama / OpenAI-Compatible Payload**")
                ollama_json_preview = {
                    "endpoint": f"{st.session_state.get('ollama_url', 'http://localhost:11434')}/api/generate",
                    "model": st.session_state.get("ollama_model", "llama3.2:1b"),
                    "system": insp_sys,
                    "prompt": insp_prompt,
                    "stream": False,
                    "options": {
                        "temperature": insp_temp,
                        "top_p": insp_topp
                    }
                }
                st.code(json.dumps(ollama_json_preview, indent=2), language="json")

            st.markdown("**3. Subword Token ID Mapping Preview:**")
            token_df = pd.DataFrame({
                "Token Position": list(range(1, min(15, len(words_list)) + 1)),
                "Subword String": words_list[:15],
                "Simulated Vocab ID": [abs(hash(w)) % 32000 for w in words_list[:15]]
            })
            st.dataframe(token_df, use_container_width=True, hide_index=True)

    # ------------------------------------------
    # MODULE 5: END-TO-END RAG PIPELINE SIMULATOR (BULLETPROOF CLOUD + PDF)
    # ------------------------------------------
    elif active_page == "mod5_rag":
        st.markdown("##### 📄 Live 5-Stage RAG Pipeline (Select Preloaded SAKEC Documents or Upload Custom PDF/TXT)")
        doc_choice = st.selectbox(
            "Select Knowledge Source Document",
            [
                "Preloaded Document: SAKEC PARAM Shavak & AI Lab Charter (2026-27)",
                "Preloaded Document: Introduction to Electric Vehicles Course Manual (T26104.3)",
                "Upload Custom PDF or TXT Document"
            ],
            key="rag_doc_select"
        )
        up_file = None
        if "Upload" in doc_choice:
            up_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"], key="rag_uploader")

        raw_doc_text = ""
        if up_file is not None:
            try:
                if up_file.name.lower().endswith(".pdf") and PdfReader is not None:
                    reader = PdfReader(up_file)
                    raw_doc_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                else:
                    raw_doc_text = up_file.read().decode("utf-8", errors="ignore")
            except Exception as e:
                st.warning(f"Could not parse uploaded file ({e}). Using built-in SAKEC charter document.")

        if not raw_doc_text.strip():
            if "Electric Vehicles" in doc_choice:
                pdf_p = os.path.join(SAMPLE_DIR, "Introduction_to_Electric_Vehicles_Course_Manual.pdf")
                if PdfReader is not None and os.path.exists(pdf_p):
                    try:
                        reader = PdfReader(pdf_p)
                        raw_doc_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                    except Exception:
                        raw_doc_text = DEFAULT_EV_COURSE_DOC
                else:
                    raw_doc_text = DEFAULT_EV_COURSE_DOC
            else:
                pdf_p = os.path.join(SAMPLE_DIR, "SAKEC_PARAM_Shavak_AI_Policy_2026.pdf")
                if PdfReader is not None and os.path.exists(pdf_p):
                    try:
                        reader = PdfReader(pdf_p)
                        raw_doc_text = "\n".join([p.extract_text() or "" for p in reader.pages])
                    except Exception:
                        raw_doc_text = DEFAULT_PARAM_SHAVAK_DOC
                else:
                    raw_doc_text = DEFAULT_PARAM_SHAVAK_DOC

        with st.expander("📄 View Raw Extracted Document Text", expanded=False):
            st.text_area("Extracted Source Text (Editable for live experimentation):", value=raw_doc_text, height=150, key="rag_raw_view")

        rc1, rc2, rc3 = st.columns(3)
        c_size = rc1.slider("Stage 1: Chunk Size (Chars)", 120, 450, 220, 20, key="rag_csize")
        c_over = rc2.slider("Stage 1: Chunk Overlap (Chars)", 0, 100, 40, 10, key="rag_cover")
        top_k_rag = rc3.slider("Stage 4: Top-K Chunks to Retrieve", 1, 5, 2, key="rag_topk")

        # Stage 1: Sliding-Window Chunking
        chunks = []
        stride = max(25, c_size - c_over)
        for i in range(0, len(raw_doc_text), stride):
            piece = raw_doc_text[i:i + c_size].strip()
            if len(piece) > 20:
                chunks.append(piece)

        st.markdown(
            f"<div class='step-card'><b>Stage 1 — Document Ingestion & Sliding-Window Chunking:</b> "
            f"Extracted <b>{len(raw_doc_text)} characters</b> and partitioned into <b>{len(chunks)} overlapping chunks</b> "
            f"(Chunk Size={c_size}, Overlap={c_over}, Stride={stride}).</div>",
            unsafe_allow_html=True
        )
        with st.expander(f"🔍 Inspect All {len(chunks)} Generated Document Chunks", expanded=False):
            for idx, ch in enumerate(chunks):
                st.markdown(f"**Chunk #{idx+1} ({len(ch)} chars):** `{ch}`")

        # Stage 2 & 3: Dense Embedding & Vector Store Indexing
        c_vecs, emb_engine = compute_embeddings(chunks)
        st.markdown(
            f"<div class='step-card'><b>Stage 2 & Stage 3 — Dense Embedding & Vector Store Indexing:</b> "
            f"Encoded {len(chunks)} chunks into <code>{c_vecs.shape}</code> unit-normalized vectors via <b>{emb_engine}</b> "
            f"and indexed in ChromaDB / In-Memory Vector Store.</div>",
            unsafe_allow_html=True
        )

        # Stage 4: Semantic Retrieval
        default_q = (
            "What chunk size and LLM models are recommended on the PARAM Shavak cluster?"
            if "PARAM" in doc_choice
            else "What is the total pack voltage and driving range of a 96-series 3.7V 50Ah EV battery?"
        )
        rag_q = st.text_input("❓ Enter Grounded User Question:", value=default_q, key="rag_question_input")

        if rag_q and len(chunks) > 0:
            q_v, _ = compute_embeddings([rag_q])
            q_v = q_v[0]
            sims = np.dot(c_vecs, q_v) / (np.linalg.norm(c_vecs, axis=1) * np.linalg.norm(q_v) + 1e-9)
            ranked_idx = np.argsort(sims)[::-1]
            top_idx = ranked_idx[:top_k_rag]

            st.markdown("<div class='step-card'><b>Stage 4 — Semantic Vector Retrieval & Cosine Similarity Scoring:</b></div>", unsafe_allow_html=True)
            df_chunk_scores = pd.DataFrame({
                "Chunk_ID": [f"Chunk #{i+1}" for i in range(len(chunks))],
                "Cosine_Similarity": np.round(sims, 4),
                "Retrieval_Status": ["Top-K Retrieved" if i in top_idx else "Filtered Out" for i in range(len(chunks))],
                "Preview": [c[:65] + "..." for c in chunks]
            })
            fig_rag = px.bar(
                df_chunk_scores, x="Chunk_ID", y="Cosine_Similarity", color="Retrieval_Status",
                hover_data=["Preview"],
                title="Cosine Similarity Score Across All Document Chunks"
            )
            st.plotly_chart(fig_rag, use_container_width=True)

            ctx_blocks = []
            for r_num, c_i in enumerate(top_idx, start=1):
                ctx_blocks.append(f"[Chunk #{c_i+1} | Cosine Score={sims[c_i]:.4f}]: {chunks[c_i]}")
                st.info(f"**Top-{r_num} Retrieved (Chunk #{c_i+1} | Cosine Similarity: {sims[c_i]:.4f}):**\n\n{chunks[c_i]}")

            # Stage 5: Augmented Prompt Construction & Grounded LLM Generation
            aug_prompt = (
                "You are a factual RAG assistant. Answer the user's question strictly using ONLY the retrieved context chunks below. "
                "Always cite the Chunk numbers used.\n\n"
                "RETRIEVED CONTEXT:\n" + "\n".join(ctx_blocks) + "\n\n"
                f"USER QUESTION: {rag_q}\n\nGROUNDED ANSWER:"
            )
            st.markdown("<div class='step-card'><b>Stage 5 — Augmented Prompt Construction & Grounded LLM Synthesis:</b></div>", unsafe_allow_html=True)
            st.code(aug_prompt, language="markdown")

            if st.button("✨ Synthesize Grounded RAG Answer (via Configured Gemini / Ollama / Simulator)", use_container_width=True, key="rag_synth_btn"):
                with st.spinner("Synthesizing grounded answer from retrieved chunks..."):
                    rag_ans, rag_src = execute_llm(aug_prompt, "You are a factual RAG synthesis engine. Cite chunk numbers and use only provided context.", 0.1, 0.9)
                st.success(f"Synthesized via: **{rag_src}**")
                st.markdown(rag_ans)

    # ------------------------------------------
    # MODULE 6: AGENTIC AI & REACT TOOL CALLING
    # ------------------------------------------
    elif active_page == "mod6_agentic":
        st.markdown("##### 🤖 Autonomous ReAct Agent Loop (`Thought -> Action -> Observation -> Final Answer`)")
        st.write(
            "Registered Deterministic Tools: **`python_calculator(expr)`**, **`ev_range_simulator(cells, ah)`**, "
            "**`sakec_policy_search(query)`**, **`mumbai_weather_sensor(area)`**"
        )
        ag_query = st.text_input(
            "Enter a Multi-Tool Compound Engineering Goal:",
            value="Calculate 96 * 3.7 for our EV pack voltage, check the weather in Chembur Mumbai, and retrieve the SAKEC PARAM Shavak RAG policy."
        )
        if st.button("⚙️ Execute Autonomous ReAct Agent Loop", use_container_width=True):
            trace_steps = []
            final_points = []

            math_exprs = re.findall(r"(\d+(?:\.\d+)?\s*[\*\+\-\/]\s*\d+(?:\.\d+)?)", ag_query)
            if math_exprs:
                expr = math_exprs[0]
                val = round(float(eval(expr, {"__builtins__": {}})), 3)
                trace_steps.append(("🧠 Thought 1", f"Detected mathematical expression `{expr}`. Calling deterministic `python_calculator` tool."))
                trace_steps.append(("🛠️ Action 1", f"`python_calculator(expression='{expr}')`"))
                trace_steps.append(("👁️ Observation 1", f"`Result = {val}`"))
                final_points.append(f"**Exact Calculation (`{expr}`):** `{val}`")

            if any(k in ag_query.lower() for k in ["weather", "chembur", "mumbai", "temperature"]):
                trace_steps.append(("🧠 Thought 2", "User requested live environmental telemetry for Chembur, Mumbai. Invoking `mumbai_weather_sensor`."))
                trace_steps.append(("🛠️ Action 2", "`mumbai_weather_sensor(location='Chembur, Mumbai')`"))
                w_obs = "Chembur, Mumbai: 31.2°C, Relative Humidity 68%, Clear Sky (Simulated Live Sensor)"
                trace_steps.append(("👁️ Observation 2", f"`{w_obs}`"))
                final_points.append(f"**Environmental Telemetry:** {w_obs}")

            if any(k in ag_query.lower() for k in ["sakec", "param", "policy", "rag", "shavak"]):
                trace_steps.append(("🧠 Thought 3", "Query references SAKEC institutional infrastructure. Calling `sakec_policy_search`."))
                trace_steps.append(("🛠️ Action 3", "`sakec_policy_search(topic='PARAM Shavak & RAG Charter')`"))
                kb_obs = "PARAM Shavak cluster supports Google Gemini API, local Ollama LLMs (llama3.2:1b, qwen2.5:1.5b), and 384-D MiniLM embeddings with 180-512 char chunking."
                trace_steps.append(("👁️ Observation 3", f"`{kb_obs}`"))
                final_points.append(f"**SAKEC Policy Observation:** {kb_obs}")

            for title_s, body_s in trace_steps:
                st.markdown(f"<div class='step-card'><b>{title_s}:</b> {body_s}</div>", unsafe_allow_html=True)

            # Also allow Gemini to synthesize the final agent answer if Gemini API key is active
            if st.session_state.get("gemini_api_key", "").strip() and "Gemini" in st.session_state.get("exec_mode", ""):
                agent_summary_prompt = "Synthesize a concise final response from these verified tool observations:\n" + "\n".join(final_points)
                gem_ans, gem_src = execute_llm(agent_summary_prompt, "You are an autonomous ReAct engineering agent.", 0.2, 0.9)
                st.success(f"🎯 **Final Agent Response (Synthesized via {gem_src}):**\n\n{gem_ans}")
            else:
                st.success("🎯 **Final Agent Response (Synthesized from Verified Tool Observations):**\n\n- " + "\n- ".join(final_points))

# ==========================================
# TAB 3: HANDS-ON TASK VERIFICATION & GRADED QUIZ
# ==========================================
with tab3:
    st.markdown("#### 🛠️ Step 1: Verify Hands-On Sandbox Task")
    st.info(mod["task_prompt"])
    mod_state = user_prog.get(active_page, {})
    already_verified = mod_state.get("task_completed", False)

    if st.button("✅ Confirm & Record Sandbox Task Completion", disabled=already_verified):
        save_module_progress(
            user["username"], active_page, mod["title"],
            task_done=True, quiz_score=mod_state.get("quiz_score", 0), max_score=len(mod["quiz"])
        )
        st.success("Hands-on task completion saved to your participant profile!")
        st.rerun()

    st.divider()
    st.markdown("#### 📝 Step 2: Graded Concept Check Quiz")
    with st.form(f"quiz_{active_page}"):
        selected_indices = []
        for idx, q_obj in enumerate(mod["quiz"]):
            st.markdown(f"**Q{idx+1}. {q_obj['q']}**")
            ans_choice = st.radio(f"Question {idx+1}", q_obj["options"], key=f"q_{active_page}_{idx}", label_visibility="collapsed")
            selected_indices.append(q_obj["options"].index(ans_choice))

        if st.form_submit_button("Submit Quiz & Update Progress Bar", use_container_width=True):
            score = sum(1 for u_idx, q_obj in zip(selected_indices, mod["quiz"]) if u_idx == q_obj["ans"])
            save_module_progress(
                user["username"], active_page, mod["title"],
                task_done=True, quiz_score=score, max_score=len(mod["quiz"])
            )
            st.success(f"🎉 Quiz Submitted! You scored **{score} / {len(mod['quiz'])}**. Progress synced to Admin Dashboard.")
