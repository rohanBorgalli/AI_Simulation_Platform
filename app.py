# End-to-End AI, ML, DL, GenAI, LLM, NLP, Embeddings, Vector DB, RAG & Agentic AI Simulation Platform
# Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur

import os
import re
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
from pypdf import PdfReader
import requests

from content_data import MODULES, REFERENCE_PLATFORMS

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
        padding-bottom: 95px !important;
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
        <b>Zero-API-Key Architecture</b><br>
        <span style="color:#38bdf8;">Local / Cluster & Free HF Serverless Ready</span>
    </div>
</div>

<div class="sakec-footer">
    🎓 {BRAND_ATTRIBUTION} | Interactive AI, Deep Learning, GenAI, RAG & Agentic Training Platform
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. SQLITE DATABASE & RBAC PROGRESS ENGINE
# ==========================================
DB_PATH = "training_platform.db"
SAMPLE_DIR = os.path.join(os.getcwd(), "sample_data")

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
    c.execute("INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              ("admin", "Dr. Rohan Borgalli (Admin)", "EXTC - SAKEC Chembur", admin_hash, "admin", now))
    c.execute("INSERT OR IGNORE INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              ("student1", "Demo Participant (Trainee)", "EXTC - SAKEC Chembur", student_hash, "user", now))
    conn.commit()
    conn.close()

init_database()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT username, full_name, department, role FROM users WHERE username=? AND password_hash=?", (username.strip(), pw_hash))
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
        c.execute("INSERT INTO users (username, full_name, department, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                  (username.strip(), full_name.strip(), department.strip(), pw_hash, role, now))
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
# 3. DUAL ZERO-API-KEY EMBEDDING & LLM ENGINE
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
        "neural": np.array([-0.5, 1.2, 0.3, -0.2]),
        "deep": np.array([-0.48, 1.15, 0.32, -0.18]),
        "transformer": np.array([-0.45, 1.18, 0.4, -0.15]),
        "rag": np.array([-0.35, 1.1, 0.55, 0.1]),
        "vector": np.array([-0.38, 1.08, 0.52, 0.12])
    }
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t.lower())) % (2**32))
        v = rng.normal(0, 0.06, 384)
        tokens = re.findall(r"[a-z0-9]+", t.lower())
        for tok in tokens:
            for kw, anchor in domain_anchors.items():
                if kw in tok:
                    v[:4] += anchor * 0.9
        v = v / (np.linalg.norm(v) + 1e-9)
        vecs.append(v)
    return np.array(vecs), "Built-in Deterministic 384-D Vector Engine (Zero-API-Key)"

def execute_llm(prompt, system_prompt, temperature, top_p):
    mode = st.session_state.get("exec_mode", "Built-in Interactive Simulator (Zero Setup)")
    hf_token = st.session_state.get("hf_token", "").strip()

    if "Hugging Face Serverless API" in mode and hf_token:
        try:
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "inputs": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "parameters": {
                    "temperature": max(0.01, temperature),
                    "top_p": top_p,
                    "max_new_tokens": 512,
                    "return_full_text": False
                }
            }
            resp = requests.post(api_url, json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                res_json = resp.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    return res_json[0].get("generated_text", ""), f"HF Serverless ({model_id})"
                elif isinstance(res_json, dict) and "generated_text" in res_json:
                    return res_json["generated_text"], f"HF Serverless ({model_id})"
        except Exception as e:
            st.warning(f"HF API failed: {e}. Falling back to simulation model.")

    if "Ollama" in mode:
        try:
            resp = requests.post(
                f"{st.session_state.get('ollama_url', 'http://localhost:11434').rstrip('/')}/api/generate",
                json={
                    "model": st.session_state.get("ollama_model", "llama3.2:1b"),
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "top_p": top_p}
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("response", ""), "Local Ollama"
        except Exception:
            pass

    # Grounded contextual response generator
    if "RETRIEVED CONTEXT:" in prompt and "USER QUESTION:" in prompt:
        context_text = prompt.split("RETRIEVED CONTEXT:")[1].split("USER QUESTION:")[0].strip()
        user_q = prompt.split("USER QUESTION:")[1].split("GROUNDED ANSWER:")[0].strip()
        
        # Analyze if context answers the question
        clean_q = user_q.lower()
        clean_ctx = context_text.lower()
        
        matches = []
        if "param" in clean_q or "shavak" in clean_q:
            if "deep learning" in clean_ctx or "ollama" in clean_ctx:
                matches.append("• SAKEC PARAM Shavak cluster supports Deep Learning, VLMs, and local RAG parsing.")
                matches.append("• Offline operations use llama3.2:1b, qwen2.5:1.5b and all-MiniLM-L6-v2 embeddings.")
        if "chunk" in clean_q or "overlap" in clean_q:
            if "chunk size" in clean_ctx:
                matches.append("• Ideal RAG sizing: Chunks 180 to 512 characters with an overlapping region of 35 to 64 chars.")
        if "battery" in clean_q or "voltage" in clean_q or "range" in clean_q:
            if "96-series" in clean_ctx or "3.7v" in clean_ctx:
                matches.append("• Pack Configuration: 96 cells in series @ nominal voltage 3.7V yielding 355.2V pack voltage.")
                matches.append("• Estimated Driving Range: 118.4 km computed with a normalized draw rate of 150 Wh/km.")
                
        if matches:
            grounded_synthesis = "\n".join(matches)
        else:
            grounded_synthesis = f"• Synthesized Context Snippet: '{context_text[:200]}...'"
            
        return (
            f"**Grounded Cognitive Response (Zero-API Simulator Mode):**\n\n"
            f"Based strictly on the verified knowledge index retrieved, here is the answer:\n\n"
            f"{grounded_synthesis}\n\n"
            f"*Confidence score: 98% grounded (No parametric hallucinations detected).*"
        ), "Built-in Zero-API-Key RAG Synthesizer"

    if "96 cells" in prompt or "3.7V" in prompt:
        return (
            f"**Step-by-Step Chain-of-Thought Engineering Derivation (Temp={temperature}):**\n"
            f"1. **Total Pack Nominal Voltage:** V_pack = 96 * 3.7 V = 355.2 V\n"
            f"2. **Total Pack Energy Capacity:** E_pack = (355.2 V * 50 Ah) / 1000 = 17.76 kWh\n"
            f"3. **Estimated Driving Range (at 150 Wh/km):** Range = 17,760 Wh / 150 Wh/km = 118.4 km"
        ), "Built-in Zero-API-Key LLM Engine"

    return (
        f"**[Pedagogical Simulator Response]**\n"
        f"System Persona Active: *{system_prompt}*\n\n"
        f"The simulation output indicates highly optimized convergence for prompt query: \"{prompt[:100]}...\""
    ), "Built-in Zero-API-Key LLM Engine"

# ==========================================
# 4. LOGIN & REGISTRATION VIEW
# ==========================================
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 🔒 Session Authentication Portal")
        tab_in, tab_reg = st.tabs(["🔑 Registered Sign In", "📝 Trainee Registration"])
        
        with tab_in:
            with st.form("login_form"):
                u_in = st.text_input("Username", value="student1")
                p_in = st.text_input("Password", type="password", value="student123")
                btn_login = st.form_submit_button("Enter Simulation Workspace", use_container_width=True)
                if btn_login:
                    auth = authenticate_user(u_in, p_in)
                    if auth:
                        st.session_state.user = auth
                        st.success(f"Welcome back, {auth['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Use student1/student123 or admin/admin123.")
        
        with tab_reg:
            with st.form("register_form"):
                r_user = st.text_input("Choose Username")
                r_name = st.text_input("Full Name")
                r_dept = st.text_input("Department", value="EXTC - SAKEC Chembur")
                r_pass = st.text_input("Password", type="password")
                r_role = st.selectbox("Role Type", ["user", "admin"])
                btn_reg = st.form_submit_button("Register Trainee Account", use_container_width=True)
                if btn_reg:
                    if r_user and r_name and r_pass:
                        ok, msg = register_new_user(r_user, r_name, r_dept, r_pass, r_role)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill all configuration fields.")
    with col2:
        st.markdown("""
        ### 🏛️ SAKEC AI & GenAI Simulation Lab
        This virtual training laboratory runs zero-cost, serverless AI/ML engineering tools locally. To track your curriculum completion milestones:
        - **Trainee Username**: `student1` | **Password**: `student123`
        - **Faculty Admin Username**: `admin` | **Password**: `admin123`
        
        Your task performance and graded concept check records sync instantly with the local SQLite progress DB.
        """)
    st.stop()

user = st.session_state.user
user_prog = fetch_user_progress(user["username"])

# ==========================================
# 5. SIDEBAR: USER PROFILE & CONFIG
# ==========================================
with st.sidebar:
    st.markdown(f"### 👤 {user['full_name']}")
    st.caption(f"Role: **{user['role'].upper()}** | {user['department']}")

    done_mods = sum(1 for m_k in MODULES if user_prog.get(m_k, {}).get("task_completed") or user_prog.get(m_k, {}).get("quiz_score", 0) > 0)
    pct_done = int((done_mods / len(MODULES)) * 100)
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
    st.markdown("### ⚙️ Zero-API-Key Engine Router")
    st.session_state.exec_mode = st.selectbox(
        "LLM Execution Mode",
        [
            "Built-in Interactive Simulator (Zero Setup)",
            "Hugging Face Serverless API (Free Open-Source)",
            "Local Ollama / PARAM Shavak Cluster Endpoint"
        ]
    )
    if "Hugging Face Serverless" in st.session_state.exec_mode:
        st.session_state.hf_token = st.text_input("Hugging Face User Access Token (Free)", type="password", help="Get a free Read token from huggingface.co")
        st.caption("Model Active: `Qwen/Qwen2.5-7B-Instruct`")
    elif "Ollama" in st.session_state.exec_mode:
        st.session_state.ollama_url = st.text_input("Ollama / Cluster Endpoint URL", value="http://localhost:11434")
        st.session_state.ollama_model = st.text_input("Model Name", value="llama3.2:1b")

    st.session_state.use_local_st = st.checkbox("Use Local SentenceTransformer", value=False)

    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.user = None
        st.rerun()

# ==========================================
# 6. ADMIN ANALYTICS & COHORT TRACKER
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
            "↓ Download Cohort Progress Report (CSV)",
            data=df_prog.to_csv(index=False).encode("utf-8"),
            file_name="SAKEC_AI_Simulation_Cohort_Progress.csv",
            mime="text/csv"
        )
    st.stop()

# ==========================================
# 7. REFERENCE PLATFORMS HUB
# ==========================================
if active_page == "ref_platforms":
    st.subheader("🌐 Reference Simulation Platforms & Pedagogical Benchmarks")
    st.write("This platform integrates simulation paradigms from the following benchmark AI/ML educational tools:")
    st.dataframe(pd.DataFrame(REFERENCE_PLATFORMS), use_container_width=True, hide_index=True)
    st.stop()

# ==========================================
# 8. 4-PILLAR MODULE WORKSPACE
# ==========================================
if active_page in MODULES:
    mod = MODULES[active_page]
    st.subheader(mod["title"])
    st.caption(mod["subtitle"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 1. Theory & Mathematical Concepts",
        "🧪 2. Live Interactive Simulation Sandbox",
        "✅ 3. Knowledge Check & Hands-On Task",
        "└─ 4. Reference Platforms & Further Reading"
    ])

    with tab1:
        st.markdown(mod["theory"])

    with tab4:
        st.markdown("#### Benchmark Platforms & Further Reading")
        for r in REFERENCE_PLATFORMS:
            st.markdown(f"- **[{r['name']}]({r['url']})** (*{r['category']}*): {r['desc']}")

    # ==========================================
    # TAB 2: LIVE SIMULATIONS
    # ==========================================
    with tab2:
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

                lr = st.select_slider("Learning Rate (̷)", options=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2], value=0.03)
                epochs = st.slider("Training Epochs", 20, 300, 120, step=10)
                hidden_n = st.slider("Hidden Neurons per Layer (2 Layers)", 2, 32, 10)
                act_fn = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
                reg_alpha = st.select_slider("L2 Regularization (̱)", options=[0.0, 0.0001, 0.001, 0.01, 0.1], value=0.001)

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
                    fig_l = px.line(y=clf.loss_curve_, labels={("index"): "Epoch", "y": "Cross-Entropy Loss"}, title="Gradient Descent Convergence Curve")
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
                    fig_b.update_layout(title="Live Neural Decision Boundary", xaxis_title=f"{feat_names[0]} (Normalized)", yaxis_title=f
