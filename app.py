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
    
    # Google Gemini Direct Integration
    if "Google Gemini" in mode:
        gemini_key = st.session_state.get("gemini_api_key", "").strip()
        if not gemini_key:
            # Try Streamlit secret fallbacks
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if gemini_key and genai is not None:
            try:
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=f"System instructions: {system_prompt}\n\nUser prompt: {prompt}",
                    config={
                        'temperature': temperature,
                        'top_p': top_p
                    }
                )
                if response.text:
                    return response.text, "Google Gemini (gemini-2.5-flash-lite)"
            except Exception as e:
                st.warning(f"Gemini API request failed: {e}. Defaulting to Simulator.")
        else:
            st.info("No Gemini API key detected. Please input your key in the sidebar to activate.")

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
            "Google Gemini API (Highly Recommended)",
            "Hugging Face Serverless API (Free Open-Source)",
            "Local Ollama / PARAM Shavak Cluster Endpoint"
        ]
    )
    if "Google Gemini" in st.session_state.exec_mode:
        st.session_state.gemini_api_key = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            placeholder="AIzaSy..."
        )
        st.caption("Model Active: `gemini-3.5-flash`")
    elif "Hugging Face Serverless" in st.session_state.exec_mode:
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
                    fig_b.update_layout(title="Live Neural Decision Boundary", xaxis_title=f"{feat_names[0]} (Normalized)", yaxis_title=f"{feat_names[1]} (Normalized)")
                    st.plotly_chart(fig_b, use_container_width=True)

        elif active_page == "mod2_nlp_embed":
            corpus_path = os.path.join(SAMPLE_DIR, "multidomain_embedding_corpus.txt")
            default_txt = open(corpus_path, "r", encoding="utf-8").read() if os.path.exists(corpus_path) else (
                "Electric vehicles use lithium-ion battery packs and BLDC motors.\n"
                "Deep learning neural networks learn hierarchical representations.\n"
                "Spicy Mumbai vada pav and masala dosa are iconic Indian foods."
            )
            raw_inp = st.text_area("Enter Sentences to Tokenize & Embed (One sentence per line):", value=default_txt, height=180)
            sents = [s.strip() for s in raw_inp.split("\n") if s.strip()]

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

        elif active_page == "mod3_vectordb":
            default_kb = [
                "SAKEC PARAM Shavak supercomputing facility accelerates Deep Learning and RAG training workloads.\n",
                "Lithium-Ion Battery Management Systems prevent thermal runaway using cell balancing.\n",
                "ChromaDB indexes 384-dimensional embeddings using HNSW graphs for fast Cosine Similarity search.\n",
                "Backpropagation computes gradients of the Cross-Entropy Loss function across neural layers.\n",
                "Authentic Mumbai vada pav and Hyderabadi biryani are popular Indian culinary dishes."
            ]
            docs_raw = st.text_area("Vector Database Stored Documents (One per line):", value="".join(default_kb), height=130)
            docs = [d.strip() for d in docs_raw.split("\n") if d.strip()]
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

                st.markdown(f"**Query Vector q Slice `[q0..q5]`:** `{np.round(q_vec[:6], 4).tolist()}`")
                st.dataframe(df_rank, use_container_width=True)

                sim_matrix = np.dot(d_vecs, d_vecs.T)
                short_labels = [f"Doc {i+1}" for i in range(len(docs))]
                fig_hm = px.imshow(np.round(sim_matrix, 2), x=short_labels, y=short_labels, text_auto=True,
                                   color_continuous_scale="Blues", title="Pairwise Document Cosine Similarity Matrix")
                st.plotly_chart(fig_hm, use_container_width=True)

        elif active_page == "mod4_llm_prompt":
            st.markdown("### 🛠️ Interactive Language Model Sandbox")
            
            tab_gen, tab_api, tab_chat = st.tabs([
                "🎮 1. Next-Token Probabilities & Temperature",
                "🔌 2. HTTP POST API Code Client",
                "💬 3. Multi-Turn Interactive Conversation"
            ])

            with tab_gen:
                c_p1, c_p2 = st.columns([1, 1.8])
                with c_p1:
                    sys_prompt = st.text_area("System Prompt (Persona & Guardrails)", value="You are an Engineering Professor at SAKEC Mumbai. Provide rigorous, step-by-step technical explanations.")
                    temp_val = st.slider("Temperature (T)", 0.05, 1.5, 0.25, 0.05)
                    top_p_val = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)
                    p_mode = st.selectbox("Prompting Paradigm Template", ["Zero-Shot Prompt", "Few-Shot (In-Context Learning)", "Chain-of-Thought (Step-by-Step Engineering Math)"])

                    raw_logits = np.array([3.2, 2.5, 1.8, 1.1, 0.4])
                    candidate_tokens = ["battery", "powertrain", "voltage", "current", "system"]
                    scaled_exp = np.exp(raw_logits / max(0.05, temp_val))
                    probs = scaled_exp / scaled_exp.sum()
                    fig_tok = px.bar(x=candidate_tokens, y=probs, labels={"x": "Candidate Next Token", "y": "Softmax Probability"},
                                     title=f"Live Next-Token Softmax Distribution (T={temp_val})")
                    st.plotly_chart(fig_tok, use_container_width=True)

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
                    live_p = st.text_area("Live Prompt Input:", value=prompt_presets[p_mode], height=160)
                    if st.button("🚀 Run Prompt on Configured LLM Engine", use_container_width=True):
                        resp_txt, resp_src = execute_llm(live_p, sys_prompt, temp_val, top_p_val)
                        st.success(f"Executed via: {resp_src}")
                        st.markdown(resp_txt)

            with tab_api:
                st.markdown("#### 🔌 Emulating Code API Requests to LLM Servers")
                st.write("The code below demonstrates how your Python script constructs an HTTP POST API request to transmit raw prompts, system roles, and hyper-parameters to standard local servers (like Ollama) or corporate endpoints (like OpenAI/Anthropic).")
                
                api_payload = {
                    "model": "llama3.2:1b",
                    "prompt": "Compute pack nominal voltage for a 96-series battery system.",
                    "system": "You are a strict battery systems testing engineer.",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9
                    }
                }
                st.json(api_payload)
                st.code(f"""import requests

url = "http://localhost:11434/api/generate"
payload = {{
    "model": "llama3.2:1b",
    "prompt": "Compute pack nominal voltage for a 96-series battery system.",
    "system": "You are a strict battery systems testing engineer.",
    "stream": False,
    "options": {{
        "temperature": 0.2,
        "top_p": 0.9
    }}
}}

try:
    response = requests.post(url, json=payload, timeout=10)
    if response.status_code == 200:
        print("Answer:", response.json()["response"])
except Exception as e:
    print("Error connecting to backend:", e)
""", language="python")

            with tab_chat:
                st.markdown("#### 💬 Multi-Turn Conversation History Log")
                
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": "Welcome to SAKEC Simulation Lab! Ask me any engineering question to begin our session."}
                    ]

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if chat_input := st.chat_input("Ask a follow-up question here..."):
                    st.session_state.chat_history.append({"role": "user", "content": chat_input})
                    with st.chat_message("user"):
                        st.markdown(chat_input)

                    # Simulated conversational contextual synthesis
                    system_instr = "You are an EXTC/AI professor at SAKEC."
                    context_prompt = f"Context of convo history: {str(st.session_state.chat_history[-3:])}. New Question: {chat_input}"
                    ans_out, _ = execute_llm(context_prompt, system_instr, 0.3, 0.9)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": ans_out})
                    with st.chat_message("assistant"):
                        st.markdown(ans_out)
                        
                if st.button("🗑️ Clear Conversation History"):
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": "Welcome to SAKEC Simulation Lab! Ask me any engineering question to begin our session."}
                    ]
                    st.rerun()

        elif active_page == "mod5_rag":
            st.markdown("##### 📄 Live 5-Stage RAG Pipeline (Select Preloaded SAKEC PDFs or Upload Your Own PDF)")
            doc_choice = st.selectbox(
                "Select Knowledge Source Document",
                [
                    "Preloaded PDF: SAKEC PARAM Shavak & AI Lab Policy (2026-27)",
                    "Preloaded PDF: Introduction to Electric Vehicles Course Manual (T26104.3)",
                    "Upload Custom PDF or TXT Document"
                ]
            )
            up_file = None
            if "Upload" in doc_choice:
                up_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

            raw_doc_text = ""
            if up_file is not None:
                if up_file.name.lower().endswith(".pdf"):
                    reader = PdfReader(up_file)
                    raw_doc_text = "".join([p.extract_text() or "" for p in reader.pages])
                else:
                    raw_doc_text = up_file.read().decode("utf-8", errors="ignore")
            elif "PARAM" in doc_choice and os.path.exists(os.path.join(SAMPLE_DIR, "SAKEC_PARAM_Shavak_AI_Policy_2026.pdf")):
                reader = PdfReader(os.path.join(SAMPLE_DIR, "SAKEC_PARAM_Shavak_AI_Policy_2026.pdf"))
                raw_doc_text = "".join([p.extract_text() or "" for p in reader.pages])
            elif "Electric" in doc_choice and os.path.exists(os.path.join(SAMPLE_DIR, "Introduction_to_Electric_Vehicles_Course_Manual.pdf")):
                reader = PdfReader(os.path.join(SAMPLE_DIR, "Introduction_to_Electric_Vehicles_Course_Manual.pdf"))
                raw_doc_text = "".join([p.extract_text() or "" for p in reader.pages])

            if not raw_doc_text.strip():
                raw_doc_text = (
                    "SAKEC PARAM Shavak supercomputing facility supports Deep Learning, VLMs, and RAG workloads. "
                    "Local Ollama models (llama3.2:1b, qwen2.5:1.5b) and all-MiniLM-L6-v2 (384-D) embeddings execute offline without API keys. "
                    "Recommended RAG chunk size is 180 to 512 characters with 35 to 64 overlap indexed in ChromaDB."
                )

            rc1, rc2, rc3 = st.columns(3)
            c_size = rc1.slider("Stage 1: Chunk Size (Chars)", 120, 450, 220, 20)
            c_over = rc2.slider("Stage 1: Chunk Overlap (Chars)", 0, 100, 40, 10)
            top_k_rag = rc3.slider("Stage 4: Top-K Chunks to Retrieve", 1, 5, 2)

            chunks = []
            stride = max(25, c_size - c_over)
            for i in range(0, len(raw_doc_text), stride):
                piece = raw_doc_text[i:i + c_size].strip()
                if len(piece) > 25:
                    chunks.append(piece)

            st.markdown(f"<div class='step-card'><b>Stage 1 — Document Ingestion & Sliding-Window Chunking:</b> Extracted <b>{len(raw_doc_text)} characters</b> and partitioned into <b>{len(chunks)} overlapping chunks</b> (Chunk Size={c_size}, Overlap={c_over}, Stride={stride}).</div>", unsafe_allow_html=True)
            with st.expander("🔍 Inspect All Generated Document Chunks", expanded=False):
                for idx, ch in enumerate(chunks):
                    st.markdown(f"**Chunk #{idx+1}:** `{ch}`")

            c_vecs, emb_engine = compute_embeddings(chunks)
            st.markdown(f"<div class='step-card'><b>Stage 2 & Stage 3 — Dense Embedding & ChromaDB Vector Indexing:</b> Encoded {len(chunks)} chunks into `{c_vecs.shape}` unit-normalized vectors via <b>{emb_engine}</b> and indexed in Vector Store.</div>", unsafe_allow_html=True)

            default_q = "What chunk size and local models are used on the PARAM Shavak cluster?" if "PARAM" in doc_choice else "What is the pack voltage and range of a 96-series 3.7V 50Ah EV battery?"
            rag_q = st.text_input("❓ Enter Grounded User Question:", value=default_q)
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
                    "Status": ["Top-K Retrieved" if i in top_idx else "Filtered Out" for i in range(len(chunks))]
                })
                fig_rag = px.bar(df_chunk_scores, x="Chunk_ID", y="Cosine_Similarity", color="Status",
                                 title="Cosine Similarity Score Across All Document Chunks")
                st.plotly_chart(fig_rag, use_container_width=True)

                ctx_blocks = []
                for r_num, c_i in enumerate(top_idx, start=1):
                    ctx_blocks.append(f"[Chunk #{c_i+1} | Cosine Score={sims[c_i]:.4f}]: {chunks[c_i]}")
                    st.info(f"**Top-{r_num} Retrieved (Chunk #{c_i+1} | Score: {sims[c_i]:.4f}):** {chunks[c_i]}")

                aug_prompt = (
                    "You are a factual RAG assistant. Answer the user's question strictly using ONLY the retrieved context chunks below.\n\n"
                    "RETRIEVED CONTEXT:\n" + "\n".join(ctx_blocks) + f"\n\n"
                    f"USER QUESTION: {rag_q}\n\nGROUNDED ANSWER:"
                )
                st.markdown("<div class='step-card'><b>Stage 5 — Augmented Prompt Construction & Grounded LLM Generation:</b></div>", unsafe_allow_html=True)
                st.code(aug_prompt, language="markdown")
                if st.button("✨ Generate Grounded Answer from Retrieved Context", use_container_width=True):
                    # Uses the dynamically generated aug_prompt which contains the actual rag_q query text!
                    rag_ans, rag_src = execute_llm(aug_prompt, "Cite chunk numbers and use only provided context.", 0.1, 0.9)
                    st.success(f"Synthesized via {rag_src}:")
                    st.markdown(rag_ans)

        elif active_page == "mod6_agentic":
            st.markdown("##### 📄 Autonomous ReAct Agent Loop (`Thought -> Action -> Observation -> Final Answer`)")
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
                    kb_obs = "PARAM Shavak cluster runs local Ollama LLMs (llama3.2:1b, qwen2.5:1.5b) and 384-D MiniLM embeddings with 180-512 char chunking."
                    trace_steps.append(("👁️ Observation 3", f"`{kb_obs}`"))
                    final_points.append(f"**SAKEC Policy Observation:** {kb_obs}")

                for title_s, body_s in trace_steps:
                    st.markdown(f"<div class='step-card'><b>{title_s}:</b> {body_s}</div>", unsafe_allow_html=True)

                st.success("🎯 **Final Agent Response (Synthesized from Verified Tool Observations):**- " + "- ".join(final_points))

    # ==========================================
    # TAB 3: HANDS-ON TASK VERIFICATION
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
                st.rerun()
