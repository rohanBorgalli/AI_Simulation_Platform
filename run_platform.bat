@echo off
echo ===============================================================================
echo   SAKEC End-to-End AI, ML, DL, GenAI, RAG ^& Agentic AI Simulation Platform
echo   Developed by Dr. Rohan Borgalli, Shah ^& Anchor Kutchhi Engineering College
echo ===============================================================================
echo [1/2] Installing / Verifying Python Dependencies...
pip install -r requirements.txt
echo [2/2] Launching Streamlit Simulation Frontend on http://localhost:8501 ...
streamlit run app.py --server.port 8501
pause
