#!/usr/bin/env bash
echo "==============================================================================="
echo "  SAKEC End-to-End AI, ML, DL, GenAI, RAG & Agentic AI Simulation Platform"
echo "  Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College"
echo "==============================================================================="
pip install -r requirements.txt
streamlit run app.py --server.port 8501
