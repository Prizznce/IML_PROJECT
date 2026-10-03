@echo off
title Veritas AI - Enterprise LLM Observability & Hallucination Defense

echo Launching Veritas Neural Observability Platform (.venv CUDA runtime)...

.\.venv\Scripts\streamlit.exe run app\app.py
if errorlevel 1 (
    echo.
    echo Fallback: Attempting via python -m streamlit...
    .\.venv\Scripts\python.exe -m streamlit run app\app.py
)
pause
