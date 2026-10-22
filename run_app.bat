@echo off
title VeritasLM - Enterprise LLM Observability and Hallucination Defense

echo Launching VeritasLM Neural Observability Platform (.venv CUDA runtime)...

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

.\.venv\Scripts\streamlit.exe run app\app.py
if errorlevel 1 (
    echo.
    echo Fallback: Attempting via python -m streamlit...
    .\.venv\Scripts\python.exe -m streamlit run app\app.py
)
pause
