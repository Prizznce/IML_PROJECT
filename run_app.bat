@echo off
title VeritasLM - Enterprise LLM Observability and Hallucination Defense

echo Launching VeritasLM Neural Observability Platform (.venv CUDA runtime)...

.\.venv\Scripts\streamlit.exe run app\app.py
if errorlevel 1 (
    echo.
    echo Fallback: Attempting via python -m streamlit...
    .\.venv\Scripts\python.exe -m streamlit run app\app.py
)
pause
