@echo off
title Condo CMMS Operations Console
cd /d "%~dp0"
echo ===================================================
echo   Starting Condo CMMS Operations Console
echo ===================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause
