@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py %*
    exit /b %ERRORLEVEL%
)

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    uv run streamlit run app.py %*
    exit /b %ERRORLEVEL%
)

python -m streamlit run app.py %*
