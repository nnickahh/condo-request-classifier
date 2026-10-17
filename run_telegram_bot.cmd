@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m condo_jev.telegram_bot %*
    exit /b %ERRORLEVEL%
)

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    uv run python -m condo_jev.telegram_bot %*
    exit /b %ERRORLEVEL%
)

python -m condo_jev.telegram_bot %*
