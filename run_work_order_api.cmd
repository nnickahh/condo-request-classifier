@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo   Condominium Work Order System (CMMS) REST API Server
echo ==============================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m condo_jev.work_order_api %*
    exit /b %ERRORLEVEL%
)

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    uv run python -m condo_jev.work_order_api %*
    exit /b %ERRORLEVEL%
)

python -m condo_jev.work_order_api %*
