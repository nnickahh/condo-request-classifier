@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==============================================================
echo   Condo Management Agent — Telegram Bot 24/7 Watchdog Runner
echo ==============================================================
echo Bot will auto-restart if interrupted or if network drops.
echo Press Ctrl+C in this window to stop the 24/7 service.
echo.

set "PYTHONUNBUFFERED=1"
if not exist "logs" mkdir logs

:LOOP
echo [%date% %time%] Launching Telegram Bot service... >> logs\telegram_bot.log
echo [%date% %time%] Launching Telegram Bot service...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m condo_jev.telegram_bot >> logs\telegram_bot.log 2>&1
) else (
    python -m condo_jev.telegram_bot >> logs\telegram_bot.log 2>&1
)

echo [%date% %time%] Process stopped (Exit code: %ERRORLEVEL%). Restarting in 5 seconds...
echo [%date% %time%] Process stopped (Exit code: %ERRORLEVEL%). Restarting in 5 seconds... >> logs\telegram_bot.log
timeout /t 5 /nobreak >nul
goto LOOP
