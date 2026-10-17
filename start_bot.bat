@echo off
title Condo Telegram Bot Service
cd /d "%~dp0"
echo ===================================================
echo   Starting Condo Management Telegram Bot Service
echo ===================================================
call .venv\Scripts\activate.bat
python -m condo_jev.telegram_bot
pause
