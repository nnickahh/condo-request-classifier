@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo   Register Condo Telegram Bot 24/7 Autostart on Windows
echo ==============================================================
echo This will schedule the bot to start automatically when Windows boots / logs on.
echo.

set "SCRIPT_PATH=%~dp0start_bot_background.vbs"

schtasks /create /tn "CondoTelegramBot" /tr "wscript.exe \"%SCRIPT_PATH%\"" /sc onlogon /rl highest /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Windows Scheduled Task 'CondoTelegramBot' registered successfully!
    echo The Telegram bot will automatically launch in the background every time you log in.
    echo.
    echo To remove this task in the future, run:
    echo   schtasks /delete /tn "CondoTelegramBot" /f
) else (
    echo.
    echo [ERROR] Failed to register task. Please right-click this script and select 'Run as Administrator'.
)

echo.
pause
