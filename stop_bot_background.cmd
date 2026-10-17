@echo off
setlocal
cd /d "%~dp0"

echo Stopping any running Telegram bot processes...
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*condo_jev.telegram_bot*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('Terminated Bot PID: ' + $_.ProcessId) }"

echo Done.
pause
