Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Run the 24/7 watchdog script completely hidden (0 = hide window, false = do not wait)
WshShell.Run """" & strPath & "\run_telegram_bot_24_7.cmd""", 0, False
