@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
py -3.12 tools\local_qa.py %*
echo.
pause
