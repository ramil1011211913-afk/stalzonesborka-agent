@echo off
title STALZONE Artifact Builder

cd /d "%~dp0"

echo ============================================
echo   STALZONE Artifact Builder (Python 3.12)
echo ============================================
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 not found!
    pause
    exit /b 1
)

echo [INFO] Killing old server on port 8123...
powershell -Command "Get-NetTCPConnection -LocalPort 8123 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 2 >nul

py -3.12 -c "import numba" 2>nul
if errorlevel 1 (
    echo [INFO] Installing Numba for Python 3.12...
    py -3.12 -m pip install numba -q
)

py -3.12 -c "import fastapi, uvicorn, requests, dotenv, scipy" 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies for Python 3.12...
    py -3.12 -m pip install -q -r requirements.txt
)

echo.
echo [1/3] Database:
py -3.12 -c "from app.database import init_db, get_artefact_stats_summary; init_db(); s=get_artefact_stats_summary(); print(f'  Artefacts: {s[\"artefacts\"]}, Lots: {s[\"active_lots\"]}, Containers: {s[\"containers\"]}')"

echo [2/3] Numba JIT:
py -3.12 -c "import app._numba_search; print('  Ready.')"

echo [3/3] Starting server...
py -3.12 -c "import os, app.main, app.builder, app._numba_search; print('WORKDIR:', os.getcwd()); print('MAIN:', app.main.__file__); print('BUILDER:', app.builder.__file__); print('NUMBA:', app._numba_search.__file__); print('DANGER LIMITS:', app._numba_search._DANGER_THRESH.tolist())"
echo.
echo   URL:  http://127.0.0.1:8123
echo   Ctrl+C to stop.
echo ============================================
echo.

py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8123
