@echo off
title Fruit Twin Launcher
echo ========================================================
echo            FRUIT TWIN PLATFORM LAUNCHER
echo ========================================================
echo.
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Fruit Twin Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React + Vite Frontend on http://localhost:3000 ...
start "Fruit Twin Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npx vite --port 3000 --host"

timeout /t 2 /nobreak >nul

echo.
echo Opening Fruit Twin in default web browser...
start http://localhost:3000

echo.
echo ========================================================
echo  All systems active!
echo  - Frontend: http://localhost:3000
echo  - Backend:  http://127.0.0.1:8000
echo  Keep the terminal windows open while using the website.
echo ========================================================
pause
