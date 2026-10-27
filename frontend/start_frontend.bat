@echo off
title Fruit Twin Frontend (Vite)
cd /d "%~dp0"
echo Starting Vite Frontend on http://localhost:3000 ...
npx vite --port 3000 --host
pause
