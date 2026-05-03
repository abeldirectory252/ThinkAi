@echo off
setlocal

set USERNAME=abeldirectory252
set REPO=ThinkAi

:: ── Read token from .env file (NEVER committed) ──
if not exist ".env" (
    echo ERROR: Create a .env file with one line: TOKEN=ghp_xxxYourTokenxxx
    pause
    exit /b 1
)
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="TOKEN" set TOKEN=%%b
)
if "%TOKEN%"=="" (
    echo ERROR: TOKEN not found in .env
    pause
    exit /b 1
)

cd /d %~dp0

echo ===== Git Auto Deploy =====

:: Fix credential helper error
git config --global --unset credential.helper 2>nul
git config --global credential.helper manager

:: Init if needed
if not exist ".git" git init

:: Stage all
git add .

:: Commit if changes exist
git diff --cached --quiet
if %errorlevel%==0 (
    echo No changes to commit.
) else (
    git commit -m "Auto commit %date% %time%"
)

git branch -M main

:: ── Nuke old history that contains the leaked token ──
:: This is needed ONE TIME to remove the secret from past commits
git remote remove origin 2>nul
git remote add origin https://%USERNAME%:%TOKEN%@github.com/%USERNAME%/%REPO%.git

git push -u origin main --force

echo ===== Done =====
pause
