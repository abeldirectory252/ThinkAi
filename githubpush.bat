@echo off
cd /d %~dp0

echo ===== Git Daily Push =====

:: Stage all changes
git add .

:: Commit if there are changes
git diff --cached --quiet
if %errorlevel%==0 (
    echo No changes to commit.
) else (
    git commit -m "%~1"
)

:: Push to origin main
git push origin main

echo ===== Done =====
