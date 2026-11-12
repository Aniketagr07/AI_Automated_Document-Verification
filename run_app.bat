@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  AI Automated Document Verification System (SIH1709)
echo =====================================================================
echo Checking Python environment...

python --version >nul 2>&1
if errorlevel 1 (
    echo [-] Error: Python was not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

echo [+] Starting Streamlit application on http://localhost:8501 ...
python -m streamlit run app.py --server.port 8501

pause
