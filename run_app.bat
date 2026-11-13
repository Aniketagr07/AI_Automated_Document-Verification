@echo off
setlocal
cd /d "%~dp0"

echo =====================================================================
echo  AI Automated Document Verification System (SIH1709)
echo =====================================================================
echo Checking Python environment...

set "PY_EXE=python"
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else (
        echo [-] Error: Python with Streamlit was not found. Please install requirements.
        pause
        exit /b 1
    )
)

echo [+] Starting Streamlit application on http://localhost:8501 ...
"%PY_EXE%" -m streamlit run app.py --server.port 8501

pause
