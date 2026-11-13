<#
.SYNOPSIS
    Convenience launcher for the AI Automated Document Verification System.
.DESCRIPTION
    Starts the Streamlit application server, checks service health, or executes the unit test suite.
.PARAMETER Headless
    Runs the server in headless background mode without popping a browser window.
.PARAMETER Port
    Specifies the port to bind (default: 8501).
.PARAMETER RunTests
    Executes the 33 unit tests without starting the server.
.PARAMETER CheckHealth
    Executes an HTTP health check against http://localhost:<Port>/_stcore/health.
.EXAMPLE
    .\run_app.ps1
    .\run_app.ps1 -Headless
    .\run_app.ps1 -RunTests
    .\run_app.ps1 -CheckHealth
#>

param(
    [switch]$Headless,
    [int]$Port = 8501,
    [switch]$RunTests,
    [switch]$CheckHealth
)

Set-Location -Path $PSScriptRoot

if ($CheckHealth) {
    Write-Host "[*] Checking health at http://localhost:$Port/_stcore/health ..." -ForegroundColor Cyan
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 5
        if ($res.StatusCode -eq 200) {
            Write-Host "[+] Service is HEALTHY on port $Port (Status: $($res.StatusCode), Response: $($res.Content))" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "[-] Service returned status code $($res.StatusCode)" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "[-] Service is NOT reachable on port $Port : $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

if ($RunTests) {
    Write-Host "[*] Executing unit test suite..." -ForegroundColor Cyan
    python -m unittest discover -s . -p "test_*.py" -v
    exit $LASTEXITCODE
}

Write-Host "=====================================================================" -ForegroundColor Blue
Write-Host " AI Automated Document Verification System (SIH1709)                " -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Blue

# Check if port is already active
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    Write-Host "[!] Port $Port is already in use by PID $($conn.OwningProcess[0])." -ForegroundColor Yellow
    Write-Host "[*] Checking if it is responding to health checks..." -ForegroundColor Cyan
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 3
        if ($res.StatusCode -eq 200) {
            Write-Host "[+] Streamlit server is already active and healthy on port $Port!" -ForegroundColor Green
            Write-Host "    Open: http://localhost:$Port" -ForegroundColor Cyan
            exit 0
        }
    } catch {}
}

# Detect Python executable with Streamlit
$PythonExe = "python"
$hasStreamlit = & $PythonExe -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    $fallbackPy = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (Test-Path $fallbackPy) {
        $PythonExe = $fallbackPy
    }
}

if ($Headless) {
    Write-Host "[+] Launching Streamlit in background on port $Port using $PythonExe..." -ForegroundColor Green
    Start-Process -FilePath $PythonExe -ArgumentList "-m streamlit run app.py --server.headless true --server.port $Port" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    & $PSCommandPath -CheckHealth -Port $Port
} else {
    Write-Host "[+] Launching Streamlit in interactive mode on port $Port using $PythonExe..." -ForegroundColor Green
    & $PythonExe -m streamlit run app.py --server.port $Port
}
