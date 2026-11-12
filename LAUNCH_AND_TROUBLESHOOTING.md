# AI Automated Document Verification System — Launch & Troubleshooting Guide

**System Name**: Automated Official Document Verification and Consistency Review System  
**Problem Domain**: SIH 2024 Problem ID SIH1709  
**Platform**: Streamlit Multi-Page Web Application  
**Primary Endpoint**: [http://localhost:8501](http://localhost:8501)  
**Health Check Endpoint**: [http://localhost:8501/_stcore/health](http://localhost:8501/_stcore/health)  

---

## 1. Diagnostic Summary & Root Cause Analysis

### The Issue
When attempting to access the application via a web browser at `http://localhost:8501`, the browser displayed the error:
> **"This site can’t be reached — localhost refused to connect."** (`ERR_CONNECTION_REFUSED`)

### Root Cause
1. **Inactive Process**: No active Streamlit server process was bound to TCP port `8501` on loopback interfaces (`127.0.0.1` / `::1`).
2. **TCP Protocol Rejection**: In Windows networking, when an HTTP client or browser initiates a TCP SYN handshake toward a port where no process has established a listening socket, the operating system TCP stack immediately responds with an active TCP RST (Reset) packet. This triggers `ERR_CONNECTION_REFUSED` in Chrome/Edge and `Connection refused` in curl.
3. **Remediation**: The Streamlit application server must be launched from the project root (`C:\Users\shiva\AI_Automated_Document-Verification`) using Python so that it binds to port 8501 and begins accepting incoming HTTP connections.

---

## 2. Environment & Prerequisites

### Hardware & Operating System
- **Operating System**: Windows 10 / Windows 11 (64-bit) or Linux/macOS
- **Memory**: Minimum 4 GB RAM (8 GB recommended for ML classification models)
- **Disk Space**: ~500 MB for Python environment and dependencies

### Software Requirements
- **Python**: Version `3.10` or higher (`3.10.x` recommended and validated)
- **Pip**: Latest version compatible with Python 3.10 (`python -m pip install --upgrade pip`)

### Installing Dependencies
From the project root directory, install all required packages:
```powershell
cd C:\Users\shiva\AI_Automated_Document-Verification
python -m pip install -r requirements.txt
```

### Key Libraries in `requirements.txt`
| Package | Version | Purpose |
|---|---|---|
| `streamlit` | `>=1.35.0` | High-performance reactive web application framework |
| `pymupdf` (`fitz`) | `>=1.24.0` | High-speed native digital PDF parsing and text stream extraction |
| `scikit-learn` | `>=1.4.0` | ML document classification and anomaly detection models |
| `pandas` / `numpy` | `>=2.0.0` / `>=1.26.0` | Tabular data manipulation and matrix operations |
| `pillow` | `>=10.0.0` | Image handling and preprocessing |
| `pytesseract` | `>=0.3.10` | Python wrapper for optional optical character recognition |
| `python-dateutil` | `>=2.8.0` | Robust multi-format date extraction and temporal inconsistency evaluation |
| `joblib` | `>=1.3.0` | Serialized model loading |

### OCR & Text Extraction Architecture (PyMuPDF vs Tesseract)
- **Primary Engine (PyMuPDF / fitz)**: The pipeline uses PyMuPDF for native digital PDF parsing. This extracts text streams, metadata, and structural layout directly from vector PDFs without requiring external binaries or cloud API keys. It runs 100% locally with zero latency.
- **Optional Fallback (Tesseract OCR)**: When evaluating scanned or photographic documents where no digital text stream exists, the system automatically checks for the `tesseract` binary in standard paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
  - If Tesseract is **not installed**, the system outputs an informational log:  
    `Tesseract OCR binary not found in standard paths. Fallback to digital text.`  
    The system continues operating normally using digital extraction without crashing or raising unhandled exceptions.
  - To enable scanned physical image OCR, install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) to the default path `C:\Program Files\Tesseract-OCR`.

---

## 3. Step-by-Step Local Launch Instructions

Always execute commands from the project root: `C:\Users\shiva\AI_Automated_Document-Verification`.

### Method A: Interactive Mode (Standard Foreground Launch)
Use this command to launch the app and automatically open your default browser:
```powershell
# In Windows PowerShell:
cd C:\Users\shiva\AI_Automated_Document-Verification
python -m streamlit run app.py
```
Or in Command Prompt (`cmd.exe`):
```cmd
cd C:\Users\shiva\AI_Automated_Document-Verification
python -m streamlit run app.py
```

### Method B: Headless Background Mode
Use this command to run Streamlit silently in the background (ideal for development servers or services):
```powershell
cd C:\Users\shiva\AI_Automated_Document-Verification
Start-Process -FilePath "python" -ArgumentList "-m streamlit run app.py --server.headless true --server.port 8501" -WindowStyle Hidden
```

### Method C: One-Click Convenience Launchers
The project includes automated launcher scripts:
1. **PowerShell Launcher** (`run_app.ps1`):
   ```powershell
   # Run app interactively:
   .\run_app.ps1

   # Run in background (headless):
   .\run_app.ps1 -Headless

   # Run unit test suite:
   .\run_app.ps1 -RunTests

   # Check server health:
   .\run_app.ps1 -CheckHealth
   ```
2. **Command Prompt Batch Launcher** (`run_app.bat`):
   Double-click `run_app.bat` in File Explorer or run:
   ```cmd
   run_app.bat
   ```

---

## 4. Service Verification & Health Checks

Once the Streamlit server is started, verify its health using any of the following methods:

### Method 1: PowerShell HTTP Request
```powershell
# 1. Test primary web interface (returns HTTP 200 OK)
$response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing
Write-Host "Status Code: $($response.StatusCode) ($($response.StatusDescription))"

# 2. Test Streamlit internal health endpoint (returns "ok" and HTTP 200)
$health = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing
Write-Host "Health Status: $($health.StatusCode), Body: $($health.Content)"
```
**Expected Output:**
```
Status Code: 200 (OK)
Health Status: 200, Body: ok
```

### Method 2: Curl
```bash
# Check headers
curl -I http://localhost:8501

# Check health endpoint
curl http://localhost:8501/_stcore/health
```
**Expected Output:**
```
HTTP/1.1 200 OK
...
ok
```

### Method 3: Active Port Inspection
Verify that port `8501` is actively listening on the loopback and dual-stack interfaces:
```powershell
Get-NetTCPConnection -LocalPort 8501 -State Listen
```
**Expected Output:**
```
LocalAddress    LocalPort  RemoteAddress  RemotePort  State   OwningProcess
::              8501       ::             0           Listen  <PID>
```

### Method 4: Browser Access
Open your web browser and navigate to:
- **Primary**: [http://localhost:8501](http://localhost:8501)
- **Alternative IPv4**: [http://127.0.0.1:8501](http://127.0.0.1:8501)
- **Alternative IPv6**: [http://[::1]:8501](http://[::1]:8501)

---

## 5. Unit Test Suite Execution

The project includes an automated test suite comprising 33 unit tests covering authentication, rule engines, temporal inconsistency checks, Aadhaar/PAN checksum validation (Verhoeff algorithm), report generation, and security role isolation.

### Test Execution Command
Run the test runner from the project root:
```powershell
cd C:\Users\shiva\AI_Automated_Document-Verification
python -m unittest discover -s . -p "test_*.py" -v
```

### Test Suite Structure
- **`test_system.py`** (25 Tests):
  - `TestAuthenticationAndSecurity`: User registration, PBKDF2-SHA256 password hashing, RBAC enforcement, prevention of privilege escalation, applicant data isolation.
  - `TestDecisionEngineAndWorkflow`: Automated finalization of clean documents, escalation of contradictory dates, missing mandatory field escalation, illegible scan detection.
  - `TestKnowledgeRepresentation`: Multi-format date parsing, schema validation, schema registry integrity.
  - `TestMachineLearningAndEvaluation`: TF-IDF document classification, model prediction tests on educational certificates and identity proofs.
  - `TestProductionRuleEngine`: Rules R001–R015 (expired documents, inverted expiry dates, future issue dates).
  - `TestReportsAndExports`: Audit report generation with unique reference IDs.
  - `TestSIHEnhancedForensicsAndChecksums`: Verhoeff algorithm Aadhaar checksum verification, PAN structure and surname matching, dual-axis uncertainty matrix.
- **`test_extended_audit.py`** (8 Tests):
  - `test_admin_bootstrap_and_authentication`: Verifies CLI bootstrap administrative provisioning.
  - `test_applicant_signup_role_security`: Verifies public registration cannot grant ADMIN role.
  - `test_canonical_statuses`: Verifies cross-module consistency of status constants (`APPROVED`, `INCONSISTENCIES_DETECTED`, `NEEDS_MANUAL_REVIEW`, `UNABLE_TO_VERIFY`).
  - `test_decision_engine_unreadable_scan`: Verifies unreadable document fallback.
  - `test_password_reset_token_workflow`: Verifies single-use, cryptographic password reset tokens.
  - `test_report_generation_all_canonical_statuses`: Verifies PDF report rendering for all canonical statuses.
  - `test_rule_engine_temporal_inversion`: Verifies temporal anomaly detection (expiry before issue date).
  - `test_user_data_isolation`: Confirms strict cross-user data isolation.

### Expected Result
```
Ran 33 tests in ~1.3s
OK
```

---

## 6. Demo Accounts & Credentials

The system implements Role-Based Access Control (RBAC) with distinct portals for Applicants and Administrative Verification Officers:

### 1. Citizen / Applicant Role
- **Login Email**: `applicant@college.edu`
- **Password**: `User@DocVerify2024!`
- **Role**: `USER`
- **Capabilities**:
  - Secure document upload (PDF, PNG, JPG).
  - Personal document dossier tracking.
  - Inspection of extraction results, confidence scores, and rule validation status.
  - Download of official certified PDF verification reports.
  - Strict data isolation: Cannot access documents or dossiers of other applicants.

### 2. Lead Verification Officer / Admin Role
- **Login Email**: `admin@docverify.gov.in`
- **Password**: `Admin@DocVerify2024!`
- **Role**: `ADMIN`
- **Capabilities**:
  - Administrative review queue access for escalated dossiers.
  - Full inspection of documents flagged with `NEEDS_MANUAL_REVIEW` or `INCONSISTENCIES_DETECTED`.
  - Manual field correction and final override (`APPROVED` / `REJECTED`).
  - Immutable audit history inspection with timestamped actor logging.
  - System-wide benchmark and machine learning evaluation reports.

### Administrative Account Provisioning (`bootstrap_admin.py`)
To provision a new administrative account or reset an existing administrator password securely from the command line:

```powershell
cd C:\Users\shiva\AI_Automated_Document-Verification

# Interactive Prompt:
python bootstrap_admin.py

# Automated / Non-interactive command:
python bootstrap_admin.py --email admin@docverify.gov.in --password "Admin@DocVerify2024!" --name "Lead Verification Officer"
```
*Note: Password must be at least 8 characters long and contain uppercase letters, lowercase letters, numbers, and special characters.*

---

## 7. Troubleshooting Matrix

| Issue / Symptom | Probable Cause | Diagnostic Command | Remediation Step |
|---|---|---|---|
| **"localhost refused to connect"** (`ERR_CONNECTION_REFUSED`) | Streamlit server process is not running. | `Get-NetTCPConnection -LocalPort 8501` | Launch the server: `python -m streamlit run app.py --server.headless true --server.port 8501`. |
| **Port 8501 Already in Use** (`Address already in use` error) | A stale Streamlit or Python process is still bound to port 8501. | `Get-NetTCPConnection -LocalPort 8501` | Terminate the occupying process: `Stop-Process -Id <PID> -Force` or run on an alternate port: `python -m streamlit run app.py --server.port 8502`. |
| **`ModuleNotFoundError: No module named 'database'`** | Python command was executed from the wrong working directory. | `Get-Location` | Ensure you `cd C:\Users\shiva\AI_Automated_Document-Verification` before running any scripts. |
| **"Tesseract OCR binary not found in standard paths"** | Tesseract OCR is not installed or not in PATH. | `Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"` | Normal behavior. Digital PDFs are automatically processed via PyMuPDF. If OCR on photos/scans is needed, install Tesseract for Windows. |
| **Login Fails with "Invalid email or password"** | Database not initialized or password entered incorrectly. | `python bootstrap_admin.py` | Run `python bootstrap_admin.py --email admin@docverify.gov.in --password "Admin@DocVerify2024!"` to re-provision the admin account. |
| **Database File Locked (`sqlite3.OperationalError: database is locked`)** | Another process is holding an exclusive transaction on `verification_system.db`. | `Get-Process python` | Stop any dangling background Python workers and restart the application. |
| **Browser Shows Blank Page or Infinite Spinner** | Web browser cached an incomplete session or websocket disconnected. | Hard refresh: `Ctrl + F5` | Hard refresh the browser window or test the health endpoint at `http://localhost:8501/_stcore/health`. |

---

## 8. Summary of Verification State

- **Server Status**: Active and listening on port `8501`.
- **Health Check**: `http://localhost:8501/_stcore/health` returns `200 OK` (`ok`).
- **Web App**: `http://localhost:8501` returns `200 OK` (Streamlit client bundle).
- **Unit Tests**: 33 of 33 tests passing cleanly.
- **Convenience Scripts**: `run_app.bat` and `run_app.ps1` provided in project root.
