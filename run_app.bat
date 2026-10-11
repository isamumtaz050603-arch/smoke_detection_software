@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"

rem Open the existing app when its server is already healthy.
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try { $app=Join-Path (Get-Location).Path 'frontend\app.py'; $listeners=@(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue); foreach ($listener in $listeners) { $proc=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $listener.OwningProcess); if ($proc.CommandLine -and $proc.CommandLine.Contains($app) -and $proc.CommandLine -match '-m\s+streamlit\s+run') { if ((Invoke-RestMethod -Uri 'http://localhost:8501/_stcore/health' -TimeoutSec 5) -eq 'ok') { exit 0 } } }; exit 1 } catch { exit 1 }"
if not errorlevel 1 (
    echo Vehicle Smoke Detection is already running. Opening the app...
    start "" "http://localhost:8501"
    exit /b 0
)

if not exist "%APP_PYTHON%" (
    echo ERROR: The app's Python environment is missing: .venv\Scripts\python.exe
    pause
    exit /b 1
)

"%APP_PYTHON%" -c "import streamlit, torch, cv2" 
if errorlevel 1 (
    echo.
    echo The app could not load its required dependencies. See the error above.
    echo If Windows reports an Application Control block, ask your administrator
    echo to review the blocked dependency. The launcher cannot override that policy.
    pause
    exit /b 1
)

echo Starting Vehicle Smoke Detection at http://localhost:8501
echo Keep this window open while using the app. Press Ctrl+C to stop.
"%APP_PYTHON%" -m streamlit run "%~dp0frontend\app.py" --server.address localhost --server.port 8501 --server.headless false
if errorlevel 1 (
    echo.
    echo The app stopped with an error. See the message above.
    pause
    exit /b 1
)
endlocal
