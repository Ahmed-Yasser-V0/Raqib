@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Raqib is not set up yet. Running setup first...
    call setup.bat
)
set PYTHONUTF8=1
rem Close an older copy of Raqib that is still running (it keeps the cameras busy)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$venv = (Resolve-Path '.venv').Path; $all = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\"; $old = @($all | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venv) -and $_.CommandLine -like '*app.py*' }); $ids = @($old | ForEach-Object { $_.ProcessId }); $kids = @($all | Where-Object { $ids -contains $_.ParentProcessId }); foreach ($p in ($kids + $old)) { Write-Host ('Closing the old Raqib server (process ' + $p.ProcessId + ')'); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
echo Starting Raqib... (http://localhost:5002)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\.venv\Scripts\python.exe' -u app.py 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $(try { [IO.File]::Open('run_log.txt','OpenOrCreate','ReadWrite','None').Close(); 'run_log.txt' } catch { 'run_log_' + (Get-Date -Format 'HHmmss') + '.txt' })"
pause
