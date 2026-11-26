# Raqib - one-time setup for Windows (run via setup.bat)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "setup_log.txt"
"Setup started $(Get-Date)" | Out-File $log -Encoding utf8
function Say($t, $c = "Cyan") { Write-Host ""; Write-Host "==== $t" -ForegroundColor $c; "==== $t" | Out-File $log -Append -Encoding utf8 }
$py = $null
foreach ($c in @("C:\Python314\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
                 "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")) {
    if (Test-Path $c) { $py = $c; break }
}
if (-not $py) { $cmd = Get-Command python -ErrorAction SilentlyContinue; if ($cmd) { $py = $cmd.Source } }
if (-not $py) { Say "Python was not found. Install it from python.org and run setup.bat again." "Red"; Read-Host "Press Enter to close"; exit 1 }
Say "1/2  Creating the project environment (.venv) with $py"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { & $py -m venv .venv 2>&1 | Tee-Object -FilePath $log -Append | Out-Host }
Say "2/2  Installing libraries (about 1 GB the first time)"
& $venvPy -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
& $venvPy -m pip install -r requirements.txt 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
if ($LASTEXITCODE -ne 0) { Say "Install FAILED. Send setup_log.txt to Claude." "Red"; Read-Host "Press Enter to close"; exit 1 }
Say "SETUP FINISHED. Double-click run.bat to start Raqib." "Green"
Read-Host "Press Enter to close"
