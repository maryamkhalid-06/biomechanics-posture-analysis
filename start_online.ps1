param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-PythonCommand {
    $venvPython = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python was not found. Install Python or create .venv first."
}

function Test-BackendHealth {
    param([int]$HealthPort)

    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$HealthPort/api/health" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

$cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
if (-not $cloudflared) {
    Write-Host ""
    Write-Host "cloudflared is not installed yet."
    Write-Host "1. Download it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    Write-Host "2. Install it"
    Write-Host "3. Run this script again: .\start_online.ps1"
    exit 1
}

$python = Get-PythonCommand
$backendOut = Join-Path $root "backend_stdout.log"
$backendErr = Join-Path $root "backend_stderr.log"

if (-not (Test-BackendHealth -HealthPort $Port)) {
    Write-Host "Starting the backend on http://127.0.0.1:$Port ..."
    $env:PORT = "$Port"
    $backendProcess = Start-Process `
        -FilePath $python `
        -ArgumentList "run_backend.py" `
        -WorkingDirectory $root `
        -RedirectStandardOutput $backendOut `
        -RedirectStandardError $backendErr `
        -WindowStyle Hidden `
        -PassThru

    $started = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (Test-BackendHealth -HealthPort $Port) {
            $started = $true
            break
        }

        if ($backendProcess.HasExited) {
            break
        }
    }

    if (-not $started) {
        Write-Host ""
        Write-Host "The backend did not become ready."
        if (Test-Path $backendErr) {
            Write-Host "Check: $backendErr"
        }
        if (Test-Path $backendOut) {
            Write-Host "Check: $backendOut"
        }
        exit 1
    }
}
else {
    Write-Host "Backend is already running on http://127.0.0.1:$Port"
}

Write-Host ""
Write-Host "Opening a public Cloudflare Tunnel for http://127.0.0.1:$Port"
Write-Host "Keep this window open while you want the public link to stay online."
Write-Host ""

& $cloudflared.Source tunnel --url "http://127.0.0.1:$Port"
