# Keymake Client — setup for Windows (PowerShell)
# Usage: .\setup.ps1 [-NonInteractive] [-Provider anthropic] [-ApiKey KEY]
param(
    [switch]$NonInteractive,
    [string]$Provider,
    [string]$ApiKey,
    [string]$Model,
    [string]$BaseUrl,
    [ValidateSet("off","audit","enforce")]
    [string]$SecurityMode
)

$ErrorActionPreference = "Stop"
$MinPython = "3.11"

$step = 0

function Step($msg) {
    $script:step++
    Write-Host "`n  " -NoNewline; Write-Host "◆" -ForegroundColor DarkYellow -NoNewline
    Write-Host " Step ${script:step}: $msg" -ForegroundColor White
}

function Ok($msg) {
    Write-Host "  │ " -ForegroundColor DarkGray -NoNewline
    Write-Host "✓" -ForegroundColor Green -NoNewline
    Write-Host " $msg"
}

function Warn($msg) {
    Write-Host "  │ " -ForegroundColor DarkGray -NoNewline
    Write-Host "⚠" -ForegroundColor Yellow -NoNewline
    Write-Host " $msg"
}

function Fail($msg) {
    Write-Host "  │ " -ForegroundColor DarkGray -NoNewline
    Write-Host "✗ $msg" -ForegroundColor Red
    exit 1
}

function ShowIntro {
    Write-Host ""
    Write-Host "  ┌──────────────────────────────────────────────────────────┐" -ForegroundColor DarkYellow
    Write-Host "  │         " -ForegroundColor DarkYellow -NoNewline
    Write-Host "Keymake Client — Setup" -ForegroundColor White -NoNewline
    Write-Host "                │" -ForegroundColor DarkYellow
    Write-Host "  └──────────────────────────────────────────────────────────┘" -ForegroundColor DarkYellow
    Write-Host ""
}

function ShowOutro($msg) {
    Write-Host ""
    Write-Host "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "  ● " -ForegroundColor Green -NoNewline
    Write-Host "$msg" -ForegroundColor White
    Write-Host "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host ""
}

ShowIntro

# ── Step 1: Platform ─────────────────────────────────────────────────
Step "Detect platform"
Ok "Platform: Windows ($([Environment]::OSVersion.Version))"

# ── Step 2: Python check ────────────────────────────────────────────
Step "Check Python"

function Find-Python {
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver) {
                $parts = $ver.Split(".")
                $reqParts = $MinPython.Split(".")
                if ([int]$parts[0] -gt [int]$reqParts[0] -or
                    ([int]$parts[0] -eq [int]$reqParts[0] -and [int]$parts[1] -ge [int]$reqParts[1])) {
                    return $cmd
                }
            }
        } catch { }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Warn "Python ${MinPython}+ not found."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  │   Installing Python via winget..." -ForegroundColor DarkGray
        winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        $python = Find-Python
    }
}
if (-not $python) { Fail "Python ${MinPython}+ required. Install from https://python.org" }
$pyVer = & $python --version
Ok "Python: $pyVer"

# ── Step 3: Environment ─────────────────────────────────────────────
Step "Set up environment"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "  │   Creating virtual environment..." -ForegroundColor DarkGray
    & $python -m venv .venv
}
& .venv\Scripts\Activate.ps1
Ok "Virtual environment activated"

# ── Step 4: Dependencies ────────────────────────────────────────────
Step "Install dependencies"

Write-Host "  │   Upgrading pip..." -ForegroundColor DarkGray
pip install -q --upgrade pip
Write-Host "  │   Installing packages..." -ForegroundColor DarkGray
pip install -q -r requirements.txt
Ok "All dependencies installed"

# ── Step 5: Configure ───────────────────────────────────────────────
Step "Configure"

$onboardArgs = @()
if ($NonInteractive) { $onboardArgs += "--non-interactive" }
if ($Provider) { $onboardArgs += "--provider"; $onboardArgs += $Provider }
if ($ApiKey) { $onboardArgs += "--api-key"; $onboardArgs += $ApiKey }
if ($Model) { $onboardArgs += "--model"; $onboardArgs += $Model }
if ($BaseUrl) { $onboardArgs += "--base-url"; $onboardArgs += $BaseUrl }
if ($SecurityMode) { $onboardArgs += "--security-mode"; $onboardArgs += $SecurityMode }

& python onboard.py @onboardArgs

ShowOutro "Setup complete! Launch with: python run.py"
