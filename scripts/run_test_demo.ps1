param(
    [switch]$WithIntegration,
    [switch]$KeepInfra
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Run-Step {
    param(
        [string]$Name,
        [string]$WorkingDir,
        [scriptblock]$Action
    )

    $start = Get-Date
    $status = "PASS"

    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    Write-Host "Dir: $WorkingDir"

    Push-Location $WorkingDir
    try {
        & $Action
    }
    catch {
        $status = "FAIL"
        throw
    }
    finally {
        Pop-Location
        $elapsed = (Get-Date) - $start
        $script:results += [pscustomobject]@{
            Step    = $Name
            Status  = $status
            Seconds = [math]::Round($elapsed.TotalSeconds, 1)
        }
        if ($status -eq "PASS") {
            Write-Host "[PASS] $Name ($([math]::Round($elapsed.TotalSeconds, 1))s)" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $Name ($([math]::Round($elapsed.TotalSeconds, 1))s)" -ForegroundColor Red
        }
    }
}

$scriptStart = Get-Date
$results = @()

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"
$infraStarted = $false

Write-Host "Repository: $repoRoot"
Write-Host "WithIntegration: $WithIntegration"
Write-Host "KeepInfra: $KeepInfra"

try {
    Run-Step -Name "Backend Fast Lane" -WorkingDir $backendDir -Action {
        & .\run_tests.bat fast
        Assert-LastExitCode "Backend Fast Lane"
    }

    if ($WithIntegration) {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            throw "Docker CLI not found. Install Docker Desktop or run without -WithIntegration."
        }

        Run-Step -Name "Start Integration Infra (Postgres/Redis/Qdrant)" -WorkingDir $repoRoot -Action {
            & docker compose up -d postgres redis qdrant
            Assert-LastExitCode "docker compose up"
        }
        $infraStarted = $true

        Run-Step -Name "Backend Integration Lane" -WorkingDir $backendDir -Action {
            & .\run_tests.bat integration
            Assert-LastExitCode "Backend Integration Lane"
        }
    }

    Run-Step -Name "Frontend Unit Tests" -WorkingDir $frontendDir -Action {
        & npm.cmd run test
        Assert-LastExitCode "Frontend Unit Tests"
    }

    Run-Step -Name "Frontend E2E Smoke (Playwright)" -WorkingDir $frontendDir -Action {
        & npm.cmd run test:e2e
        Assert-LastExitCode "Frontend E2E"
    }
}
finally {
    if ($infraStarted -and -not $KeepInfra) {
        Write-Host ""
        Write-Host "=== Stop Integration Infra ===" -ForegroundColor Yellow
        Push-Location $repoRoot
        try {
            & docker compose down -v
        }
        finally {
            Pop-Location
        }
    }
}

$total = (Get-Date) - $scriptStart

Write-Host ""
Write-Host "=== Test Demo Summary ===" -ForegroundColor Magenta
$results | Format-Table -AutoSize
Write-Host ("Total Time: {0}s" -f [math]::Round($total.TotalSeconds, 1))

