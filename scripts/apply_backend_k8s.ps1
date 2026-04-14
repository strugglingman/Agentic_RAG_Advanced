param(
    [string]$RenderedDir = "config/k8s/backend/rendered-local"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRenderedDir = Join-Path $repoRoot $RenderedDir

if (-not (Test-Path $resolvedRenderedDir)) {
    throw "Rendered backend manifest directory not found: $resolvedRenderedDir"
}

$manifestFiles = @(
    "00-serviceaccount.from_local.yaml",
    "00-configmap.from_local.yaml",
    "01-secret.from_local.yaml",
    "02-service.from_local.yaml",
    "03-deployment.from_local.yaml"
)

foreach ($manifestFile in $manifestFiles) {
    $manifestPath = Join-Path $resolvedRenderedDir $manifestFile
    if (-not (Test-Path $manifestPath)) {
        throw "Missing rendered manifest: $manifestPath"
    }

    Write-Host "Applying $manifestPath"
    kubectl apply -f $manifestPath
}
