param(
    [string]$ManifestDir = "config/k8s/frontend/rendered-local"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedManifestDir = Join-Path $repoRoot $ManifestDir

if (-not (Test-Path $resolvedManifestDir)) {
    throw "Frontend manifest directory not found: $resolvedManifestDir"
}

$files = @(
    "00-configmap.from_local.yaml",
    "01-secret.from_local.yaml",
    "02-service.from_local.yaml",
    "03-deployment.from_local.yaml"
)

foreach ($file in $files) {
    $path = Join-Path $resolvedManifestDir $file
    if (-not (Test-Path $path)) {
        throw "Missing rendered frontend manifest: $path"
    }
    kubectl apply -f $path
}

Write-Host "Applied frontend manifests from $resolvedManifestDir"
