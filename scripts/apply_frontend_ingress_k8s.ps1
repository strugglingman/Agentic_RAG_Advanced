param(
    [string]$ManifestFile = "config/k8s/frontend/rendered-local/04-ingress.from_local.yaml"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedManifest = if ([System.IO.Path]::IsPathRooted($ManifestFile)) {
    $ManifestFile
} else {
    Join-Path $repoRoot $ManifestFile
}

if (-not (Test-Path $resolvedManifest)) {
    throw "Frontend ingress manifest not found: $resolvedManifest"
}

kubectl apply -f $resolvedManifest

Write-Host "Applied frontend ingress manifest from $resolvedManifest"