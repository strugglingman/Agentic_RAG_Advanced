param(
    [string]$KustomizeDir = "config/k8s/qdrant"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedDir = Join-Path $repoRoot $KustomizeDir

if (-not (Test-Path $resolvedDir)) {
    throw "Kustomize directory not found: $resolvedDir"
}

Write-Host "Applying qdrant Kubernetes resources from: $resolvedDir"
kubectl apply -k $resolvedDir
