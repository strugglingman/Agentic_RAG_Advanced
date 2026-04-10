param(
    [string]$TemplateFile = "config/eks/eks_cluster_create.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_cluster_create.from_local.json",
    [switch]$Apply,
    [switch]$WaitUntilActive
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTemplate = Join-Path $repoRoot $TemplateFile
$resolvedMap = Join-Path $repoRoot $MapFile
$resolvedOutput = Join-Path $repoRoot $OutputFile

if (-not (Test-Path $resolvedTemplate)) {
    throw "Template file not found: $resolvedTemplate"
}
if (-not (Test-Path $resolvedMap)) {
    throw "Map file not found: $resolvedMap"
}

$template = Get-Content $resolvedTemplate -Raw
$mapLines = Get-Content $resolvedMap
$replacements = @{}

foreach ($line in $mapLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.TrimStart().StartsWith("#")) { continue }
    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) { continue }
    $replacements[$parts[0]] = $parts[1]
}

foreach ($key in $replacements.Keys) {
    $escapedKey = [regex]::Escape($key)
    $template = [regex]::Replace($template, $escapedKey, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $replacements[$key] })
}

$renderedObject = $template | ConvertFrom-Json
$renderedJson = $renderedObject | ConvertTo-Json -Depth 20
$renderedJson | Set-Content $resolvedOutput -Encoding UTF8

Write-Host "Rendered cluster config to: $resolvedOutput"

if (-not $Apply) {
    Write-Host "Render only mode. Use -Apply to call aws eks create-cluster."
    return
}

$outputUri = "file://$resolvedOutput"
$createResult = aws eks create-cluster --cli-input-json $outputUri 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create EKS cluster: $createResult"
}

Write-Host $createResult

if ($WaitUntilActive) {
    $clusterName = $renderedObject.name
    Write-Host "Waiting for EKS cluster to become ACTIVE: $clusterName"
    aws eks wait cluster-active --name $clusterName
    if ($LASTEXITCODE -ne 0) {
        throw "Cluster wait failed for: $clusterName"
    }
    Write-Host "Cluster is ACTIVE: $clusterName"
}
