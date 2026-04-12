param(
    [string]$TemplateFile = "config/eks/eks_control_plane_logging_enable.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_control_plane_logging_enable.from_local.json",
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

$templateContent = Get-Content $resolvedTemplate -Raw
$mapLines = Get-Content $resolvedMap
$replacements = @{}

foreach ($line in $mapLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.Trim().StartsWith("#")) { continue }
    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) { continue }
    $replacements[$parts[0].Trim()] = $parts[1].Trim()
}

foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

$templateContent | Set-Content $resolvedOutput -Encoding UTF8
Write-Host "Rendered EKS control plane logging config to: $resolvedOutput"

if (-not $Apply) {
    Write-Host "Render only mode. Use -Apply to call aws eks update-cluster-config."
    return
}

$config = $templateContent | ConvertFrom-Json
$loggingJson = $config.logging | ConvertTo-Json -Compress -Depth 20

$updateOutput = aws eks update-cluster-config `
    --name $config.name `
    --logging $loggingJson 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update EKS control plane logging: $updateOutput"
}

Write-Host $updateOutput

if ($WaitUntilActive) {
    Write-Host "Waiting for EKS cluster to become ACTIVE after logging update: $($config.name)"
    aws eks wait cluster-active --name $config.name
    if ($LASTEXITCODE -ne 0) {
        throw "Cluster did not return to ACTIVE state after logging update: $($config.name)"
    }
}
