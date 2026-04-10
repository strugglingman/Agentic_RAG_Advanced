param(
    [string]$TemplateFile = "config/eks/eks_managed_nodegroup_create.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_managed_nodegroup_create.from_local.json",
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
Write-Host "Rendered node group create JSON to: $resolvedOutput"

if (-not $Apply) {
    return
}

$applyOutput = aws eks create-nodegroup --cli-input-json "file://$resolvedOutput" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create managed node group: $applyOutput"
}

Write-Host $applyOutput

if ($WaitUntilActive) {
    $nodegroupName = ($templateContent | ConvertFrom-Json).nodegroupName
    $clusterName = ($templateContent | ConvertFrom-Json).clusterName
    Write-Host "Waiting for node group to become ACTIVE: $nodegroupName"
    aws eks wait nodegroup-active --cluster-name $clusterName --nodegroup-name $nodegroupName
    if ($LASTEXITCODE -ne 0) {
        throw "Managed node group did not become ACTIVE."
    }
}
