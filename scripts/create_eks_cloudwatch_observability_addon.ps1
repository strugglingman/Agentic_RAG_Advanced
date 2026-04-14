param(
    [string]$TemplateFile = "config/eks/eks_addon_cloudwatch_observability_create.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_addon_cloudwatch_observability_create.from_local.json",
    [switch]$Apply,
    [switch]$WaitUntilActive
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param(
        [string]$RepoRoot,
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }

    return (Join-Path $RepoRoot $Path)
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTemplate = Resolve-RepoPath -RepoRoot $repoRoot -Path $TemplateFile
$resolvedMap = Resolve-RepoPath -RepoRoot $repoRoot -Path $MapFile
$resolvedOutput = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputFile

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
Write-Host "Rendered CloudWatch Observability addon create JSON to: $resolvedOutput"

if (-not $Apply) {
    return
}

$config = $templateContent | ConvertFrom-Json

$existingAddonJson = aws eks describe-addon --cluster-name $config.clusterName --addon-name $config.addonName 2>$null
if ($LASTEXITCODE -eq 0 -and $existingAddonJson) {
    Write-Host "Addon already exists: $($config.addonName)"
} else {
    $applyOutput = aws eks create-addon --cli-input-json "file://$resolvedOutput" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create CloudWatch Observability addon: $applyOutput"
    }
    Write-Host $applyOutput
}

if ($WaitUntilActive) {
    Write-Host "Waiting for addon to become ACTIVE: $($config.addonName)"
    aws eks wait addon-active --cluster-name $config.clusterName --addon-name $config.addonName
    if ($LASTEXITCODE -ne 0) {
        throw "CloudWatch Observability addon did not become ACTIVE."
    }
}
