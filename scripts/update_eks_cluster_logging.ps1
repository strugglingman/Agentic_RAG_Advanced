param(
    [string]$TemplateFile = "config/eks/eks_cluster_logging_enable.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_cluster_logging_enable.from_local.json",
    [string]$ClusterName = "",
    [string]$Region = "eu-north-1",
    [switch]$Apply,
    [switch]$WaitUntilActive,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTemplate = Join-Path $repoRoot $TemplateFile
$resolvedMap = Join-Path $repoRoot $MapFile
$resolvedOutput = Join-Path $repoRoot $OutputFile

function Pause-Step {
    param([string]$Message)
    if (-not $NoPause) {
        Read-Host "$Message`nPress Enter to continue"
    }
}

function Read-KeyValueFile {
    param([string]$Path)

    $result = @{}
    foreach ($line in Get-Content $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.TrimStart().StartsWith("#")) { continue }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) { continue }
        $result[$parts[0].Trim()] = $parts[1].Trim()
    }
    return $result
}

if (-not (Test-Path $resolvedTemplate)) {
    throw "Template file not found: $resolvedTemplate"
}

# Step 1: Render template with local placeholder values.
$templateContent = Get-Content $resolvedTemplate -Raw
$replacements = @{}

if (Test-Path $resolvedMap) {
    $replacements = Read-KeyValueFile -Path $resolvedMap
}

if ($ClusterName) {
    $replacements["<your-eks-cluster-name>"] = $ClusterName
}

if (-not $replacements.ContainsKey("<your-eks-cluster-name>")) {
    throw "Missing cluster name. Provide -ClusterName or add <your-eks-cluster-name> in $resolvedMap"
}

foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

$templateContent | Set-Content $resolvedOutput -Encoding UTF8
Write-Host "Rendered logging update payload to: $resolvedOutput"

# Step 2: Show rendered payload before any AWS command runs.
Pause-Step -Message "Step 2/5 - Review rendered payload"
Get-Content $resolvedOutput

if (-not $Apply) {
    Write-Host "Render-only mode complete. Re-run with -Apply to execute AWS commands."
    return
}

$renderedObject = $templateContent | ConvertFrom-Json
$clusterName = $renderedObject.name

if (-not $clusterName) {
    throw "Cluster name missing in rendered payload."
}

# Step 3: Update EKS cluster config to enable control plane logging.
Pause-Step -Message "Step 3/5 - Run aws eks update-cluster-config"
aws eks update-cluster-config `
    --region $Region `
    --cli-input-json "file://$resolvedOutput"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update EKS cluster logging config."
}

# Step 4: Optionally wait until cluster returns to ACTIVE state.
if ($WaitUntilActive) {
    Pause-Step -Message "Step 4/5 - Wait for cluster to become ACTIVE"
    aws eks wait cluster-active --name $clusterName --region $Region
    if ($LASTEXITCODE -ne 0) {
        throw "Cluster did not become ACTIVE: $clusterName"
    }
}

# Step 5: Verify enabled control plane logging types.
Pause-Step -Message "Step 5/5 - Verify cluster logging configuration"
aws eks describe-cluster `
    --region $Region `
    --name $clusterName `
    --query "cluster.logging.clusterLogging"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to verify cluster logging settings."
}

Write-Host "EKS control plane logging update flow completed."

# Future extension point:
# Add new EKS hardening command blocks below with the same
# Pause-Step -> command -> exit-code check pattern.
