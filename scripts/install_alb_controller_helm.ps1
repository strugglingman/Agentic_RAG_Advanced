param(
    [string]$TemplateFile = "config/eks/eks_alb_controller_helm_install.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/eks/eks_alb_controller_helm_install.from_local.json",
    [string]$ClusterName = "",
    [string]$Region = "",
    [string]$VpcId = "",
    [switch]$Apply,
    [switch]$WaitRollout,
    [int]$RolloutTimeoutSeconds = 180,
    [switch]$NoPause
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

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTemplate = Resolve-RepoPath -RepoRoot $repoRoot -Path $TemplateFile
$resolvedMap = Resolve-RepoPath -RepoRoot $repoRoot -Path $MapFile
$resolvedOutput = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputFile

if (-not (Test-Path $resolvedTemplate)) {
    throw "Template file not found: $resolvedTemplate"
}

# Step 1: Render Helm install config from template.
Pause-Step -Message "Step 1/6 - Render ALB controller Helm install config"
$templateContent = Get-Content $resolvedTemplate -Raw

$replacements = @{}
if (Test-Path $resolvedMap) {
    $replacements = Read-KeyValueFile -Path $resolvedMap
}
if ($ClusterName) {
    $replacements["<your-eks-cluster-name>"] = $ClusterName
}
if ($Region) {
    $replacements["<your-aws-region>"] = $Region
}
if ($VpcId) {
    $replacements["<your-vpc-id>"] = $VpcId
}

foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

$config = $templateContent | ConvertFrom-Json

# Explicit CLI arg should always win, even if template has no <your-vpc-id> placeholder.
if ($VpcId) {
    $config.vpcId = $VpcId
} elseif ((-not $config.vpcId) -and $replacements.ContainsKey("<your-vpc-id>")) {
    # Fallback for map-driven runs when template value is empty.
    $config.vpcId = $replacements["<your-vpc-id>"]
}

if (-not $config.clusterName -or $config.clusterName -eq "<your-eks-cluster-name>") {
    throw "Missing cluster name. Pass -ClusterName or set <your-eks-cluster-name> in $resolvedMap"
}
if (-not $config.region -or $config.region -eq "<your-aws-region>") {
    throw "Missing region. Pass -Region or set <your-aws-region> in $resolvedMap"
}

$rendered = [PSCustomObject]@{
    clusterName = $config.clusterName
    region = $config.region
    namespace = $config.namespace
    releaseName = $config.releaseName
    serviceAccount = $config.serviceAccount
    chartVersion = $config.chartVersion
    vpcId = $config.vpcId
}

$outputDir = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
($rendered | ConvertTo-Json -Depth 10) | Set-Content -Path $resolvedOutput -Encoding UTF8

Write-Host "Rendered Helm install config to: $resolvedOutput"

if (-not $Apply) {
    Write-Host "Render-only mode complete. Re-run with -Apply to install AWS Load Balancer Controller."
    return
}

# Step 2: Validate required tools.
Pause-Step -Message "Step 2/6 - Validate required CLI tools"
foreach ($tool in @("helm", "kubectl")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found in PATH"
    }
}

# Step 3: Add/update Helm repo.
Pause-Step -Message "Step 3/6 - Add and update Helm chart repo"
$repoListRaw = helm repo list -o yaml 2>$null
if ($LASTEXITCODE -ne 0 -or -not $repoListRaw -or ($repoListRaw -notmatch "name:\s*eks")) {
    helm repo add eks https://aws.github.io/eks-charts
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to add eks Helm repo"
    }
}
helm repo update eks
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update eks Helm repo"
}

# Step 4: Install/upgrade controller chart.
Pause-Step -Message "Step 4/6 - Install or upgrade AWS Load Balancer Controller"
$helmArgs = @(
    "upgrade",
    "--install",
    $rendered.releaseName,
    "eks/aws-load-balancer-controller",
    "-n", $rendered.namespace,
    "--set", "clusterName=$($rendered.clusterName)",
    "--set", "serviceAccount.create=false",
    "--set", "serviceAccount.name=$($rendered.serviceAccount)",
    "--version", "$($rendered.chartVersion)"
)

if (-not [string]::IsNullOrWhiteSpace($rendered.region) -and $rendered.region -ne "<your-aws-region>") {
    $helmArgs += @("--set", "region=$($rendered.region)")
}

if (-not [string]::IsNullOrWhiteSpace($rendered.vpcId) -and $rendered.vpcId -ne "<your-vpc-id>") {
    $helmArgs += @("--set", "vpcId=$($rendered.vpcId)")
}

$helmRaw = & helm @helmArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install/upgrade AWS Load Balancer Controller: $helmRaw"
}
Write-Host $helmRaw

# Step 5: Optional rollout wait.
if ($WaitRollout) {
    Pause-Step -Message "Step 5/6 - Wait for deployment rollout"
    kubectl rollout status deployment/$($rendered.releaseName) `
        -n $($rendered.namespace) `
        --timeout="$($RolloutTimeoutSeconds)s"
    if ($LASTEXITCODE -ne 0) {
        throw "Controller rollout did not complete within timeout."
    }
}

# Step 6: Verify resources.
Pause-Step -Message "Step 6/6 - Verify controller deployment and pods"
$deployRaw = kubectl get deployment $($rendered.releaseName) -n $($rendered.namespace) -o wide 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to get controller deployment: $deployRaw"
}
Write-Host $deployRaw

$podsRaw = kubectl get pods `
    -n $($rendered.namespace) `
    -l "app.kubernetes.io/name=aws-load-balancer-controller" `
    -o wide 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list controller pods: $podsRaw"
}
Write-Host $podsRaw
