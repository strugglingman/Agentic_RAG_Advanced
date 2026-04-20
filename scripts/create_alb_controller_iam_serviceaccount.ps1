param(
    [string]$TemplateFile = "config/eks/eks_alb_controller_iam_serviceaccount.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$PolicyMetadataFile = "config/iam/alb_controller_iam_policy.from_local.json",
    [string]$OutputFile = "config/eks/eks_alb_controller_iam_serviceaccount.from_local.json",
    [string]$ClusterName = "",
    [string]$Region = "",
    [string]$PolicyArn = "",
    [switch]$Apply,
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
$resolvedPolicyMetadata = Resolve-RepoPath -RepoRoot $repoRoot -Path $PolicyMetadataFile
$resolvedOutput = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputFile

if (-not (Test-Path $resolvedTemplate)) {
    throw "Template file not found: $resolvedTemplate"
}

# Step 1: Render install config from template + map + explicit args.
Pause-Step -Message "Step 1/5 - Render ALB controller IAM service account config"
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

foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

$config = $templateContent | ConvertFrom-Json

if (-not $config.clusterName -or $config.clusterName -eq "<your-eks-cluster-name>") {
    throw "Missing cluster name. Pass -ClusterName or set <your-eks-cluster-name> in $resolvedMap"
}
if (-not $config.region -or $config.region -eq "<your-aws-region>") {
    throw "Missing region. Pass -Region or set <your-aws-region> in $resolvedMap"
}

$resolvedPolicyArn = $PolicyArn
if (-not $resolvedPolicyArn -and (Test-Path $resolvedPolicyMetadata)) {
    $policyMetadata = Get-Content $resolvedPolicyMetadata -Raw | ConvertFrom-Json
    $resolvedPolicyArn = $policyMetadata.policyArn
}

if (-not $resolvedPolicyArn) {
    throw "Missing policy ARN. Pass -PolicyArn or create it first with scripts/create_alb_controller_iam_policy.ps1"
}

$rendered = [PSCustomObject]@{
    clusterName = $config.clusterName
    region = $config.region
    namespace = $config.namespace
    serviceAccount = $config.serviceAccount
    roleName = $config.roleName
    policyName = $config.policyName
    policyArn = $resolvedPolicyArn
}

$outputDir = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
($rendered | ConvertTo-Json -Depth 10) | Set-Content -Path $resolvedOutput -Encoding UTF8

Write-Host "Rendered IAM service account config to: $resolvedOutput"

if (-not $Apply) {
    Write-Host "Render-only mode complete. Re-run with -Apply to execute eksctl create iamserviceaccount."
    return
}

# Step 2: Check tool availability.
Pause-Step -Message "Step 2/5 - Validate required CLI tools"
if (-not (Get-Command eksctl -ErrorAction SilentlyContinue)) {
    throw "eksctl not found in PATH"
}
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "kubectl not found in PATH"
}

# Step 3: Create or update IAM service account with eksctl.
Pause-Step -Message "Step 3/5 - Run eksctl create iamserviceaccount"
$eksctlRaw = eksctl create iamserviceaccount `
    --cluster=$($rendered.clusterName) `
    --namespace=$($rendered.namespace) `
    --name=$($rendered.serviceAccount) `
    --attach-policy-arn=$($rendered.policyArn) `
    --override-existing-serviceaccounts `
    --region=$($rendered.region) `
    --role-name=$($rendered.roleName) `
    --approve 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create/update IAM service account: $eksctlRaw"
}
Write-Host $eksctlRaw

# Step 4: Verify service account annotation.
Pause-Step -Message "Step 4/5 - Verify service account annotation"
$saRaw = kubectl get serviceaccount $($rendered.serviceAccount) `
    -n $($rendered.namespace) `
    -o json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to fetch service account after eksctl run: $saRaw"
}
$sa = $saRaw | ConvertFrom-Json
$roleArn = $sa.metadata.annotations."eks.amazonaws.com/role-arn"
if (-not $roleArn) {
    throw "Service account exists but is missing eks.amazonaws.com/role-arn annotation."
}
Write-Host "Service account role annotation: $roleArn"

# Step 5: Save final metadata.
Pause-Step -Message "Step 5/5 - Save resolved IAM service account metadata"
$finalMetadata = [PSCustomObject]@{
    clusterName = $rendered.clusterName
    region = $rendered.region
    namespace = $rendered.namespace
    serviceAccount = $rendered.serviceAccount
    roleName = $rendered.roleName
    roleArn = $roleArn
    policyArn = $rendered.policyArn
}
$finalJson = $finalMetadata | ConvertTo-Json -Depth 10
$finalJson | Set-Content -Path $resolvedOutput -Encoding UTF8
Write-Host "Saved final IAM service account metadata to: $resolvedOutput"
Write-Host $finalJson
