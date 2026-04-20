param(
    [string]$ControllerVersion = "v2.14.1",
    [string]$PolicyName = "AWSLoadBalancerControllerIAMPolicy",
    [string]$OutputPolicyFile = "config/iam/aws_load_balancer_controller_iam_policy.from_upstream.json",
    [string]$MetadataFile = "config/iam/alb_controller_iam_policy.from_local.json",
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

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutputPolicyFile = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputPolicyFile
$resolvedMetadataFile = Resolve-RepoPath -RepoRoot $repoRoot -Path $MetadataFile

$outputDir = Split-Path -Parent $resolvedOutputPolicyFile
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$policyUrl = "https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/$ControllerVersion/docs/install/iam_policy.json"

# Step 1: Download the controller IAM policy from the official upstream release.
Pause-Step -Message "Step 1/5 - Download official IAM policy JSON"
curl.exe -fsSL $policyUrl -o $resolvedOutputPolicyFile
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download IAM policy from: $policyUrl"
}
Write-Host "Downloaded IAM policy to: $resolvedOutputPolicyFile"

# Step 2: Validate JSON content before any AWS API call.
Pause-Step -Message "Step 2/5 - Validate downloaded IAM policy JSON"
$policyDocument = Get-Content $resolvedOutputPolicyFile -Raw | ConvertFrom-Json
if (-not $policyDocument.Statement -or $policyDocument.Statement.Count -eq 0) {
    throw "Downloaded IAM policy JSON has no statements: $resolvedOutputPolicyFile"
}

if (-not $Apply) {
    Write-Host "Render/download only mode complete. Re-run with -Apply to create IAM policy in AWS."
    return
}

# Step 3: Discover current AWS account and check whether policy already exists.
Pause-Step -Message "Step 3/5 - Check existing IAM policy"
$callerIdentityRaw = aws sts get-caller-identity --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve AWS caller identity: $callerIdentityRaw"
}
$accountId = ($callerIdentityRaw | ConvertFrom-Json).Account

$existingPolicyArnRaw = aws iam list-policies `
    --scope Local `
    --query "Policies[?PolicyName=='$PolicyName'].Arn | [0]" `
    --output text 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list local IAM policies: $existingPolicyArnRaw"
}

$existingPolicyArn = $existingPolicyArnRaw.Trim()
$policyArn = $null

# Step 4: Create policy if absent, otherwise reuse existing ARN.
Pause-Step -Message "Step 4/5 - Create or reuse IAM policy"
if (-not $existingPolicyArn -or $existingPolicyArn -eq "None") {
    $createPolicyRaw = aws iam create-policy `
        --policy-name $PolicyName `
        --policy-document "file://$resolvedOutputPolicyFile" `
        --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create IAM policy: $createPolicyRaw"
    }
    $policyArn = (($createPolicyRaw | ConvertFrom-Json).Policy.Arn)
    Write-Host "Created IAM policy: $policyArn"
} else {
    $policyArn = $existingPolicyArn
    Write-Host "IAM policy already exists: $policyArn"
}

# Step 5: Persist metadata for downstream scripts.
Pause-Step -Message "Step 5/5 - Save metadata for ALB controller install"
$metadataDir = Split-Path -Parent $resolvedMetadataFile
New-Item -ItemType Directory -Path $metadataDir -Force | Out-Null

$metadata = [PSCustomObject]@{
    policyName = $PolicyName
    policyArn = $policyArn
    accountId = $accountId
    controllerVersion = $ControllerVersion
    sourcePolicyUrl = $policyUrl
    sourcePolicyFile = $OutputPolicyFile
}

$metadataJson = $metadata | ConvertTo-Json -Depth 8
$metadataJson | Set-Content -Path $resolvedMetadataFile -Encoding UTF8

Write-Host "Saved policy metadata to: $resolvedMetadataFile"
Write-Host $metadataJson
