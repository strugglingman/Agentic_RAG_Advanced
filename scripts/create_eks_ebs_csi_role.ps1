param(
    [string]$RoleName = "agentic-rag-eks-ebs-csi-role",
    [string]$TrustPolicyFile = "config/iam/agentic_rag_eks_ebs_csi_role_trust_policy.template.json",
    [string]$ManagedPolicyArn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTrustPolicy = Join-Path $repoRoot $TrustPolicyFile

if (-not (Test-Path $resolvedTrustPolicy)) {
    throw "Trust policy file not found: $resolvedTrustPolicy"
}

$trustPolicyFileUri = "file://$resolvedTrustPolicy"

$existingRole = $null
try {
    $existingRoleJson = aws iam get-role --role-name $RoleName 2>$null
    if ($LASTEXITCODE -eq 0 -and $existingRoleJson) {
        $existingRole = $existingRoleJson | ConvertFrom-Json
    }
} catch {
}

if ($null -eq $existingRole) {
    Write-Host "Creating IAM role: $RoleName"
    $createOutput = aws iam create-role `
        --role-name $RoleName `
        --assume-role-policy-document $trustPolicyFileUri 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create IAM role: $createOutput"
    }
} else {
    Write-Host "IAM role already exists: $RoleName"
}

$attachedPoliciesJson = aws iam list-attached-role-policies --role-name $RoleName 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list attached role policies: $attachedPoliciesJson"
}
$attachedPolicies = $attachedPoliciesJson | ConvertFrom-Json
$alreadyAttached = $attachedPolicies.AttachedPolicies | Where-Object { $_.PolicyArn -eq $ManagedPolicyArn }

if ($null -eq $alreadyAttached) {
    Write-Host "Attaching managed policy: $ManagedPolicyArn"
    $attachOutput = aws iam attach-role-policy `
        --role-name $RoleName `
        --policy-arn $ManagedPolicyArn 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to attach managed policy: $attachOutput"
    }
} else {
    Write-Host "Managed policy already attached: $ManagedPolicyArn"
}

$roleJson = aws iam get-role --role-name $RoleName 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to fetch IAM role after creation: $roleJson"
}
$role = $roleJson | ConvertFrom-Json
$output = [PSCustomObject]@{
    roleName = $role.Role.RoleName
    arn = $role.Role.Arn
    trustPolicyFile = $TrustPolicyFile
    managedPolicyArn = $ManagedPolicyArn
}

$outputJson = $output | ConvertTo-Json -Depth 5
$outputPath = Join-Path $repoRoot "config/iam/eks_ebs_csi_role.from_local.json"
$outputJson | Set-Content $outputPath -Encoding UTF8

Write-Host "Role ready:"
Write-Host $outputJson
Write-Host "Saved metadata to: $outputPath"
