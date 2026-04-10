param(
    [string]$RoleName = "agentic-rag-eks-node-role",
    [string]$TrustPolicyFile = "config/iam/agentic_rag_eks_node_role_trust_policy.template.json",
    [string[]]$ManagedPolicyArns = @(
        "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly",
        "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
    )
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

foreach ($managedPolicyArn in $ManagedPolicyArns) {
    $alreadyAttached = $attachedPolicies.AttachedPolicies | Where-Object { $_.PolicyArn -eq $managedPolicyArn }
    if ($null -eq $alreadyAttached) {
        Write-Host "Attaching managed policy: $managedPolicyArn"
        $attachOutput = aws iam attach-role-policy `
            --role-name $RoleName `
            --policy-arn $managedPolicyArn 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to attach managed policy '$managedPolicyArn': $attachOutput"
        }
    } else {
        Write-Host "Managed policy already attached: $managedPolicyArn"
    }
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
    managedPolicyArns = $ManagedPolicyArns
}

$output | ConvertTo-Json -Depth 5 | Write-Host
