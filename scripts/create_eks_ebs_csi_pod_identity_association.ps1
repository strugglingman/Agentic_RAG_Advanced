param(
    [string]$TemplateFile = "config/eks/eks_ebs_csi_pod_identity_association.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$RoleName = "agentic-rag-eks-ebs-csi-role",
    [switch]$Apply
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

$config = $templateContent | ConvertFrom-Json

$roleJson = aws iam get-role --role-name $RoleName 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve role ARN for '$RoleName': $roleJson"
}
$roleArn = (($roleJson | ConvertFrom-Json).Role.Arn)

$rendered = [PSCustomObject]@{
    clusterName = $config.clusterName
    namespace = $config.namespace
    serviceAccount = $config.serviceAccount
    roleName = $RoleName
    roleArn = $roleArn
}

$outputPath = Join-Path $repoRoot "config/eks/eks_ebs_csi_pod_identity_association.from_local.json"
($rendered | ConvertTo-Json -Depth 5) | Set-Content $outputPath -Encoding UTF8
Write-Host "Rendered pod identity association metadata to: $outputPath"

if (-not $Apply) {
    return
}

$existingJson = aws eks list-pod-identity-associations --cluster-name $config.clusterName 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list pod identity associations: $existingJson"
}
$existing = $existingJson | ConvertFrom-Json
$matching = @($existing.associations | Where-Object {
    $_.namespace -eq $config.namespace -and $_.serviceAccount -eq $config.serviceAccount
})

if ($matching.Count -gt 0) {
    Write-Host "Pod identity association already exists for $($config.namespace)/$($config.serviceAccount)."
    return
}

$createOutput = aws eks create-pod-identity-association `
    --cluster-name $config.clusterName `
    --namespace $config.namespace `
    --service-account $config.serviceAccount `
    --role-arn $roleArn 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create pod identity association: $createOutput"
}

Write-Host $createOutput
