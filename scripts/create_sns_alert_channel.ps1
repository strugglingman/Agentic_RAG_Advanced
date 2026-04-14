param(
    [string]$TemplateFile = "config/cloudwatch/sns_alert_channel.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/cloudwatch/sns_alert_channel.from_local.json",
    [string]$EmailEndpoint = "",
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

function Get-ReplacementsFromMap {
    param([string]$Path)
    $mapLines = Get-Content $Path
    $replacements = @{}
    foreach ($line in $mapLines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.Trim().StartsWith("#")) { continue }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) { continue }
        $replacements[$parts[0].Trim()] = $parts[1].Trim()
    }
    return $replacements
}

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
$replacements = Get-ReplacementsFromMap -Path $resolvedMap
foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

$config = $templateContent | ConvertFrom-Json
if (-not [string]::IsNullOrWhiteSpace($EmailEndpoint)) {
    $config.emailEndpoint = $EmailEndpoint
}

($config | ConvertTo-Json -Depth 10) | Set-Content $resolvedOutput -Encoding UTF8
Write-Host "Rendered SNS alert channel JSON to: $resolvedOutput"

if (-not $Apply) {
    return
}

if ([string]::IsNullOrWhiteSpace($config.topicName)) {
    throw "topicName is required."
}
if ([string]::IsNullOrWhiteSpace($config.region)) {
    throw "region is required."
}
if ([string]::IsNullOrWhiteSpace($config.emailEndpoint) -or $config.emailEndpoint -eq "<your-alert-email>") {
    throw "emailEndpoint is required. Pass -EmailEndpoint or update template."
}

$createTopicOutput = aws sns create-topic `
    --name $config.topicName `
    --region $config.region `
    --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create/get SNS topic: $createTopicOutput"
}

$topicArn = (($createTopicOutput | ConvertFrom-Json).TopicArn)
if ([string]::IsNullOrWhiteSpace($topicArn)) {
    throw "Failed to resolve topic ARN."
}

$subscriptionsOutput = aws sns list-subscriptions-by-topic `
    --topic-arn $topicArn `
    --region $config.region `
    --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list subscriptions: $subscriptionsOutput"
}

$subscriptions = ($subscriptionsOutput | ConvertFrom-Json).Subscriptions
$existing = @($subscriptions | Where-Object {
    $_.Protocol -eq "email" -and $_.Endpoint -eq $config.emailEndpoint
})

if ($existing.Count -gt 0) {
    Write-Host "Email subscription already exists: $($config.emailEndpoint)"
    $subscriptionArn = $existing[0].SubscriptionArn
} else {
    $subscribeOutput = aws sns subscribe `
        --topic-arn $topicArn `
        --protocol email `
        --notification-endpoint $config.emailEndpoint `
        --region $config.region `
        --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create email subscription: $subscribeOutput"
    }
    $subscriptionArn = (($subscribeOutput | ConvertFrom-Json).SubscriptionArn)
    Write-Host "Email subscription created. Confirmation is required from inbox: $($config.emailEndpoint)"
}

$result = [PSCustomObject]@{
    region = $config.region
    topicName = $config.topicName
    topicArn = $topicArn
    emailEndpoint = $config.emailEndpoint
    subscriptionArn = $subscriptionArn
}

$resultPath = Join-Path $repoRoot "config/cloudwatch/sns_alert_channel_result.from_local.json"
($result | ConvertTo-Json -Depth 10) | Set-Content $resultPath -Encoding UTF8
Write-Host "SNS alert channel ready:"
Write-Host ($result | ConvertTo-Json -Depth 10)
Write-Host "Saved result to: $resultPath"
