param(
    [string]$TemplateFile = "config/cloudwatch/cloudwatch_alarm_baseline.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/cloudwatch/cloudwatch_alarm_baseline.from_local.json",
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

function Assert-NoUnresolvedPlaceholders {
    param(
        [string]$Content,
        [string]$SourceLabel
    )

    $matches = [regex]::Matches($Content, "<[^>\r\n]+>")
    if ($matches.Count -eq 0) {
        return
    }

    $tokens = @($matches | ForEach-Object { $_.Value } | Sort-Object -Unique)
    $joined = $tokens -join ", "
    throw "Unresolved placeholders remain in ${SourceLabel}: $joined"
}

function Resolve-RdsInstanceIdFromEndpoint {
    param([hashtable]$Replacements)
    $endpointKey = "<your-rds-endpoint>"
    if (-not $Replacements.ContainsKey($endpointKey)) {
        return $null
    }
    $endpoint = $Replacements[$endpointKey]
    if ([string]::IsNullOrWhiteSpace($endpoint)) {
        return $null
    }
    return ($endpoint -split "\.")[0]
}

function Get-SnsTopicArnByName {
    param(
        [string]$TopicName,
        [string]$Region
    )

    $listOutput = aws sns list-topics --region $Region --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list SNS topics: $listOutput"
    }
    $topics = ($listOutput | ConvertFrom-Json).Topics
    $suffix = ":" + $TopicName
    $match = @($topics | Where-Object { $_.TopicArn.EndsWith($suffix) })
    if ($match.Count -eq 0) {
        throw "SNS topic not found: $TopicName (region $Region). Create topic first."
    }
    return $match[0].TopicArn
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

if ($templateContent.Contains("<your-rds-db-instance-identifier>")) {
    $derivedRdsId = Resolve-RdsInstanceIdFromEndpoint -Replacements $replacements
    if ([string]::IsNullOrWhiteSpace($derivedRdsId)) {
        throw "Cannot resolve <your-rds-db-instance-identifier>. Set it explicitly or ensure <your-rds-endpoint> exists in map."
    }
    $templateContent = $templateContent.Replace("<your-rds-db-instance-identifier>", $derivedRdsId)
}

foreach ($key in $replacements.Keys) {
    $templateContent = $templateContent.Replace($key, $replacements[$key])
}

Assert-NoUnresolvedPlaceholders -Content $templateContent -SourceLabel $resolvedTemplate

$config = $templateContent | ConvertFrom-Json
($config | ConvertTo-Json -Depth 20) | Set-Content $resolvedOutput -Encoding UTF8
Write-Host "Rendered CloudWatch baseline alarm JSON to: $resolvedOutput"

if (-not $Apply) {
    return
}

if ([string]::IsNullOrWhiteSpace($config.region)) {
    throw "region is required."
}
if ([string]::IsNullOrWhiteSpace($config.snsTopicName)) {
    throw "snsTopicName is required."
}

$topicArn = Get-SnsTopicArnByName -TopicName $config.snsTopicName -Region $config.region
Write-Host "Using SNS topic ARN: $topicArn"

$results = @()

foreach ($alarm in $config.alarms) {
    if ($alarm.enabled -eq $false) {
        continue
    }

    $baseArgs = @(
        "cloudwatch", "put-metric-alarm",
        "--region", $config.region,
        "--alarm-name", $alarm.alarmName,
        "--alarm-description", $alarm.alarmDescription,
        "--evaluation-periods", [string]$alarm.evaluationPeriods,
        "--datapoints-to-alarm", [string]$alarm.datapointsToAlarm,
        "--comparison-operator", $alarm.comparisonOperator,
        "--threshold", [string]$alarm.threshold,
        "--treat-missing-data", $alarm.treatMissingData,
        "--alarm-actions", $topicArn
    )

    if ($alarm.metricQueries) {
        $metricsJson = $alarm.metricQueries | ConvertTo-Json -Depth 20 -Compress
        $args = $baseArgs + @(
            "--metrics", $metricsJson
        )
    } else {
        $args = $baseArgs + @(
            "--namespace", $alarm.namespace,
            "--metric-name", $alarm.metricName,
            "--statistic", $alarm.statistic,
            "--period", [string]$alarm.period
        )

        if ($alarm.dimensions) {
            foreach ($d in $alarm.dimensions) {
                $args += @("--dimensions", "Name=$($d.Name),Value=$($d.Value)")
            }
        }
    }

    $output = & aws @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        $outputText = ($output | Out-String).Trim()
        throw "Failed to create/update alarm '$($alarm.alarmName)': $outputText"
    }

    $results += [PSCustomObject]@{
        alarmName = $alarm.alarmName
        status = "upserted"
    }
    Write-Host "Alarm upserted: $($alarm.alarmName)"
}

$resultPath = Join-Path $repoRoot "config/cloudwatch/cloudwatch_alarm_baseline_result.from_local.json"
($results | ConvertTo-Json -Depth 10) | Set-Content $resultPath -Encoding UTF8
Write-Host "Alarm creation summary saved to: $resultPath"
