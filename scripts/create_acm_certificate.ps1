param(
    [string]$TemplateFile = "config/acm/acm_certificate_request.template.json",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$OutputFile = "config/acm/acm_certificate_request.from_local.json",
    [string]$DomainName,
    [string[]]$SubjectAlternativeNames,
    [string]$Region,
    [switch]$Apply,
    [switch]$WaitForDnsValidationRecords,
    [int]$MaxPollAttempts = 24,
    [int]$PollIntervalSeconds = 5
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

function Read-KeyValueFile {
    param([string]$Path)

    $result = @{}

    foreach ($line in Get-Content $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.TrimStart().StartsWith("#")) { continue }

        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) { continue }

        $result[$parts[0].Trim()] = $parts[1]
    }

    return $result
}

function Get-MapValue {
    param(
        [hashtable]$MapValues,
        [string]$Key
    )

    if ($MapValues.ContainsKey($Key) -and -not [string]::IsNullOrWhiteSpace($MapValues[$Key])) {
        return $MapValues[$Key]
    }

    return $null
}

function Get-IdempotencyToken {
    param(
        [string]$PrimaryDomain,
        [string[]]$AlternativeNames
    )

    $tokenSource = "$PrimaryDomain|$($AlternativeNames -join ',')"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($tokenSource)
    $hashBytes = [System.Security.Cryptography.SHA256]::HashData($bytes)
    $hex = [Convert]::ToHexString($hashBytes).ToLowerInvariant()
    return $hex.Substring(0, 32)
}

function Assert-NoUnresolvedPlaceholders {
    param([string]$Content)

    $matches = [regex]::Matches($Content, "<your-[^>]+>")
    if ($matches.Count -eq 0) {
        return
    }

    $remaining = $matches | ForEach-Object { $_.Value } | Select-Object -Unique
    throw "Unresolved placeholders remain in certificate template: $($remaining -join ', ')"
}

function Write-ValidationRecords {
    param([object]$Certificate)

    $records = @($Certificate.DomainValidationOptions | Where-Object { $_.ResourceRecord })
    if ($records.Count -eq 0) {
        Write-Host "Validation CNAMEs are not available yet."
        return $false
    }

    Write-Host "DNS validation records:"
    foreach ($record in $records) {
        Write-Host "- Domain: $($record.DomainName)"
        Write-Host "  Name:   $($record.ResourceRecord.Name)"
        Write-Host "  Type:   $($record.ResourceRecord.Type)"
        Write-Host "  Value:  $($record.ResourceRecord.Value)"
    }

    return $true
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedTemplate = Resolve-RepoPath -RepoRoot $repoRoot -Path $TemplateFile
$resolvedOutput = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputFile
$resolvedMap = Resolve-RepoPath -RepoRoot $repoRoot -Path $MapFile

if (-not (Test-Path $resolvedTemplate)) {
    throw "Template file not found: $resolvedTemplate"
}

$mapValues = @{}
if (Test-Path $resolvedMap) {
    $mapValues = Read-KeyValueFile -Path $resolvedMap
}

$resolvedDomainName = if ($DomainName) { $DomainName } else { Get-MapValue -MapValues $mapValues -Key "<your-primary-domain>" }
$resolvedSanNames = @()

if ($SubjectAlternativeNames -and $SubjectAlternativeNames.Count -gt 0) {
    $resolvedSanNames = @($SubjectAlternativeNames | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
} else {
    $mapSan = Get-MapValue -MapValues $mapValues -Key "<your-www-domain>"
    if ($mapSan) {
        $resolvedSanNames = @($mapSan)
    }
}

$resolvedRegion = if ($Region) {
    $Region
} elseif ($env:AWS_REGION) {
    $env:AWS_REGION
} elseif ($env:AWS_DEFAULT_REGION) {
    $env:AWS_DEFAULT_REGION
} else {
    Get-MapValue -MapValues $mapValues -Key "<your-aws-region>"
}

$templateContent = Get-Content $resolvedTemplate -Raw

if ($resolvedDomainName) {
    $templateContent = $templateContent.Replace("<your-primary-domain>", $resolvedDomainName)
}
if ($resolvedSanNames.Count -gt 0) {
    $templateContent = $templateContent.Replace("<your-www-domain>", $resolvedSanNames[0])
}

Assert-NoUnresolvedPlaceholders -Content $templateContent

$renderedObject = $templateContent | ConvertFrom-Json

if (-not $resolvedDomainName) {
    throw "Missing primary domain. Pass -DomainName or set <your-primary-domain> in $resolvedMap"
}

if ($resolvedSanNames.Count -eq 0) {
    $renderedObject.PSObject.Properties.Remove("SubjectAlternativeNames")
} else {
    $renderedObject.SubjectAlternativeNames = @($resolvedSanNames)
}

$outputDir = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$renderedJson = $renderedObject | ConvertTo-Json -Depth 10
$renderedJson | Set-Content -Path $resolvedOutput -Encoding UTF8

Write-Host "Rendered ACM certificate request to: $resolvedOutput"

if (-not $Apply) {
    Write-Host "Render only mode. Use -Apply to call aws acm request-certificate."
    return
}

if (-not $resolvedRegion) {
    throw "Missing AWS region. Pass -Region, set AWS_REGION/AWS_DEFAULT_REGION, or set <your-aws-region> in $resolvedMap"
}

$idempotencyToken = Get-IdempotencyToken -PrimaryDomain $resolvedDomainName -AlternativeNames $resolvedSanNames
$outputUri = "file://$resolvedOutput"

$requestRaw = aws acm request-certificate --region $resolvedRegion --cli-input-json $outputUri --idempotency-token $idempotencyToken --output json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to request ACM certificate: $requestRaw"
}

$requestResult = $requestRaw | ConvertFrom-Json
$certificateArn = $requestResult.CertificateArn

Write-Host "Requested ACM certificate: $certificateArn"

if (-not $WaitForDnsValidationRecords) {
    Write-Host "Run with -WaitForDnsValidationRecords to poll ACM and print the required validation CNAMEs."
    return
}

$recordsAvailable = $false

for ($attempt = 1; $attempt -le $MaxPollAttempts; $attempt++) {
    $describeRaw = aws acm describe-certificate --region $resolvedRegion --certificate-arn $certificateArn --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to describe ACM certificate: $describeRaw"
    }

    $certificate = ($describeRaw | ConvertFrom-Json).Certificate
    Write-Host "Attempt $attempt/$MaxPollAttempts - certificate status: $($certificate.Status)"

    $recordsAvailable = Write-ValidationRecords -Certificate $certificate
    if ($recordsAvailable) {
        break
    }

    if ($attempt -lt $MaxPollAttempts) {
        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

if (-not $recordsAvailable) {
    Write-Host "ACM request was created, but validation CNAMEs did not appear within the polling window."
    Write-Host "Re-run describe-certificate later using ARN: $certificateArn"
}