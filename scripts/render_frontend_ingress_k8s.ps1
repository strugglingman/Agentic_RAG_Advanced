param(
    [string]$RootEnvFile = "env_from_ec2.env",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$TemplateFile = "config/k8s/frontend/04-ingress.template.yaml",
    [string]$OutputFile = "config/k8s/frontend/rendered-local/04-ingress.from_local.yaml"
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

function Assert-NoUnresolvedPlaceholders {
    param(
        [string]$Content,
        [string]$TemplateFileName
    )

    $matches = [regex]::Matches($Content, "__[A-Z0-9_]+__")
    if ($matches.Count -eq 0) {
        return
    }

    $remaining = $matches | ForEach-Object { $_.Value } | Select-Object -Unique
    throw "Unresolved placeholders remain in ${TemplateFileName}: $($remaining -join ', ')"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRootEnv = Resolve-RepoPath -RepoRoot $repoRoot -Path $RootEnvFile
$resolvedMap = Resolve-RepoPath -RepoRoot $repoRoot -Path $MapFile
$resolvedTemplate = Resolve-RepoPath -RepoRoot $repoRoot -Path $TemplateFile
$resolvedOutput = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputFile

if (-not (Test-Path $resolvedMap)) {
    throw "Map file not found: $resolvedMap"
}

if (-not (Test-Path $resolvedTemplate)) {
    throw "Ingress template file not found: $resolvedTemplate"
}

$rootEnv = @{}
if (Test-Path $resolvedRootEnv) {
    $rootEnv = Read-KeyValueFile -Path $resolvedRootEnv
}

$mapValues = Read-KeyValueFile -Path $resolvedMap

$primaryDomain = if ($rootEnv.ContainsKey("FRONTEND_PRIMARY_DOMAIN") -and $rootEnv["FRONTEND_PRIMARY_DOMAIN"]) {
    $rootEnv["FRONTEND_PRIMARY_DOMAIN"]
} else {
    $mapValues["<your-primary-domain>"]
}

$wwwDomain = if ($rootEnv.ContainsKey("FRONTEND_WWW_DOMAIN") -and $rootEnv["FRONTEND_WWW_DOMAIN"]) {
    $rootEnv["FRONTEND_WWW_DOMAIN"]
} else {
    $mapValues["<your-www-domain>"]
}

$certificateArn = if ($rootEnv.ContainsKey("FRONTEND_ACM_CERTIFICATE_ARN") -and $rootEnv["FRONTEND_ACM_CERTIFICATE_ARN"]) {
    $rootEnv["FRONTEND_ACM_CERTIFICATE_ARN"]
} else {
    $mapValues["<your-acm-certificate-arn>"]
}

if (-not $primaryDomain) {
    throw "Missing primary domain. Set FRONTEND_PRIMARY_DOMAIN or <your-primary-domain> in $resolvedMap"
}
if (-not $wwwDomain) {
    throw "Missing www domain. Set FRONTEND_WWW_DOMAIN or <your-www-domain> in $resolvedMap"
}
if (-not $certificateArn) {
    throw "Missing ACM certificate ARN. Set FRONTEND_ACM_CERTIFICATE_ARN or <your-acm-certificate-arn> in $resolvedMap"
}

$content = Get-Content $resolvedTemplate -Raw
$content = $content.Replace("__PRIMARY_DOMAIN__", $primaryDomain)
$content = $content.Replace("__WWW_DOMAIN__", $wwwDomain)
$content = $content.Replace("__ACM_CERTIFICATE_ARN__", $certificateArn)

Assert-NoUnresolvedPlaceholders -Content $content -TemplateFileName (Split-Path -Leaf $resolvedTemplate)

$outputDir = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Set-Content -Path $resolvedOutput -Value $content -Encoding UTF8
Write-Host "Rendered frontend ingress manifest to: $resolvedOutput"