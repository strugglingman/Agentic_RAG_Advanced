param(
    [string]$RootEnvFile = "env_from_ec2.env",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$TemplateDir = "config/k8s/frontend",
    [string]$OutputDir = "config/k8s/frontend/rendered-local"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRootEnv = Join-Path $repoRoot $RootEnvFile
$resolvedMap = Join-Path $repoRoot $MapFile
$resolvedTemplateDir = Join-Path $repoRoot $TemplateDir
$resolvedOutputDir = Join-Path $repoRoot $OutputDir

function Read-KeyValueFile {
    param([string]$Path)

    $result = @{}
    foreach ($line in Get-Content $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.Trim().StartsWith("#")) { continue }
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) { continue }
        $result[$parts[0].Trim()] = $parts[1]
    }
    return $result
}

function Escape-YamlDoubleQuoted {
    param([AllowNull()][string]$Value)

    if ($null -eq $Value) {
        return ""
    }

    return $Value.Replace('\', '\\').Replace('"', '\"').Replace("`r", '').Replace("`n", '\n')
}

if (-not (Test-Path $resolvedRootEnv)) {
    throw "Root env file not found: $resolvedRootEnv"
}

if (-not (Test-Path $resolvedMap)) {
    throw "Map file not found: $resolvedMap"
}

$rootEnv = Read-KeyValueFile -Path $resolvedRootEnv
$mapValues = Read-KeyValueFile -Path $resolvedMap

$frontendImage = "$($mapValues["<your-ecr-frontend-repo>"]):latest"
if (-not $frontendImage -or $frontendImage -eq ":latest") {
    throw "Missing <your-ecr-frontend-repo> in $resolvedMap"
}

$postgresUser = $rootEnv["POSTGRES_USER"]
$postgresPassword = $rootEnv["POSTGRES_PASSWORD"]
$postgresDb = $rootEnv["POSTGRES_DB"]
$postgresHost = $rootEnv["POSTGRES_HOST"]
$postgresPort = $rootEnv["POSTGRES_PORT"]

if (-not $postgresUser) { throw "Missing POSTGRES_USER in $resolvedRootEnv" }
if (-not $postgresPassword) { throw "Missing POSTGRES_PASSWORD in $resolvedRootEnv" }
if (-not $postgresDb) { throw "Missing POSTGRES_DB in $resolvedRootEnv" }
if (-not $postgresHost) { throw "Missing POSTGRES_HOST in $resolvedRootEnv" }
if (-not $postgresPort) { $postgresPort = "5432" }

$encodedPostgresUser = [System.Uri]::EscapeDataString($postgresUser)
$encodedPostgresPassword = [System.Uri]::EscapeDataString($postgresPassword)
$databaseUrl = "postgresql://{0}:{1}@{2}:{3}/{4}?schema=chatbot" -f $encodedPostgresUser, $encodedPostgresPassword, $postgresHost, $postgresPort, $postgresDb

$nextAuthUrl = $(if ($rootEnv.ContainsKey("FRONTEND_PUBLIC_BASE_URL") -and $rootEnv["FRONTEND_PUBLIC_BASE_URL"]) {
    $rootEnv["FRONTEND_PUBLIC_BASE_URL"]
} elseif ($mapValues.ContainsKey("<your-frontend-public-base-url>") -and $mapValues["<your-frontend-public-base-url>"]) {
    $mapValues["<your-frontend-public-base-url>"]
} else {
    "http://127.0.0.1:3000"
})
$nextAuthSecret = $rootEnv["NEXTAUTH_SECRET"]
$serviceAuthSecret = $rootEnv["SERVICE_AUTH_SECRET"]
$serviceAuthIssuer = $rootEnv["SERVICE_AUTH_ISSUER"]
$serviceAuthAudience = $rootEnv["SERVICE_AUTH_AUDIENCE"]
$nextPublicUploadLimit = $(if ($rootEnv.ContainsKey("NEXT_PUBLIC_UPLOAD_FILE_LIMIT_MB") -and $rootEnv["NEXT_PUBLIC_UPLOAD_FILE_LIMIT_MB"]) { $rootEnv["NEXT_PUBLIC_UPLOAD_FILE_LIMIT_MB"] } else { "25" })

if (-not $nextAuthSecret) { throw "Missing NEXTAUTH_SECRET in $resolvedRootEnv" }
if (-not $serviceAuthSecret) { throw "Missing SERVICE_AUTH_SECRET in $resolvedRootEnv" }
if (-not $serviceAuthIssuer) { throw "Missing SERVICE_AUTH_ISSUER in $resolvedRootEnv" }
if (-not $serviceAuthAudience) { throw "Missing SERVICE_AUTH_AUDIENCE in $resolvedRootEnv" }

$replacements = @{
    "__FRONTEND_IMAGE__" = $frontendImage
    "__NEXTAUTH_URL__" = $nextAuthUrl
    "__SERVICE_AUTH_ISSUER__" = $serviceAuthIssuer
    "__SERVICE_AUTH_AUDIENCE__" = $serviceAuthAudience
    "__NEXT_PUBLIC_UPLOAD_FILE_LIMIT_MB__" = $nextPublicUploadLimit
    "__NEXTAUTH_SECRET__" = $nextAuthSecret
    "__DATABASE_URL__" = $databaseUrl
    "__SERVICE_AUTH_SECRET__" = $serviceAuthSecret
}

New-Item -ItemType Directory -Force -Path $resolvedOutputDir | Out-Null

$templateFiles = @(
    "00-configmap.template.yaml",
    "01-secret.template.yaml",
    "02-service.yaml",
    "03-deployment.template.yaml"
)

foreach ($templateFile in $templateFiles) {
    $sourcePath = Join-Path $resolvedTemplateDir $templateFile
    $content = Get-Content $sourcePath -Raw

    foreach ($key in $replacements.Keys) {
        $value = $replacements[$key]
        if ($null -eq $value) {
            $value = ""
        }
        $content = $content.Replace($key, (Escape-YamlDoubleQuoted -Value ([string]$value)))
    }

    if ($templateFile -like "*.template.yaml") {
        $outputName = $templateFile.Replace(".template.yaml", ".from_local.yaml")
    } else {
        $outputName = $templateFile.Replace(".yaml", ".from_local.yaml")
    }
    $outputPath = Join-Path $resolvedOutputDir $outputName
    Set-Content -Path $outputPath -Value $content -Encoding UTF8
    Write-Host "Rendered $outputPath"
}
