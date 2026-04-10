param(
    [string]$BackendEnvFile = "env.from_ec2_backend.env",
    [string]$RootEnvFile = "env_from_ec2.env",
    [string]$MapFile = "config/aws_deployment.local.map",
    [string]$TemplateDir = "config/k8s/backend",
    [string]$OutputDir = "config/k8s/backend/rendered-local"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedBackendEnv = Join-Path $repoRoot $BackendEnvFile
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

if (-not (Test-Path $resolvedBackendEnv)) {
    throw "Backend env file not found: $resolvedBackendEnv"
}

if (-not (Test-Path $resolvedRootEnv)) {
    throw "Root env file not found: $resolvedRootEnv"
}

if (-not (Test-Path $resolvedMap)) {
    throw "Map file not found: $resolvedMap"
}

$backendEnv = Read-KeyValueFile -Path $resolvedBackendEnv
$rootEnv = Read-KeyValueFile -Path $resolvedRootEnv
$mapValues = Read-KeyValueFile -Path $resolvedMap

$rdsHost = $mapValues["<your-rds-endpoint>"]
$rdsPort = $mapValues["<your-rds-port>"]

if (-not $rdsHost) {
    throw "Missing <your-rds-endpoint> in $resolvedMap"
}

if (-not $rdsPort) {
    $rdsPort = "5432"
}

$databaseUrl = $backendEnv["DATABASE_URL"] -replace "@localhost:5433", "@$rdsHost`:$rdsPort"
$checkpointUrl = $backendEnv["CHECKPOINT_POSTGRES_DATABASE_URL"] -replace "@localhost:5433", "@$rdsHost`:$rdsPort"

$replacements = @{
    "__BACKEND_IMAGE__" = "$($mapValues["<your-ecr-backend-repo>"]):latest"
    "__ENV__" = $backendEnv["ENV"]
    "__TESTING__" = $backendEnv["TESTING"]
    "__DEBUG__" = $backendEnv["DEBUG"]
    "__LOG_LEVEL__" = $backendEnv["LOG_LEVEL"]
    "__LOG_FORMAT__" = $backendEnv["LOG_FORMAT"]
    "__LOG_PATH__" = $backendEnv["LOG_PATH"]
    "__SHOW_SCORES__" = $backendEnv["SHOW_SCORES"]
    "__CORS_ORIGINS__" = $backendEnv["CORS_ORIGINS"]
    "__OPENAI_MODEL__" = $backendEnv["OPENAI_MODEL"]
    "__OPENAI_SIMPLE_MODEL__" = $(if ($backendEnv.ContainsKey("OPENAI_SIMPLE_MODEL")) { $backendEnv["OPENAI_SIMPLE_MODEL"] } else { $backendEnv["OPENAI_MODEL"] })
    "__OPENAI_VISION_MODEL__" = $(if ($backendEnv.ContainsKey("OPENAI_VISION_MODEL")) { $backendEnv["OPENAI_VISION_MODEL"] } else { $backendEnv["OPENAI_MODEL"] })
    "__CHECKPOINT_ENABLED__" = $backendEnv["CHECKPOINT_ENABLED"]
    "__LANGCHAIN_TRACING_V2__" = $backendEnv["LANGCHAIN_TRACING_V2"]
    "__LANGCHAIN_PROJECT__" = $backendEnv["LANGCHAIN_PROJECT"]
    "__LANGCHAIN_ENDPOINT__" = $backendEnv["LANGCHAIN_ENDPOINT"]
    "__EMBEDDING_PROVIDER__" = $backendEnv["EMBEDDING_PROVIDER"]
    "__OPENAI_EMBEDDING_MODEL__" = $backendEnv["OPENAI_EMBEDDING_MODEL"]
    "__EMBEDDING_MODEL_NAME__" = $backendEnv["EMBEDDING_MODEL_NAME"]
    "__RERANKER_PROVIDER__" = $backendEnv["RERANKER_PROVIDER"]
    "__RERANKER_MODEL_NAME__" = $backendEnv["RERANKER_MODEL_NAME"]
    "__COHERE_RERANK_MODEL__" = $backendEnv["COHERE_RERANK_MODEL"]
    "__VECTOR_DB_PROVIDER__" = $backendEnv["VECTOR_DB_PROVIDER"]
    "__QDRANT_COLLECTION_NAME__" = $backendEnv["QDRANT_COLLECTION_NAME"]
    "__QDRANT_PREFER_GRPC__" = $backendEnv["QDRANT_PREFER_GRPC"]
    "__QDRANT_SPARSE_MODEL__" = $backendEnv["QDRANT_SPARSE_MODEL"]
    "__DATA_DIR__" = $backendEnv["DATA_DIR"]
    "__CONVERSATION_MESSAGE_LIMIT__" = $backendEnv["CONVERSATION_MESSAGE_LIMIT"]
    "__REDIS_CACHE_TTL__" = $backendEnv["REDIS_CACHE_TTL"]
    "__REDIS_CACHE_LIMIT__" = $backendEnv["REDIS_CACHE_LIMIT"]
    "__USE_HYBRID__" = $backendEnv["USE_HYBRID"]
    "__USE_RERANKER__" = $backendEnv["USE_RERANKER"]
    "__USE_SEMANTIC_ROUTER__" = $backendEnv["USE_SEMANTIC_ROUTER"]
    "__USE_SELF_REFLECTION__" = $backendEnv["USE_SELF_REFLECTION"]
    "__TOP_K__" = $backendEnv["TOP_K"]
    "__CANDIDATES__" = $backendEnv["CANDIDATES"]
    "__CHAT_MAX_TOKENS__" = $backendEnv["CHAT_MAX_TOKENS"]
    "__FORCE_INTERNAL_RETRIEVAL__" = $backendEnv["FORCE_INTERNAL_RETRIEVAL"]
    "__ENFORCE_CITATIONS__" = $backendEnv["ENFORCE_CITATIONS"]
    "__FUSION_METHOD__" = $backendEnv["FUSION_METHOD"]
    "__FUSE_ALPHA__" = $backendEnv["FUSE_ALPHA"]
    "__RRF_K__" = $backendEnv["RRF_K"]
    "__MIN_HYBRID__" = $backendEnv["MIN_HYBRID"]
    "__AVG_HYBRID__" = $backendEnv["AVG_HYBRID"]
    "__MIN_SEM_SIM__" = $backendEnv["MIN_SEM_SIM"]
    "__AVG_SEM_SIM__" = $backendEnv["AVG_SEM_SIM"]
    "__MIN_RERANK__" = $backendEnv["MIN_RERANK"]
    "__AVG_RERANK__" = $backendEnv["AVG_RERANK"]
    "__RERANKER_THRESHOLD_RELAXATION__" = $backendEnv["RERANKER_THRESHOLD_RELAXATION"]
    "__MIN_RAW_BM25__" = $backendEnv["MIN_RAW_BM25"]
    "__DECOMPOSITION_ENABLED__" = $backendEnv["DECOMPOSITION_ENABLED"]
    "__DECOMPOSITION_MAX_WORKERS__" = $backendEnv["DECOMPOSITION_MAX_WORKERS"]
    "__CHUNK_SIZE__" = $backendEnv["CHUNK_SIZE"]
    "__CHUNK_OVERLAP__" = $backendEnv["CHUNK_OVERLAP"]
    "__CHUNKING_STRATEGY__" = $backendEnv["CHUNKING_STRATEGY"]
    "__TIKTOKEN_ENCODING__" = $backendEnv["TIKTOKEN_ENCODING"]
    "__CONTEXTUAL_RETRIEVAL_ENABLED__" = $backendEnv["CONTEXTUAL_RETRIEVAL_ENABLED"]
    "__CONTEXTUAL_RETRIEVAL_MODEL__" = $backendEnv["CONTEXTUAL_RETRIEVAL_MODEL"]
    "__CONTEXTUAL_RETRIEVAL_MAX_WORKERS__" = $backendEnv["CONTEXTUAL_RETRIEVAL_MAX_WORKERS"]
    "__TEXT_MAX__" = $backendEnv["TEXT_MAX"]
    "__UPSERT_BATCH_SIZE__" = $backendEnv["UPSERT_BATCH_SIZE"]
    "__UPLOAD_BASE__" = $backendEnv["UPLOAD_BASE"]
    "__MAX_UPLOAD_MB__" = $backendEnv["MAX_UPLOAD_MB"]
    "__ALLOWED_EXTENSIONS__" = $backendEnv["ALLOWED_EXTENSIONS"]
    "__MIME_TYPES__" = $backendEnv["MIME_TYPES"]
    "__FOLDER_SHARED__" = $backendEnv["FOLDER_SHARED"]
    "__DEPT_SPLIT__" = $backendEnv["DEPT_SPLIT"]
    "__DOWNLOAD_BASE__" = $backendEnv["DOWNLOAD_BASE"]
    "__MAX_DOWNLOAD_SIZE_MB__" = $backendEnv["MAX_DOWNLOAD_SIZE_MB"]
    "__DOWNLOAD_TIMEOUT__" = $backendEnv["DOWNLOAD_TIMEOUT"]
    "__SERVICE_AUTH_ISSUER__" = $backendEnv["SERVICE_AUTH_ISSUER"]
    "__SERVICE_AUTH_AUDIENCE__" = $backendEnv["SERVICE_AUTH_AUDIENCE"]
    "__ORG_STRUCTURE_FILE__" = $backendEnv["ORG_STRUCTURE_FILE"]
    "__RATELIMIT_STORAGE_URI__" = $backendEnv["RATELIMIT_STORAGE_URI"]
    "__DEFAULT_RATE_LIMITS__" = $backendEnv["DEFAULT_RATE_LIMITS"]
    "__MCP_SERVER_COMMAND__" = $backendEnv["MCP_SERVER_COMMAND"]
    "__WEB_SEARCH_ENABLED__" = $backendEnv["WEB_SEARCH_ENABLED"]
    "__WEB_SEARCH_PROVIDER__" = $backendEnv["WEB_SEARCH_PROVIDER"]
    "__WEB_SEARCH_MAX_RESULTS__" = $backendEnv["WEB_SEARCH_MAX_RESULTS"]
    "__BROWSER_USE_ENABLED__" = $backendEnv["BROWSER_USE_ENABLED"]
    "__BROWSER_HEADLESS__" = $backendEnv["BROWSER_HEADLESS"]
    "__BROWSER_TIMEOUT__" = $backendEnv["BROWSER_TIMEOUT"]
    "__BROWSER_MAX_STEPS__" = $backendEnv["BROWSER_MAX_STEPS"]
    "__BROWSER_LOG_PATH__" = $backendEnv["BROWSER_LOG_PATH"]
    "__SLACK_ENABLED__" = $backendEnv["SLACK_ENABLED"]
    "__REFLECTION_MODE__" = $backendEnv["REFLECTION_MODE"]
    "__REFLECTION_THRESHOLD_EXCELLENT__" = $backendEnv["REFLECTION_THRESHOLD_EXCELLENT"]
    "__REFLECTION_THRESHOLD_GOOD__" = $backendEnv["REFLECTION_THRESHOLD_GOOD"]
    "__REFLECTION_THRESHOLD_PARTIAL__" = $backendEnv["REFLECTION_THRESHOLD_PARTIAL"]
    "__REFLECTION_MIN_CONTEXTS__" = $backendEnv["REFLECTION_MIN_CONTEXTS"]
    "__REFLECTION_AUTO_REFINE__" = $backendEnv["REFLECTION_AUTO_REFINE"]
    "__REFLECTION_MAX_REFINEMENT_ATTEMPTS__" = $backendEnv["REFLECTION_MAX_REFINEMENT_ATTEMPTS"]
    "__OTEL_ENABLED__" = $backendEnv["OTEL_ENABLED"]
    "__OTEL_SERVICE_NAME__" = $backendEnv["OTEL_SERVICE_NAME"]
    "__OTEL_SERVICE_NAMESPACE__" = $backendEnv["OTEL_SERVICE_NAMESPACE"]
    "__OTEL_SERVICE_VERSION__" = $backendEnv["OTEL_SERVICE_VERSION"]
    "__OTEL_DEPLOYMENT_ENVIRONMENT__" = $backendEnv["OTEL_DEPLOYMENT_ENVIRONMENT"]
    "__OPENAI_API_KEY__" = $backendEnv["OPENAI_API_KEY"]
    "__LANGCHAIN_API_KEY__" = $backendEnv["LANGCHAIN_API_KEY"]
    "__COHERE_API_KEY__" = $backendEnv["COHERE_API_KEY"]
    "__BRAVE_API_KEY__" = $backendEnv["BRAVE_API_KEY"]
    "__E2B_API_KEY__" = $backendEnv["E2B_API_KEY"]
    "__TAVILY_API_KEY__" = $backendEnv["TAVILY_API_KEY"]
    "__DATABASE_URL__" = $databaseUrl
    "__CHECKPOINT_POSTGRES_DATABASE_URL__" = $checkpointUrl
    "__REDIS_URL__" = $rootEnv["REDIS_URL"]
    "__SERVICE_AUTH_SECRET__" = $backendEnv["SERVICE_AUTH_SECRET"]
    "__SLACK_SIGNING_SECRET__" = $backendEnv["SLACK_SIGNING_SECRET"]
    "__SMTP_SERVER__" = $backendEnv["SMTP_SERVER"]
    "__SMTP_PORT__" = $backendEnv["SMTP_PORT"]
    "__SMTP_USER__" = $backendEnv["SMTP_USER"]
    "__SMTP_PASSWORD__" = $backendEnv["SMTP_PASSWORD"]
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
