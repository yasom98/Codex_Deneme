[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoPath,

    [Parameter(Mandatory = $true)]
    [string]$CutoffUtc,

    [Parameter()]
    [string]$Symbol = "BTCUSDT",

    [Parameter()]
    [string]$AcceptedRunId = "20260215T230000Z_clean15m",

    [Parameter()]
    [string]$RefreshSessionId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ"),

    [Parameter()]
    [string]$DataConfigTemplatePath = "",

    [Parameter()]
    [string]$FeaturesConfigPath = "",

    [Parameter()]
    [string]$PythonExe = "python",

    [Parameter()]
    [int]$RequestLimit = 1000,

    [Parameter()]
    [int]$HistoricalMaxCandlesPerTimeframe = 0,

    [Parameter()]
    [int]$MaxRetries = 2,

    [Parameter()]
    [double]$RetryBackoffSeconds = 1.0,

    [Parameter()]
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Format-CommandArgument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        throw "$Label is required."
    }
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Assert-PositiveInt {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Value,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if ($Value -le 0) {
        throw "$Label must be > 0."
    }
}

function Assert-NonNegativeInt {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Value,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if ($Value -lt 0) {
        throw "$Label must be >= 0."
    }
}

if ($Symbol -ne "BTCUSDT") {
    throw "This wrapper persists the proven BTCUSDT route only. Current repo lineage anchors and symbol normalization remain BTC-specific."
}

Assert-PositiveInt -Value $RequestLimit -Label "RequestLimit"
Assert-NonNegativeInt -Value $HistoricalMaxCandlesPerTimeframe -Label "HistoricalMaxCandlesPerTimeframe"
Assert-PositiveInt -Value $MaxRetries -Label "MaxRetries"

try {
    $cutoff = [DateTimeOffset]::Parse($CutoffUtc, [System.Globalization.CultureInfo]::InvariantCulture)
}
catch {
    throw "CutoffUtc is not a valid timestamp: $CutoffUtc"
}

if ($cutoff.Offset -ne [TimeSpan]::Zero) {
    throw "CutoffUtc must be explicit UTC."
}

$normalizedCutoffUtc = $cutoff.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:sszzz")

$resolvedRepoPath = Resolve-RequiredPath -PathValue $RepoPath -Label "RepoPath"
$refreshScriptPath = Resolve-RequiredPath -PathValue (Join-Path $resolvedRepoPath "scripts\\refresh_market_data_tail.py") -Label "refresh_market_data_tail.py"

if ([string]::IsNullOrWhiteSpace($DataConfigTemplatePath)) {
    $DataConfigTemplatePath = Join-Path $resolvedRepoPath "configs\\data.yaml"
}
if ([string]::IsNullOrWhiteSpace($FeaturesConfigPath)) {
    $FeaturesConfigPath = Join-Path $resolvedRepoPath "configs\\features.yaml"
}

$resolvedDataConfigTemplatePath = Resolve-RequiredPath -PathValue $DataConfigTemplatePath -Label "DataConfigTemplatePath"
$resolvedFeaturesConfigPath = Resolve-RequiredPath -PathValue $FeaturesConfigPath -Label "FeaturesConfigPath"

foreach ($legacyName in @(
    "BTC_USDT_1m_price_data.csv",
    "BTC_USDT_5m_price_data.csv",
    "BTC_USDT_15m_price_data.csv"
)) {
    $legacyPath = Join-Path $resolvedRepoPath $legacyName
    if (-not (Test-Path -LiteralPath $legacyPath)) {
        throw "Missing immutable legacy reference file: $legacyPath"
    }
}

$pythonCommand = Get-Command $PythonExe -ErrorAction Stop
$pythonSource = if ($pythonCommand.Source) { $pythonCommand.Source } else { $pythonCommand.Path }
if (-not $pythonSource) {
    throw "Unable to resolve Python executable route for: $PythonExe"
}

$tempConfigName = "data.windows.binance_gapfill.$RefreshSessionId.yaml"
$tempConfigPath = Join-Path (Split-Path -Parent $resolvedDataConfigTemplatePath) $tempConfigName
$tempConfigCreated = $false

Write-Host "Repo path: $resolvedRepoPath"
Write-Host "Python executable route: $pythonSource"
Write-Host "Accepted run id: $AcceptedRunId"
Write-Host "Refresh session id: $RefreshSessionId"
Write-Host "Symbol: $Symbol"
Write-Host "Explicit cutoff UTC: $normalizedCutoffUtc"
Write-Host "Request limit: $RequestLimit"

try {
    $configContent = Get-Content -LiteralPath $resolvedDataConfigTemplatePath -Raw
    $configContent = [System.Text.RegularExpressions.Regex]::Replace(
        $configContent,
        '(?m)^input_root:.*$',
        'input_root: ..'
    )
    Set-Content -LiteralPath $tempConfigPath -Value $configContent -Encoding UTF8
    $tempConfigCreated = $true

    $effectiveArgs = @(
        $refreshScriptPath,
        "--mode", "separate_binance_perpetual_backfill",
        "--accepted-run-id", $AcceptedRunId,
        "--refresh-session-id", $RefreshSessionId,
        "--legacy-input-root", $resolvedRepoPath,
        "--data-config", $tempConfigPath,
        "--features-config", $resolvedFeaturesConfigPath,
        "--request-limit", "$RequestLimit",
        "--historical-max-candles-per-timeframe", "$HistoricalMaxCandlesPerTimeframe",
        "--target-end-utc", $normalizedCutoffUtc,
        "--max-retries", "$MaxRetries",
        "--retry-backoff-seconds", "$RetryBackoffSeconds",
        "--log-level", $LogLevel
    )

    $displayCommand = @($pythonSource) + ($effectiveArgs | ForEach-Object { Format-CommandArgument -Value $_ })
    Write-Host ""
    Write-Host "Effective command:"
    Write-Host ($displayCommand -join " ")
    Write-Host ""

    & $pythonSource @effectiveArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Gap-fill command failed with exit code $exitCode."
    }
}
finally {
    if ($tempConfigCreated -and (Test-Path -LiteralPath $tempConfigPath)) {
        Remove-Item -LiteralPath $tempConfigPath -Force
    }
}
