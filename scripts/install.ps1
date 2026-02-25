param(
    [string]$Version = $env:PIXY_VERSION,
    [string]$InstallDir = $env:PIXY_INSTALL_DIR,
    [string]$Repo = $env:PIXY_REPO,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
pixy installer (Windows)

Usage:
  install.ps1 [-Version <tag>] [-InstallDir <dir>] [-Repo <owner/repo>] [-Help]

Defaults:
  Version: latest
  InstallDir: $env:USERPROFILE\\.local\\bin
  Repo: sundy-li/pixy

Examples:
  ./scripts/install.ps1
  ./scripts/install.ps1 -Version v0.1.0
  ./scripts/install.ps1 -InstallDir "$env:USERPROFILE\\bin"
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = "latest"
}
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:USERPROFILE ".local\\bin"
}
if ([string]::IsNullOrWhiteSpace($Repo)) {
    $Repo = "sundy-li/pixy"
}

$archRaw = $env:PROCESSOR_ARCHITECTURE
if ([string]::IsNullOrWhiteSpace($archRaw)) {
    throw "Cannot detect PROCESSOR_ARCHITECTURE"
}

$arch = switch ($archRaw.ToLowerInvariant()) {
    "amd64" { "x86_64" }
    "x86_64" { "x86_64" }
    "arm64" { "aarch64" }
    default { throw "Unsupported architecture: $archRaw" }
}

if ($arch -ne "x86_64") {
    throw "Windows release currently supports x86_64 only. Detected: $arch"
}

$target = "$arch-pc-windows-msvc"
$asset = "pixy-$target.zip"

if ($Version -eq "latest") {
    $baseUrl = "https://github.com/$Repo/releases/latest/download"
    $displayVersion = "latest"
} else {
    if (-not $Version.StartsWith("v")) {
        $Version = "v$Version"
    }
    $baseUrl = "https://github.com/$Repo/releases/download/$Version"
    $displayVersion = $Version
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("pixy-install-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
    $assetPath = Join-Path $tmp $asset
    $sumsPath = Join-Path $tmp "SHA256SUMS"

    Write-Host "==> Downloading $asset ($displayVersion)"
    Invoke-WebRequest -Uri "$baseUrl/$asset" -OutFile $assetPath

    Write-Host "==> Downloading SHA256SUMS"
    Invoke-WebRequest -Uri "$baseUrl/SHA256SUMS" -OutFile $sumsPath

    $sumLine = Get-Content $sumsPath | Where-Object {
        $_ -match ("^([a-fA-F0-9]{64})\s+(\./)?" + [regex]::Escape($asset) + "$")
    } | Select-Object -First 1

    if (-not $sumLine) {
        throw "Checksum entry not found for $asset"
    }

    if ($sumLine -notmatch "^([a-fA-F0-9]{64})\s+") {
        throw "Invalid checksum line format for $asset"
    }
    $expected = $matches[1].ToLowerInvariant()
    $actual = (Get-FileHash -Path $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()

    if ($expected -ne $actual) {
        throw "Checksum mismatch for $asset. Expected: $expected, Actual: $actual"
    }

    Write-Host "==> Checksum verified"

    Expand-Archive -Path $assetPath -DestinationPath $tmp -Force

    $binPath = Join-Path $tmp "pixy.exe"
    if (-not (Test-Path $binPath)) {
        throw "Archive does not contain pixy.exe"
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $dest = Join-Path $InstallDir "pixy.exe"
    Copy-Item -Path $binPath -Destination $dest -Force

    Write-Host "==> pixy installed to $dest"

    $pathParts = @()
    if (-not [string]::IsNullOrWhiteSpace($env:PATH)) {
        $pathParts = $env:PATH.Split(';')
    }
    if (-not ($pathParts -contains $InstallDir)) {
        Write-Host "==> Add to PATH (PowerShell):"
        Write-Host "    `$env:Path = \"$InstallDir;`$env:Path\""
    }

    Write-Host "==> Verify: pixy --help"
}
finally {
    Remove-Item -Path $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
