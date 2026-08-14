# Builds the Cholesky Decomposition Simulator with g++.
# Usage: powershell -File build.ps1

$ErrorActionPreference = "Stop"

$gpp = Get-Command g++ -ErrorAction SilentlyContinue
if (-not $gpp) {
    Write-Error "g++ not found on PATH. Install a MinGW-w64 toolchain (e.g. 'winget install BrechtSanders.WinLibs.POSIX.UCRT') and re-run."
}

$root = $PSScriptRoot
$sources = @(
    (Join-Path $root "backend\Matrix.cpp"),
    (Join-Path $root "backend\Cholesky.cpp"),
    (Join-Path $root "frontend\main.cpp")
)
$output = Join-Path $root "cholesky_simulator.exe"

& g++ -std=c++17 -Wall -Wextra -O2 -o $output @sources

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build succeeded: $output"
} else {
    Write-Error "Build failed."
}
