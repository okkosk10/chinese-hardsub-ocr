$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$version = & py -3.11 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
if (-not $version) { throw "Python 3.11을 찾을 수 없습니다. python.org에서 3.11을 설치하세요." }
Write-Host "Python $version"

if (-not (Test-Path .venv)) { & py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Warning "FFmpeg/ffprobe가 PATH에 없습니다. README의 설치 안내를 따르세요."
} else {
    ffmpeg -version | Select-Object -First 1
    ffprobe -version | Select-Object -First 1
}
Write-Host "테스트: .\.venv\Scripts\python.exe -m pytest"
Write-Host "GUI:    .\run_gui.ps1"

