$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw "먼저 .\setup.ps1을 실행하세요." }
& .\.venv\Scripts\python.exe -m hardsub_ocr.app

