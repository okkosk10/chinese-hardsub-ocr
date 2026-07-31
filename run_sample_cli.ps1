$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# 아래 영상 경로와 crop(x,y,width,height)을 실제 값으로 수정하세요.
$Video = "D:\videos\sample.mp4"
$Crop = "400,700,1120,180"
& .\.venv\Scripts\python.exe -m hardsub_ocr.cli `
  --input $Video --start "00:00:00" --end "00:01:00" `
  --crop $Crop --interval 0.5 --output-dir ".\output"

