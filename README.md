# 중국어 하드서브 OCR

영상 화면에 박힌 중국어 하드서브 영역을 사용자가 직접 지정하고, 지정 구간을 OCR하여 UTF-8 SRT와 상세 JSON으로 만드는 Windows용 데스크톱/CLI 도구입니다. 기존 일본어 음성 자막 제작기와는 완전히 별개인 독립 프로젝트이며, 모든 작업은 이 폴더 안에서만 수행합니다.

## 특징

- PySide6 미리보기(QMediaPlayer/QVideoWidget), 재생·일시정지·정지·타임라인
- 현재 위치를 시작/종료로 지정하거나 `HH:MM:SS[.mmm]` 직접 입력
- 레터박스/필러박스를 제외해 원본 영상 좌표로 변환하는 드래그 crop
- FFmpeg stdout rawvideo 파이프를 통한 순차 처리(프레임 PNG 대량 저장 없음)
- OpenCV 변화 감지로 불필요한 OCR 생략, OCR worker 1개, FFmpeg 스레드 기본 2개
- RapidOCR/ONNX Runtime 기반 중국어 OCR 및 교체 가능한 `OcrEngine` 인터페이스
- RapidFuzz 기반 문장 병합, 짧은 문장에 더 엄격한 임계값, 겹치지 않는 SRT
- GUI 중지 또는 CLI Ctrl+C 시 FFmpeg 종료 후 현재 SRT/JSON을 원자적으로 저장
- 30초 시험 OCR, 실시간 진행/로그/최근 영상과 crop 기억, 선택적 디버그 이미지
- 전환 직후 0.15초 안정화 후 전환 구간에만 2~3개 보조 프레임 OCR
- 여러 OCR 후보의 confidence·합의도·중국어 비율·길이 안정성 기반 선택
- 두 줄 OCR 경계의 1~6자 중복과 한 후보에만 나타난 1~3자 의심 접미 제거

## Windows 설치

1. [Python 3.11](https://www.python.org/downloads/) 64비트를 설치하고 `py -3.11 --version`으로 확인합니다.
2. FFmpeg Windows 빌드를 설치하고 `bin` 폴더를 사용자 PATH에 추가합니다. 새 PowerShell에서 `ffmpeg -version` 및 `ffprobe -version`이 모두 성공해야 합니다.
3. PowerShell에서 프로젝트 폴더로 이동해 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

직접 설치하려면 다음과 같습니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

첫 RapidOCR 실행에는 ONNX 모델 초기화로 시간이 걸릴 수 있습니다. 모델 다운로드가 필요한 패키지 버전에서는 인터넷 연결이 필요합니다.

## GUI 사용법

```powershell
.\run_gui.ps1
# 또는
.\.venv\Scripts\python.exe -m hardsub_ocr.app
```

1. 영상을 선택합니다. 미리보기 코덱 오류가 나도 FFmpeg 기반 OCR은 별도로 동작할 수 있습니다.
2. 재생/타임라인으로 이동한 뒤 현재 위치를 시작·종료로 지정하거나 시간을 직접 입력합니다.
3. 미리보기 위에서 **본편 중국어 하드서브만** 드래그합니다. 외부 광고·워터마크는 포함하지 마세요. 좌표는 원본 해상도의 `x,y,width,height`로 표시됩니다. 창 크기/DPI와 검은 여백은 변환 과정에서 보정됩니다.
4. 먼저 `30초 시험 OCR`로 결과를 확인하고, 필요하면 전처리·변화 임계값·유사도 임계값을 조정합니다.
5. `전체 구간 OCR`을 실행합니다. 중지는 현재까지의 결과를 안전하게 저장합니다.

`빠른 모드`는 전환 구간 후보 2개와 한 가지 전처리를 사용합니다. `정밀 모드`는 전환 구간 후보 3개에 대해 원본 crop과 2배 grayscale을 비교하므로 더 느리지만 전환 프레임 오인식 확인에 유리합니다. 고급 OCR 안정화 설정에서 안정화 시간, 후보 구간·개수, 후보 합의, 줄 중복 제거와 의심 접미 제거를 조정할 수 있습니다.

VLC 등에서 외부 한국어 SRT를 끄거나 화면의 다른 위치로 옮긴 뒤 OCR하세요. 한국어 자막이 crop에 겹치면 함께 인식될 수 있습니다.

## CLI

```powershell
.\.venv\Scripts\Activate.ps1
python -m hardsub_ocr.cli --help
python -m hardsub_ocr.cli `
  --input "D:\videos\ipx-193.mp4" `
  --start "00:04:00" --end "00:07:00" `
  --crop "400,700,1120,180" --interval 0.5 `
  --output-dir ".\output"
```

시간은 `HH:MM:SS`, `HH:MM:SS.mmm`, 초 숫자를 지원합니다. `--test-seconds 30`, `--change-threshold`, `--similarity-threshold`, `--ffmpeg-threads`, `--preprocess-mode`, `--save-debug-images`, `--verbose`도 사용할 수 있습니다.

OCR 안정화 옵션은 `--processing-mode fast|precise`, `--transition-settle-seconds`, `--candidate-window-seconds`, `--candidate-frame-count`, `--candidate-consensus/--no-candidate-consensus`, `--line-overlap-dedup/--no-line-overlap-dedup`, `--suspicious-suffix-removal/--no-suspicious-suffix-removal`입니다. 기존 명령은 옵션을 추가하지 않아도 새 기본값으로 그대로 동작합니다.

## 결과

입력이 `ipx-193.mp4`이면 출력 폴더에 다음이 생성됩니다.

- `ipx-193.zh-ocr.srt`: UTF-8 중국어 자막
- `ipx-193.zh-ocr.json`: 설정, 진행 상태, 세그먼트 및 프레임별 OCR 이벤트
- `ipx-193.zh-ocr.log`: CLI 실행 로그

JSON은 작업 중 주기적으로 임시 파일에 쓴 뒤 교체하므로 중단 시 손상 위험을 줄입니다. 디버그 옵션은 전환·낮은 confidence·예외 프레임만 `output/debug`에 저장합니다.

각 전환 이벤트에는 원본 OCR 줄, 중복 제거 전후 텍스트, 제거된 경계 문자열, 전체 후보와 점수, 선택·거부 후보, 합의 점수, 선택 이유, 혼합 프레임 판정과 제거한 불안정 접미가 기록됩니다. 디버그 이미지를 켜면 후보별 원본 crop과 전처리 이미지 경로도 JSON에 기록됩니다.

## 정확도와 자원 조정

- 글자가 작으면 `gray2x`(기본) 또는 `gray3x`, 윤곽이 약하면 `threshold`/`adaptive`를 시험합니다.
- 누락이 많으면 변화 임계값을 낮추고 샘플 간격을 줄입니다. 오인식 전환이 많으면 유사도 임계값을 낮춥니다.
- 자원을 줄이려면 간격을 늘리고 FFmpeg 스레드를 1로 낮추며 디버그 이미지를 끕니다. 프로세스 우선순위는 Below Normal로 설정됩니다.
- 검은 테두리·복잡한 배경 때문에 변화 감지가 과민하면 crop을 글자 주변으로 더 좁히세요.

미리보기 실패는 Windows 코덱/QMediaPlayer 문제일 수 있습니다. FFmpeg 명령 실패 시 로그에 반환 코드와 stderr가 남습니다. PATH를 바꿨다면 앱과 PowerShell을 다시 시작하세요.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest
```

실제 입력 영상은 저장소에 포함되지 않습니다. 단위 테스트는 시간/crop/레터박스 변환/중국어 정규화/유사도/세그먼트/SRT/JSON/변화 감지를 실제 영상 없이 검증합니다. 실제 OCR 품질은 사용자의 영상과 crop으로 30초 시험을 거쳐 확인해야 합니다.

## 구조와 향후 계획

GUI와 CLI는 동일한 `OcrPipeline`을 사용하며 영상, 감지, OCR, 자막 모듈이 분리되어 있습니다. `OcrEngine` 프로토콜 구현을 추가하면 향후 PaddleOCR, Apple Vision 또는 다른 ONNX 엔진으로 교체할 수 있습니다. 다음 개선 후보는 여러 전처리 결과 중 confidence가 가장 높은 결과 선택, 문자 후보 중심 변화 감지, OCR 결과 교정 UI, 재개 가능한 체크포인트입니다.
