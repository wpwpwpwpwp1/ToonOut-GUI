# 한 앱의 선택형 NVIDIA GPU 가속 팩

## 문제

CPU만 사용하는 PC에서는 ToonOut이 바로 동작해야 하지만, 사용자가 원하면 같은 앱 안에서 NVIDIA GPU 지원을 추가하고 싶다. 대상 PC에는 Python, pip, CUDA Toolkit이 없을 수 있다. CPU용 PyTorch가 로드된 프로세스에서 GPU용 PyTorch를 덮어쓰는 것은 DLL 충돌과 복구 실패 위험이 있다.

## 검토한 선택지

- CPU판과 GPU판 앱을 따로 배포한다.
- 모든 사용자에게 CUDA 라이브러리를 포함한 하나의 큰 앱을 배포한다.
- 실행 중 현재 앱의 PyTorch 파일을 교체한다.
- 한 앱이 선택형 GPU worker 팩을 설치하고 별도 프로세스로 실행한다.

## 결정

하나의 ToonOut 앱은 CPU 추론을 기본 포함한다. 호환되는 NVIDIA GPU가 감지되면 사용자가 `ToonOut-NVIDIA-GPU-Pack.zip`을 앱 안에서 설치할 수 있다.

- 팩은 기본적으로 `%LOCALAPPDATA%\ToonOut\runtimes\nvidia-gpu`에 설치한다.
  `0.1.13`부터는 결정 010에 따라 사용자가 다른 빈 폴더를 선택하거나 설치된 팩을
  이동할 수 있다.
- manifest와 모든 파일의 SHA-256 및 크기를 검증한 뒤 기존 팩을 교체한다.
- GPU 팩은 PyTorch 2.7.1과 CUDA 12.8 런타임을 포함한 독립 worker다.
- GUI와 CPU PyTorch는 그대로 두고 GPU worker를 별도 프로세스로 실행한다.
- 본체는 worker의 출력 환경을 UTF-8로 고정한다. worker의 JSON-lines
  메시지는 비 ASCII 문자를 `\uXXXX`로 이스케이프하여 Windows 시스템
  코드페이지와 무관하게 상태 문구와 경로를 보존한다. 이 환경 설정을
  따르지 않는 이전 frozen worker를 위해 본체는 stdout 원본 바이트를
  UTF-8, 현재 시스템 인코딩, CP949 순서로 판별한다.
- 사용자는 같은 앱에서 CPU/GPU 전환과 GPU 팩 삭제를 할 수 있다.
- 시스템 Python, pip, CUDA Toolkit을 설치하거나 수정하지 않는다.

## 결과와 제약

- 기본 앱 크기는 CPU판 수준으로 유지되고 GPU 사용자만 수 GB의 팩을 추가한다.
- RTX 50 시리즈를 지원하기 위해 CUDA 12.8 wheel을 사용한다.
- GPU worker 팩을 별도로 빌드하고 기본 앱과 함께 또는 다운로드 서버에서 제공해야 한다.
- 현재 worker는 배치마다 종료되므로 새 배치를 시작하면 모델을 다시 로드한다. 안정화 후 장시간 worker 프로토콜로 바꾸면 세션 내 모델 재사용이 가능하다.
- 가속 팩 설치는 관리자 권한이 필요 없는 사용자별 폴더만 사용한다.
