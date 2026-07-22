# AMD ROCm GPU 팩

ToonOut은 NVIDIA CUDA worker와 분리된 AMD ROCm worker 팩을 사용할 수 있다. AMD
팩은 시스템 Python이나 ROCm SDK를 수정하지 않고 앱의 별도 프로세스에서 실행된다.

현재 AMD 팩은 ROCm의 nightly multi-architecture wheel을 사용한다. nightly 패키지는
날짜에 따라 달라질 수 있으므로 최종 사용자 PC에서 `pip install`을 실행하지 않는다.
대신 검증한 빌드 PC에서 팩을 한 번 만들고, 생성된 분할 파일과 dependency lock을
Release에 함께 보관한다. 설치된 팩의 모든 파일은 기존 GPU 팩과 동일하게 SHA-256으로
검증된다.

## 팩 빌드

Windows x64와 Python 3.12가 필요하다. 저장소 루트에서 다음 명령을 실행한다.

```powershell
.\scripts\build_amd_gpu_pack.ps1
```

기본값은 nightly multi-arch 인덱스에 게시된 `gfx900`부터 `gfx1250`까지의 주요
커널 팩을 한 worker에 포함한다. 특정 GPU만 지원하는 작은 테스트 팩은 다음처럼 만든다.

```powershell
.\scripts\build_amd_gpu_pack.ps1 -GfxTargets gfx1030,gfx1100,gfx1201 -KeepBuildTree
```

스크립트는 다음 파일을 `dist`에 만든다.

- `ToonOut-AMD-ROCm-GPU-Pack.parts.json`
- `ToonOut-AMD-ROCm-GPU-Pack.partNN`
- `ToonOut-AMD-ROCm-GPU-Pack.requirements.lock.txt`

빌드 입력은 `https://rocm.nightlies.amd.com/whl-multi-arch/`이다. 각 GPU target은
`torch[device-gfxNNNN]` 및 `torchvision[device-gfxNNNN]` extra로 설치되며 ROCm host
라이브러리와 해당 GPU 커널 패키지는 pip가 같은 nightly 집합에서 해결한다.

## 테스트 설치

생성한 ZIP 또는 분할 manifest를 ToonOut 실행 파일과 같은 폴더에 두거나, 처리 장치
창에서 AMD GPU 팩 설치를 누르고 해당 파일을 선택한다. 앱은 팩을 사용자 runtime
폴더로 설치한 뒤 별도 worker를 실행한다.

실제 배포 전에는 포함된 각 gfx target의 하드웨어에서 모델 로드와 한 장 이상의 배경
제거를 확인해야 한다. nightly 패키지는 AMD의 정식 지원 매트릭스 밖 GPU를 포함할 수
있으므로, 지원을 보장하지 않은 target은 문서에 실험적으로 표시한다.

팩을 만들기 전 현재 ROCm PyTorch 환경의 기본 연산만 빠르게 확인하려면 다음 스모크
테스트를 실행한다.

```powershell
python .\scripts\smoke_amd_rocm.py
```

이 테스트는 ToonOut 추론에서 공통으로 사용하는 FP32 convolution, bilinear resize,
reduction, sigmoid 및 GPU-to-CPU 전송을 1024×1024 입력으로 확인한다.
