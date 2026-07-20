# ToonOut 첫 릴리스와 자동 업데이트 설정

자동 업데이트 코드는 공개키만 포함한다. 아래 개인키 생성과 GitHub Secret 등록은
저장소 소유자가 한 번 직접 해야 한다.

## 1. 업데이트 서명 키 생성

프로젝트용 Python 환경에서 다음을 실행한다.

```powershell
python scripts/generate_update_signing_key.py
```

생성 결과:

- `.secrets/update-private-key.txt`: 개인키. Git에 추가하지 않는다.
- `update_config.py`: 앱에 포함할 공개키. Git에 커밋한다.

개인키 파일은 암호화된 비밀번호 관리자나 오프라인 저장 장치에 한 번 더 백업한다.
이 키를 잃으면 이미 배포된 앱이 새 manifest를 검증할 수 없다. 키를 새로 만드는
`--force` 옵션은 기존 사용자용 키 전환 설계 없이 사용하지 않는다.

## 2. GitHub Secret 등록

1. GitHub 저장소 `wpwpwpwpwp1/ToonOut-GUI`를 연다.
2. `Settings → Secrets and variables → Actions`로 이동한다.
3. `New repository secret`을 누른다.
4. 이름을 `UPDATE_SIGNING_PRIVATE_KEY_B64`로 지정한다.
5. `.secrets/update-private-key.txt`의 한 줄 전체를 값으로 붙여 넣는다.

정식 공개 Windows 배포에는 신뢰된 Authenticode 코드 서명 인증서(PFX)가 필요하다.
PFX 파일 전체를 Base64로 만든 `WINDOWS_CODESIGN_CERT_B64`와 인증서 암호인
`WINDOWS_CODESIGN_PASSWORD`를 Secret으로 추가한다. 두 값이 있으면 앱 EXE와
설치 프로그램을 서명·검증해 정식 Release로 게시한다. 두 값이 없으면 설치 파일은
서명하지 않고 `unsigned beta` 사전 릴리스로 게시하며 Windows SmartScreen에서
알 수 없는 게시자 경고가 나타날 수 있다.

GitHub의 `Settings → Actions → General → Workflow permissions`에서
`Read and write permissions`도 선택한다. Release 파일을 올리기 위해 필요하다.

## 3. 소스와 기본 브랜치 올리기

릴리스 소스는 GitHub 기본 브랜치인 `main`에 올린다. 첫 푸시 전에 로컬 브랜치
이름도 맞춘다.

```powershell
git branch -M main
git add .
git status --short
git commit -m "Build ToonOut desktop app with secure auto updates"
git push -u origin main
```

커밋 전 상태에서 `.secrets/`, `dist/`, `build/`, 모델, 사용자 이미지가 포함되지
않았는지 반드시 확인한다. `update_config.py`의 공개키는 포함되어야 한다.

## 4. 첫 설치 파일 릴리스

`app_version.py`의 `APP_VERSION`과 태그가 같아야 한다. 현재 첫 버전은 `0.1.0`이다.

```powershell
git tag v0.1.0
git push origin v0.1.0
```

태그가 올라가면 `.github/workflows/release.yml`이 다음을 자동 실행한다.

1. 전체 테스트
2. CPU 앱 PyInstaller 빌드
3. Inno Setup 사용자별 설치 파일 빌드
4. Ed25519 서명 `update.json` 생성
5. GitHub Release에 설치 파일과 manifest 게시

Actions 탭에서 `Build ToonOut release`가 통과했는지 확인한 뒤 Release의
`ToonOut-Setup-0.1.0.exe`로 최초 설치를 시험한다.

NVIDIA GPU 팩은 약 3.27GB여서 GitHub Release의 단일 파일 제한을 넘는다. CUDA
12.8 빌드 환경에서 `scripts\build_gpu_pack.ps1`을 실행하면 `.parts.json`과
`.partNN` 조각도 함께 만들어진다. `.parts.json`과 모든 조각을 Release에 추가하고,
셋 중 하나라도 빠지지 않았는지 확인한다. GPU 팩 빌드는 큰 원본 ZIP과 분할본을
동시에 만들므로 기본 GitHub-hosted Windows runner에서 만들지 않고, 22GB 이상 여유가
있는 검증된 Windows 빌드 머신에서 만든다. GPU worker EXE도 팩을 만들기 전에
`scripts\sign_windows_binary.ps1`로 서명한다.

CPU Release가 만들어진 뒤 같은 태그에 GPU 파일을 추가한다.

```powershell
gh release upload v0.1.0 `
  dist\ToonOut-NVIDIA-GPU-Pack.parts.json `
  dist\ToonOut-NVIDIA-GPU-Pack.part01 `
  dist\ToonOut-NVIDIA-GPU-Pack.part02
```

조각 수는 팩 크기에 따라 늘 수 있으므로 실제 `.parts.json`의 `parts` 목록과
Release 자산이 정확히 일치하는지 확인한다. 원본 3GB ZIP은 Release에 올리지 않는다.

### 앱 아이콘을 바꿀 때

투명한 정사각형 원본을 `assets/toonout-icon.png`에 둔 뒤 다음 명령으로
Windows 다중 해상도 아이콘을 다시 만든다.

```powershell
python scripts/create_app_icon.py
```

생성되는 `assets/toonout.ico`는 앱 창, 작업 표시줄, EXE, 설치 프로그램에서
공통으로 사용된다. 두 아이콘 파일은 모두 릴리스 소스에 포함한다.

## 5. 다음 버전 배포

코드 변경과 함께 `APP_VERSION`을 예를 들어 `0.1.1`로 올린 뒤 커밋하고 태그를
푸시한다.

```powershell
git add .
git commit -m "Release ToonOut 0.1.1"
git push
git tag v0.1.1
git push origin v0.1.1
```

기존 `0.1.0` 앱을 실행해 다음 흐름을 확인한다.

- 새 Release 자동 확인
- 상단 다운로드 진행 상태
- `업데이트 설치 · 0.1.1` 버튼
- 클릭 한 번으로 앱 종료, 설치, 새 버전 실행
- 모델 위치와 GPU 팩 설정 유지

## Windows 게시자 서명

Ed25519 서명은 ToonOut 앱이 업데이트 manifest의 진위를 검증하고, Authenticode는
Windows가 게시자를 확인하는 별개의 보호 장치다. 인증서가 없는 사전 릴리스는
테스트 사용자에게 Windows 경고와 실행 방법을 명확히 안내한다. 인증서 갱신 시
Secret도 만료 전에 교체한다. 새 인증서의 평판이 쌓이기 전에는 SmartScreen 확인
화면이 나타날 수 있다.
