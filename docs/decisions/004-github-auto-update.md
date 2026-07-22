# GitHub Releases 기반 자동 업데이트

## 문제

사용자가 새 버전을 직접 찾아 내려받지 않아도 ToonOut이 업데이트를 확인하고
설치 파일을 준비해야 한다. 실행 중인 Windows 앱은 자기 파일을 안전하게 교체할
수 없고, GPU 팩과 모델은 앱 본체보다 훨씬 크며 별도의 호환 규칙을 가진다.

## 검토한 선택지

- 새 버전 알림만 표시하고 브라우저에서 다운로드하게 한다.
- GitHub Releases에서 자동 확인·다운로드하고 사용자가 한 번 눌러 설치한다.
- 사용자 동의 없이 백그라운드에서 완전히 자동 설치한다.
- MSIX/AppInstaller로 배포한다.
- PyInstaller 앱과 Inno Setup 설치 프로그램을 GitHub Releases로 배포한다.

## 결정

PyInstaller `onedir` 앱을 Inno Setup 사용자별 설치 프로그램으로 묶고,
GitHub Actions가 버전 태그마다 GitHub Release를 만든다.

- 앱은 공개 저장소의 고정된 `releases/latest/download/update.json` 주소를
  시작 후 백그라운드에서 확인한다. GitHub 토큰이나 사용자 식별자는 보내지 않는다.
- 새 버전이면 설치 파일을 자동으로 `%LOCALAPPDATA%\ToonOut\updates`에 받는다.
- manifest의 Ed25519 서명을 내장 공개키로 검증하고, 설치 파일의 크기와
  SHA-256을 다시 검증한다.
- 검증이 끝난 뒤에만 `업데이트 설치` 버튼을 보여 준다. 사용자가 누르면 현재
  앱을 종료하고 Inno Setup을 조용한 모드로 실행한다.
- Inno Setup은 이전 PyInstaller 실행 파일과 `_internal` 페이로드를 먼저 제거해
  새 버전에서 사라진 라이브러리가 남지 않게 한다.
- 새 앱은 설치 직후 실행되어 아직 종료 중인 설치기 잠금이 풀릴 때까지 업데이트
  캐시 정리를 재시도한다. 제거 프로그램도 이 일회성 캐시를 삭제한다.
- 이미지 처리 중에는 설치를 시작할 수 없다. 완료된 결과와 대기열을 방해하지 않는다.
- 모델과 선택형 GPU 팩은 사용자 데이터 폴더에 남기며 앱 업데이트에 포함하지 않는다.
- GPU 호환성은 기존 worker protocol과 GPU 팩 manifest가 계속 판단한다.
- 업데이트 확인이나 다운로드 실패는 현재 버전 사용을 막지 않는다.

## 보안 키

- Ed25519 개인키는 로컬 비밀 파일과 GitHub Actions Secret에만 둔다.
- 공개키만 `update_config.py`에 저장해 앱 실행 파일에 포함한다.
- GitHub 계정이나 Release 파일만 바뀌어도 개인키가 없으면 유효한 manifest를
  만들 수 없다.
- 공개 배포 전에는 설치 프로그램에 Authenticode 코드 서명을 추가해 Windows
  SmartScreen의 게시자 확인도 제공한다. Ed25519 검증은 앱 내부 공급망 검증이고
  Authenticode는 Windows 사용자 경험과 게시자 확인을 위한 별도 계층이다.

## 결과와 제약

- GitHub Release와 Actions가 배포 서버와 빌드 파이프라인 역할을 한다.
- 첫 릴리스 전에 저장소 소스, 기본 브랜치, 공개키와 GitHub Secret을 구성해야 한다.
- 설치 프로그램은 `%LOCALAPPDATA%\Programs\ToonOut`에 설치하므로 관리자 권한이
  필요 없다.
- 서명 키가 구성되지 않은 개발 빌드는 자동 업데이트를 시도하지 않는다.
