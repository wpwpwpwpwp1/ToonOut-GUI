#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef ProjectRoot
  #define ProjectRoot ".."
#endif

#define AppName "ToonOut"
#define AppPublisher "ToonOut"
#define AppExecutable "ToonOut.exe"
#define AppSource ProjectRoot + "\build\app\dist\ToonOut"

[Setup]
AppId={{5D2739C5-F58A-4F9A-A976-4DAF3B211B73}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}.0
DefaultDirName={localappdata}\Programs\ToonOut
DefaultGroupName=ToonOut
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#ProjectRoot}\dist
OutputBaseFilename=ToonOut-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
SetupLogging=yes
SetupIconFile={#ProjectRoot}\assets\toonout.ico
UninstallDisplayIcon={app}\{#AppExecutable}
LicenseFile={#ProjectRoot}\LICENSE

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; PyInstaller's private payload can contain files that disappear between
; releases. Remove the previous managed payload before copying the new one.
Type: files; Name: "{app}\{#AppExecutable}"
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\ToonOut"; Filename: "{app}\{#AppExecutable}"
Name: "{autodesktop}\ToonOut"; Filename: "{app}\{#AppExecutable}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕 화면에 ToonOut 바로가기 만들기"; GroupDescription: "추가 바로가기:"

[Run]
Filename: "{app}\{#AppExecutable}"; Parameters: "--cleanup-update-cache"; Description: "ToonOut 실행"; Flags: nowait runascurrentuser

[UninstallDelete]
; Downloaded installers are disposable app-owned cache, unlike models and the
; optional GPU runtime, which users manage explicitly inside ToonOut.
Type: filesandordirs; Name: "{localappdata}\ToonOut\updates"
