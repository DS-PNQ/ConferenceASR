; ConfLive installer — Inno Setup 6 script.
; Build on Windows with:  iscc installer\ConfLive.iss
; (install Inno Setup from https://jrsoftware.org/isinfo.php,
;  or let installer\build.ps1 / the GitHub workflow do it.)
;
; Layout expected (produced by installer\build.ps1):
;   installer\stage\ConfLive.exe (+ .NET runtimes if self-contained)
;   installer\stage\backend\...   (conference-asr-translator: run.py, app/, config.yaml, ...)
#define MyAppName "ConfLive"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "ConfLive"
#define MyAppURL "https://github.com/DS-PNQ/ConferenceASR"

[Setup]
AppId={{3F2A1B4C-7D8E-4F5A-9B1C-ConfLive000001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\release
OutputBaseFilename=ConfLive-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; C# frontend (self-contained publish output)
Source: "stage\\ConfLive.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "stage\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs \
  Excludes: "backend"
; Python backend (source layout; models download on first run)
Source: "stage\\backend\\*"; DestDir: "{app}\\backend"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppName}.exe"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppName}.exe"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
{ CUDA/CPU selection happens at runtime: the C# app checks nvidia-smi and
  starts the Python backend with DEVICE=cuda, otherwise DEVICE=cpu.
  Python itself auto-falls back to CPU on any CUDA failure. }

procedure InitializeWizard;
begin
  CreateOutputMsgPage(wpWelcome,
    'Requirements', 'Before you install',
    'ConfLive ships a Python inference backend (Zipformer ASR, Hy-MT2 translation).' + #13#10 +
    'Please have ready:' + #13#10 +
    '  1. Python 3.11 or 3.12 (python.org, check "Add to PATH")' + #13#10 +
    '  2. Internet access on first run (models auto-download)' + #13#10 +
    '  3. NVIDIA GPU drivers for CUDA mode (optional; CPU fallback is automatic).');
end;
