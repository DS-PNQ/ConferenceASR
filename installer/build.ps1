# Build + stage everything the installer needs.
# Run from the repo folder that contains csharp/, installer/, app/, run.py:
#   powershell -ExecutionPolicy Bypass -File installer/build.ps1
# Output:
#   installer/stage/ConfLive.exe ... (self-contained C# frontend)
#   installer/stage/backend/...      (Python backend source)
#   release/ConfLive-Setup-1.0.0.exe (if Inno Setup's iscc is installed)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Stage = Join-Path $PSScriptRoot "stage"
$FrontOut = Join-Path $Stage "."
$BackOut = Join-Path $Stage "backend"

Write-Host "== 1/3 publish C# frontend (self-contained win-x64) =="
dotnet publish (Join-Path $Root "csharp/ConfLive/ConfLive.csproj") `
  -c Release -r win-x64 --self-contained `
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true `
  -o $FrontOut

Write-Host "== 1b/3 build Electron React UI (single file) =="
Push-Location (Join-Path $Root "electron/ui")
npm ci --no-audit --no-fund
npm run build
Pop-Location

Write-Host "== 2/3 stage Python backend =="
$exclude = @("csharp", "installer", "release", ".venv", "__pycache__",
             ".git", ".cache", "stage", "*.pyc", "node_modules", "dist")
if (Test-Path $BackOut) { Remove-Item $BackOut -Recurse -Force }
New-Item -ItemType Directory $BackOut -Force | Out-Null
Get-ChildItem $Root -Force | Where-Object {
  ($exclude -notcontains $_.Name) -and ($_.Extension -notin @(".pyc"))
} | ForEach-Object {
  if ($_.PSIsContainer) {
    Copy-Item $_.FullName (Join-Path $BackOut $_.Name) -Recurse -Force
  } else {
    Copy-Item $_.FullName $BackOut -Force
  }
}
Get-ChildItem $BackOut -Include __pycache__ -Recurse -Directory |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $BackOut -Include node_modules -Recurse -Directory |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "== 3/3 build Setup.exe (needs Inno Setup) =="
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($iscc) {
  & $iscc.Source (Join-Path $PSScriptRoot "ConfLive.iss")
  Write-Host "Installer built in release/"
} else {
  Write-Host "iscc not found — stage/ is ready; install Inno Setup 6 and rerun,"
  Write-Host "or ship installer/stage directly (ConfLive.exe runs the backend itself)."
}
