# nvclaude installer - Windows PowerShell.
# Usage:  irm https://raw.githubusercontent.com/aviclaw01/nvclaude/main/install.ps1 | iex
$ErrorActionPreference = "Stop"
$Src = if ($env:NVCLAUDE_SRC) { $env:NVCLAUDE_SRC } else { "https://raw.githubusercontent.com/aviclaw01/nvclaude/main/nvclaude.py" }
$Dir = Join-Path $env:LOCALAPPDATA "nvclaude"
New-Item -ItemType Directory -Force $Dir | Out-Null

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Host "==> Installing Python" -ForegroundColor Green
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

$Target = Join-Path $Dir "nvclaude.py"
if (Test-Path $Src) { Copy-Item $Src $Target -Force } else {
  Invoke-RestMethod $Src -OutFile $Target
  try {
    $expected = ((Invoke-RestMethod "https://raw.githubusercontent.com/aviclaw01/nvclaude/main/nvclaude.py.sha256") -split '\s+')[0]
    $actual = (Get-FileHash $Target -Algorithm SHA256).Hash.ToLower()
    if ($expected -and $actual -ne $expected.ToLower()) { Remove-Item $Target; throw "checksum mismatch for nvclaude.py (expected $expected, got $actual)" }
    if ($expected) { Write-Host "==> Checksum verified" -ForegroundColor Green }
  } catch [System.Net.WebException] { }
}

$Py = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }
$Shim = "@echo off`r`n$Py `"%LOCALAPPDATA%\nvclaude\nvclaude.py`" %*`r`n"
Set-Content -Path (Join-Path $Dir "nvclaude.cmd") -Value $Shim -NoNewline

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$Dir*") {
  [Environment]::SetEnvironmentVariable("Path", "$Dir;$UserPath", "User")
  $env:Path = "$Dir;$env:Path"
}

$ver = (Select-String -Path $Target -Pattern '__version__ = "([^"]+)"' | Select-Object -First 1).Matches.Groups[1].Value
Write-Host "==> nvclaude $ver installed. From now on just type:  nvclaude" -ForegroundColor Green
Write-Host "    (already-open terminals need to be restarted to see the new PATH entry)" -ForegroundColor DarkGray
& (Join-Path $Dir "nvclaude.cmd") @args
