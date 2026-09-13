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
if (Test-Path $Src) { Copy-Item $Src $Target -Force } else { Invoke-RestMethod $Src -OutFile $Target }

$Py = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }
$Shim = "@echo off`r`n$Py `"%LOCALAPPDATA%\nvclaude\nvclaude.py`" %*`r`n"
Set-Content -Path (Join-Path $Dir "nvclaude.cmd") -Value $Shim -NoNewline

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$Dir*") {
  [Environment]::SetEnvironmentVariable("Path", "$Dir;$UserPath", "User")
  $env:Path = "$Dir;$env:Path"
}

Write-Host "==> nvclaude installed. From now on just type:  nvclaude" -ForegroundColor Green
& (Join-Path $Dir "nvclaude.cmd") @args
