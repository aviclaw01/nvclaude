# claude-nemotron.ps1 — Windows PowerShell twin of claude-nemotron.sh
#   irm <raw-url-of-this-file> | iex                       # interactive picker
#   & ./claude-nemotron.ps1 ultra                          # skip the picker
# Switch later: ollama launch claude --model <model>   Undo: ollama launch claude --restore
param([string]$Choice = "")
$ErrorActionPreference = "Stop"
$Models = @(
  @{ Tag = "nemotron-3-nano:30b-cloud"; Desc = "Nemotron 3 Nano 30B  - fastest, cheapest" },
  @{ Tag = "nemotron-3-super:cloud";    Desc = "Nemotron 3 Super 120B (12B active) - balanced" },
  @{ Tag = "nemotron-3-ultra:cloud";    Desc = "Nemotron 3 Ultra 550B (55B active) - strongest, 1M context" }
)
function Say($m) { Write-Host "==> $m" -ForegroundColor Green }

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Say "Installing Ollama"; winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
}
try { Invoke-RestMethod -TimeoutSec 3 http://localhost:11434/api/version | Out-Null }
catch { Say "Starting Ollama server"; Start-Process ollama -ArgumentList serve -WindowStyle Hidden; Start-Sleep 3 }

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
  Say "Installing Claude Code"; irm https://claude.ai/install.ps1 | iex
  $env:Path = [Environment]::GetEnvironmentVariable("Path","User") + ";" + $env:Path
}

Say "Checking Ollama sign-in (needed for :cloud models)"; ollama signin

switch -Regex ($Choice) {
  '^nano$'      { $Model = $Models[0].Tag }
  '^super$'     { $Model = $Models[1].Tag }
  '^ultra$'     { $Model = $Models[2].Tag }
  '^nemotron'   { $Model = $Choice }
  '^$' {
    Write-Host "`nChoose a Nemotron model:"
    for ($i = 0; $i -lt $Models.Count; $i++) { "  {0}) {1,-28} {2}" -f ($i+1), $Models[$i].Tag, $Models[$i].Desc }
    $n = Read-Host "  [1-$($Models.Count)] (default 3)"; if (-not $n) { $n = 3 }
    $Model = $Models[[int]$n - 1].Tag
  }
  default { throw "unknown model shortcut '$Choice' (use nano | super | ultra | a full nemotron-* tag)" }
}
Say "Launching Claude Code on $Model"
ollama launch claude --model $Model --yes
