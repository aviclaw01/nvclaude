#!/usr/bin/env bash
# claude-nemotron — one command: install Ollama + Claude Code, pick an NVIDIA Nemotron
# cloud model, and launch Claude Code on it. macOS / Linux / WSL / Git Bash.
#
#   curl -fsSL <raw-url-of-this-file> | bash                 # interactive picker
#   curl -fsSL <raw-url-of-this-file> | bash -s -- ultra     # skip the picker
#   ./claude-nemotron.sh super -- --continue                 # extra args go to claude
#
# Switch models later with:  ollama launch claude --model <model>
# Undo everything with:      ollama launch claude --restore
set -euo pipefail

MODELS=(
  "nemotron-3-nano:30b-cloud|Nemotron 3 Nano 30B  — fastest, cheapest"
  "nemotron-3-super:cloud|Nemotron 3 Super 120B (12B active) — balanced"
  "nemotron-3-ultra:cloud|Nemotron 3 Ultra 550B (55B active) — strongest, 1M context"
)

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

os="$(uname -s)"
# Read prompts from the terminal even when piped through `curl | bash`; fall back to stdin.
IN=""; { : </dev/tty; } 2>/dev/null && IN=/dev/tty
ask() { if [ -n "$IN" ]; then "$@" <"$IN"; else "$@"; fi; }

# ---- 1. Ollama ---------------------------------------------------------------
if ! have ollama; then
  say "Installing Ollama"
  case "$os" in
    Linux)  curl -fsSL https://ollama.com/install.sh | sh ;;
    Darwin) have brew && brew install ollama || die "Install Ollama from https://ollama.com/download/mac then re-run" ;;
    MINGW*|MSYS*|CYGWIN*) winget install -e --id Ollama.Ollama || die "Install Ollama from https://ollama.com/download/windows then re-run" ;;
    *) die "Unsupported OS: $os" ;;
  esac
fi

# Make sure the server is up (Linux installer creates a systemd service; macOS/Windows use the app).
if ! curl -fs -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
  say "Starting Ollama server"
  if [ "$os" = "Darwin" ] && have brew; then brew services start ollama >/dev/null 2>&1 || true; fi
  if ! curl -fs -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    (nohup ollama serve >/dev/null 2>&1 &) ; sleep 3
  fi
  curl -fs -m 3 http://localhost:11434/api/version >/dev/null 2>&1 || die "Ollama server is not reachable on :11434"
fi

# ---- 2. Claude Code ----------------------------------------------------------
if ! have claude; then
  say "Installing Claude Code"
  curl -fsSL https://claude.ai/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
  have claude || die "claude not on PATH after install; open a new shell and re-run"
fi

# ---- 3. Ollama account (cloud models need it; no-op if already signed in) -----
say "Checking Ollama sign-in (needed for :cloud models)"
ask ollama signin || die "Sign-in failed"

# ---- 4. Pick a model ----------------------------------------------------------
choice="${1:-}"; [ $# -gt 0 ] && shift
case "$choice" in
  nano)  model="${MODELS[0]%%|*}" ;;
  super) model="${MODELS[1]%%|*}" ;;
  ultra) model="${MODELS[2]%%|*}" ;;
  nemotron*) model="$choice" ;;
  "")
    echo; echo "Choose a Nemotron model:"
    i=1; for m in "${MODELS[@]}"; do printf '  %d) %-28s %s\n' "$i" "${m%%|*}" "${m#*|}"; i=$((i+1)); done
    printf '  [1-%d] (default 3): ' "${#MODELS[@]}"; ask read -r n
    n="${n:-3}"; [ "$n" -ge 1 ] && [ "$n" -le "${#MODELS[@]}" ] 2>/dev/null || die "invalid choice"
    model="${MODELS[$((n-1))]%%|*}" ;;
  *) die "unknown model shortcut '$choice' (use nano | super | ultra | a full nemotron-* tag)" ;;
esac

# ---- 5. Launch ----------------------------------------------------------------
say "Launching Claude Code on $model"
[ "${1:-}" = "--" ] && shift
if [ "${DRY_RUN:-0}" = "1" ]; then echo "ollama launch claude --model $model --yes -- $*"; exit 0; fi
exec ollama launch claude --model "$model" --yes -- "$@"
