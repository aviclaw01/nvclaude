#!/usr/bin/env bash
# nvclaude installer - macOS / Linux / WSL / Git Bash.
#   curl -fsSL https://raw.githubusercontent.com/aviclaw01/nvclaude/main/install.sh | bash
#   curl -fsSL ... | bash -s -- ultra        # pass a first command/model
set -euo pipefail
BASE="https://raw.githubusercontent.com/aviclaw01/nvclaude/main"
SRC="${NVCLAUDE_SRC:-$BASE/nvclaude.py}"
BIN="$HOME/.local/bin"; mkdir -p "$BIN"
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- python3 -----------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  say "python3 is required and not installed."
  if command -v brew >/dev/null 2>&1; then say "Installing with Homebrew"; brew install python
  elif command -v apt-get >/dev/null 2>&1; then say "Installing with apt"; sudo apt-get update -qq && sudo apt-get install -y -qq python3
  elif command -v dnf >/dev/null 2>&1; then say "Installing with dnf"; sudo dnf install -y python3
  elif command -v pacman >/dev/null 2>&1; then say "Installing with pacman"; sudo pacman -S --noconfirm python
  else die "Install Python 3.9+ from https://www.python.org/downloads/ then re-run"; fi
  command -v python3 >/dev/null 2>&1 || die "python3 still not found after install; open a new shell and re-run"
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || die "Python 3.9+ required; found $(python3 --version 2>&1)"

# ---- download + optional checksum ---------------------------------------------
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
if [[ "$SRC" == /* ]]; then cp "$SRC" "$tmp"; else curl -fsSL "$SRC" -o "$tmp"; fi
if [[ "$SRC" != /* ]]; then
  if sum="$(curl -fsSL "$BASE/nvclaude.py.sha256" 2>/dev/null | awk '{print $1}')" && [ -n "$sum" ]; then
    have="$( (command -v sha256sum >/dev/null && sha256sum "$tmp" || shasum -a 256 "$tmp") | awk '{print $1}')"
    [ "$have" = "$sum" ] || die "checksum mismatch for nvclaude.py (expected $sum, got $have). Try again in a minute; if it persists, report it."
    say "Checksum verified"
  fi
fi
python3 -m py_compile "$tmp" || die "downloaded nvclaude.py does not compile; try again"
install -m 0755 "$tmp" "$BIN/nvclaude"

# ---- PATH ---------------------------------------------------------------------
case ":$PATH:" in *":$BIN:"*) ;; *)
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    [ -f "$rc" ] && ! grep -q '# nvclaude' "$rc" && printf '\nexport PATH="$HOME/.local/bin:$PATH"  # nvclaude\n' >> "$rc"
  done; export PATH="$BIN:$PATH" ;;
esac

ver="$(grep -o '__version__ = "[^"]*"' "$BIN/nvclaude" | cut -d'"' -f2 || true)"
say "nvclaude ${ver:-} installed. From now on just type:  $(printf '\033[1m')nvclaude$(printf '\033[0m')"

# ---- run it (needs a keyboard for the key prompt / picker) --------------------
if [ ! -t 0 ]; then
  if { : </dev/tty; } 2>/dev/null; then exec </dev/tty
  elif [ $# -eq 0 ]; then say "No interactive terminal detected. Open a terminal and run:  nvclaude"; exit 0; fi
fi
exec "$BIN/nvclaude" "$@"
