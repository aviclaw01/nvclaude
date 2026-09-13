#!/usr/bin/env bash
# nvclaude installer — macOS / Linux / WSL / Git Bash.   Usage:  curl -fsSL https://nvclaude.sh | bash
set -euo pipefail
SRC="${NVCLAUDE_SRC:-https://raw.githubusercontent.com/aviclaw01/nvclaude/main/nvclaude.py}"
BIN="$HOME/.local/bin"; mkdir -p "$BIN"
command -v python3 >/dev/null || { echo "python3 is required. macOS: xcode-select --install   Debian/Ubuntu: sudo apt install python3" >&2; exit 1; }
if [[ "$SRC" == /* ]]; then cp "$SRC" "$BIN/nvclaude"; else curl -fsSL "$SRC" -o "$BIN/nvclaude"; fi
chmod +x "$BIN/nvclaude"
case ":$PATH:" in *":$BIN:"*) ;; *)
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile"; do
    [ -f "$rc" ] && ! grep -q 'nvclaude' "$rc" && printf '\nexport PATH="$HOME/.local/bin:$PATH"  # nvclaude\n' >> "$rc"
  done; export PATH="$BIN:$PATH" ;;
esac
printf '\033[1;32m==>\033[0m nvclaude installed. From now on just type:  \033[1mnvclaude\033[0m\n'
if [ ! -t 0 ] && { : </dev/tty; } 2>/dev/null; then exec </dev/tty; fi   # curl | bash: give the picker a keyboard
exec "$BIN/nvclaude" "$@"
