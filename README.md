# nvclaude

Run **Claude Code** on any free model from [build.nvidia.com](https://build.nvidia.com/models). One file, no dependencies.

## Install (once)

macOS / Linux / WSL:
```bash
curl -fsSL https://raw.githubusercontent.com/aviclaw01/nvclaude/main/install.sh | bash
```

Windows PowerShell:
```powershell
irm https://raw.githubusercontent.com/aviclaw01/nvclaude/main/install.ps1 | iex
```

The installer puts `nvclaude` on your PATH, installs Claude Code if missing, asks once for your
[NVIDIA API key](https://build.nvidia.com/settings/api-keys) (free), and opens the model picker.

## Use

```
nvclaude              launch Claude Code on your last model
nvclaude pick         choose a model from the full catalog (type to filter)
nvclaude ultra        Nemotron 3 Ultra   (also: nano | super | lightning)
nvclaude list         print the catalog
nvclaude key          change the API key
nvclaude serve        proxy only, for VS Code or other clients
nvclaude -- --continue   pass args through to claude
```

Inside Claude Code, `/model` also lists every NVIDIA model.

## How it works

A local proxy (127.0.0.1:8787) translates Claude Code's Anthropic Messages API into NVIDIA's
OpenAI-compatible API, including streaming and tool calls. Your key is stored in `~/.nvclaude.json`
(owner-only permissions). Nothing else on your Claude Code setup is changed; run plain `claude` to go back to Anthropic.

Env knobs: `NVCLAUDE_PORT`, `NVCLAUDE_MAX_TOKENS` (default 32768), `NVCLAUDE_DEBUG=1`.

## Uninstall

```
rm ~/.local/bin/nvclaude ~/.nvclaude.json          # macOS / Linux
Remove-Item -Recurse $env:LOCALAPPDATA\nvclaude; Remove-Item ~\.nvclaude.json   # Windows
```
