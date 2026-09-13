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
OpenAI-compatible API, including streaming and tool calls. Malformed tool-call JSON from weaker models
(trailing commas, single quotes, fences, truncation, wrappers) is repaired before Claude Code sees it; unrepairable calls
are surfaced as text instead of breaking the turn. Your key is stored in `~/.nvclaude.json`
(owner-only permissions). Nothing else on your Claude Code setup is changed; run plain `claude` to go back to Anthropic.

Reasoning models' thinking is shown as Claude Code thinking blocks (set `NVCLAUDE_SHOW_THINKING=0` to hide it).
Reasoning effort picked in `/model` is sent as `reasoning_effort`; JSON-schema outputs become `response_format`. When a model
rejects a parameter or a tool schema, the proxy retries with it dropped or simplified and remembers that for the session.

Env knobs: `NVCLAUDE_PORT`, `NVCLAUDE_MAX_TOKENS` (default 32768), `NVCLAUDE_PING_SECS` (default 15), `NVCLAUDE_DEBUG=1`
(request/latency log), `NVCLAUDE_DUMP=<file>` (append every raw Anthropic request, for debugging).

## What the launcher sets

Routing (always forced): `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY=` (empty), `ANTHROPIC_MODEL`,
`ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`.

Toggles (only if you haven't set them yourself): `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` (NVIDIA models in `/model`),
`CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
`CLAUDE_CODE_ATTRIBUTION_HEADER=0`, `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`.

## Uninstall

```
rm ~/.local/bin/nvclaude ~/.nvclaude.json          # macOS / Linux
Remove-Item -Recurse $env:LOCALAPPDATA\nvclaude; Remove-Item ~\.nvclaude.json   # Windows
```

## Tests

```
python3 -m unittest discover -s tests -v
```
