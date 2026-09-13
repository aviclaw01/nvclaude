#!/usr/bin/env python3
"""
nvclaude — run Claude Code on any free model from build.nvidia.com.

One file, no dependencies (Python 3.8+). It:
  1. installs Claude Code if missing
  2. asks once for your NVIDIA API key (build.nvidia.com/settings/api-keys)
  3. lets you pick any chat model from build.nvidia.com (or --model)
  4. runs a local Anthropic->OpenAI translating proxy and launches `claude` on it

Usage:
  nvclaude                 # first run: picker; later: launches on the last model
  nvclaude pick            # choose a different model
  nvclaude ultra           # shortcuts: nano | super | ultra  (Nemotron 3)
  nvclaude nvidia/nemotron-3.5-lightning-30b-a3b   # or any exact model id
  nvclaude list            # print the catalog
  nvclaude key             # re-enter the NVIDIA API key
  nvclaude serve           # proxy only (for VS Code / other clients)
  nvclaude -- --continue   # anything after -- goes to claude

Inside Claude Code, /model lists every NVIDIA model too (gateway discovery).
Config lives in ~/.nvclaude.json (chmod 600).
"""
import argparse, json, os, re, shutil, socket, subprocess, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("NVCLAUDE_UPSTREAM", "https://integrate.api.nvidia.com/v1")
CONFIG = os.path.join(os.path.expanduser("~"), ".nvclaude.json")
PREFIX = "nvclaude/"          # exposed model ids must contain "claude" for Claude Code's /model discovery
SKIP = re.compile(r"embed|rerank|reward|safety|guard|ocr|parse|whisper|parakeet|riva|-vl-|vision|fuyu|paligemma|neva|kosmos|florence|clip", re.I)
STATE = {"api_key": "", "model": ""}

# ----------------------------------------------------------------------------- helpers
def log(*a): print("\033[1;32m==>\033[0m", *a, flush=True)
def die(m): print("\033[1;31merror:\033[0m", m, file=sys.stderr); sys.exit(1)

def load_cfg():
    try: return json.load(open(CONFIG))
    except Exception: return {}

def save_cfg(c):
    with open(CONFIG, "w") as f: json.dump(c, f, indent=2)
    try: os.chmod(CONFIG, 0o600)
    except Exception: pass

def http(url, method="GET", body=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)

def catalog():
    """Chat-capable models on build.nvidia.com (public endpoint, no key needed)."""
    data = json.load(http(UPSTREAM + "/models", timeout=20))["data"]
    ids = sorted(m["id"] for m in data if not SKIP.search(m["id"]))
    ids.sort(key=lambda i: (0 if "nemotron" in i else 1, i))   # NVIDIA's own models first
    return ids

def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0: return p
    die("no free port")

# ----------------------------------------------------------------------------- Anthropic -> OpenAI
def blocks_text(c):
    if isinstance(c, str): return c
    return "\n".join(b.get("text", "") for b in (c or []) if b.get("type") == "text")

def to_openai(req):
    msgs = []
    sys_txt = blocks_text(req.get("system"))
    if sys_txt: msgs.append({"role": "system", "content": sys_txt})
    for m in req.get("messages", []):
        role, c = m["role"], m.get("content")
        if isinstance(c, str):
            if c.strip(): msgs.append({"role": role, "content": c})
            continue
        if role == "assistant":
            text, calls = [], []
            for b in c or []:
                t = b.get("type")
                if t == "text": text.append(b.get("text", ""))
                elif t == "tool_use":
                    calls.append({"id": b["id"], "type": "function",
                                  "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}})
            out = {"role": "assistant", "content": "\n".join(text)}
            if calls: out["tool_calls"] = calls
            if out["content"] or calls: msgs.append(out)
        else:
            text = []
            for b in c or []:
                t = b.get("type")
                if t == "text": text.append(b.get("text", ""))
                elif t == "tool_result":
                    body = blocks_text(b.get("content")) or ""
                    if b.get("is_error"): body = "Error: " + body
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body or "(no output)"})
                elif t in ("image", "document"): text.append("[attachment omitted]")
            if "\n".join(text).strip(): msgs.append({"role": "user", "content": "\n".join(text)})
    model = req.get("model", "")
    model = model[len(PREFIX):] if model.startswith(PREFIX) else (model if "/" in model else STATE["model"])
    out = {"model": model, "messages": msgs, "stream": bool(req.get("stream")),
           "max_tokens": min(int(req.get("max_tokens") or 8192), int(os.environ.get("NVCLAUDE_MAX_TOKENS", 32768)))}
    if out["stream"]: out["stream_options"] = {"include_usage": True}
    for k in ("temperature", "top_p"):
        if k in req: out[k] = req[k]
    if req.get("stop_sequences"): out["stop"] = req["stop_sequences"]
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
              "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
             for t in req.get("tools", []) if t.get("input_schema") or t.get("type") in (None, "custom")]
    if tools:
        out["tools"] = tools
        tc = req.get("tool_choice") or {}
        out["tool_choice"] = {"auto": "auto", "any": "required", "none": "none",
                              "tool": {"type": "function", "function": {"name": tc.get("name")}}}.get(tc.get("type"), "auto")
    return out

STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn"}

def usage_of(u):
    u = u or {}
    return {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}

# ----------------------------------------------------------------------------- proxy
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *a):
        if os.environ.get("NVCLAUDE_DEBUG"): super().log_message(fmt, *a)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"type": "error", "error": {"type": "api_error", "message": msg}})

    def do_HEAD(self): self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            try: ids = catalog()
            except Exception as e: return self._err(502, str(e))
            data = [{"id": PREFIX + i, "display_name": i, "description": "build.nvidia.com", "type": "model"} for i in ids]
            return self._json(200, {"data": data, "has_more": False})
        self._err(404, "not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: req = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._err(400, "bad json")
        path = self.path.split("?")[0]
        if path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": max(1, len(json.dumps(req)) // 4)})
        if path != "/v1/messages": return self._err(404, "not found")
        body = json.dumps(to_openai(req)).encode()
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + STATE["api_key"], "Accept": "text/event-stream"}
        try: up = http(UPSTREAM + "/chat/completions", "POST", body, hdr, timeout=600)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:4000]
            if e.code == 400 and "stream_options" in msg:      # some models reject it; retry without
                o = json.loads(body); o.pop("stream_options", None)
                try: up = http(UPSTREAM + "/chat/completions", "POST", json.dumps(o).encode(), hdr, timeout=600)
                except urllib.error.HTTPError as e2: return self._err(e2.code, e2.read().decode(errors="replace")[:4000])
            else: return self._err(e.code, msg)
        except Exception as e: return self._err(502, "upstream: %s" % e)
        if req.get("stream"): self._stream(up, req)
        else: self._once(up, req)

    def _once(self, up, req):
        r = json.load(up); ch = (r.get("choices") or [{}])[0]; m = ch.get("message", {})
        content = []
        if m.get("content"): content.append({"type": "text", "text": m["content"]})
        for tc in m.get("tool_calls") or []:
            try: args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception: args = {"_raw": tc["function"].get("arguments")}
            content.append({"type": "tool_use", "id": tc.get("id") or "toolu_%x" % time.time_ns(), "name": tc["function"]["name"], "input": args})
        self._json(200, {"id": r.get("id", "msg_1"), "type": "message", "role": "assistant", "model": req.get("model"),
                         "content": content, "stop_reason": STOP.get(ch.get("finish_reason"), "end_turn"),
                         "stop_sequence": None, "usage": usage_of(r.get("usage"))})

    def _stream(self, up, req):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def ev(t, d):
            chunk = ("event: %s\ndata: %s\n\n" % (t, json.dumps(d))).encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        ev("message_start", {"type": "message_start", "message": {"id": "msg_%x" % time.time_ns(), "type": "message", "role": "assistant",
            "model": req.get("model"), "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
        idx, open_kind, tools, usage, finish, last_ping = -1, None, {}, None, None, time.time()
        def close():
            nonlocal open_kind
            if open_kind is not None: ev("content_block_stop", {"type": "content_block_stop", "index": idx}); open_kind = None
        try:
            for raw in up:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: j = json.loads(data)
                except Exception: continue
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
                    if d.get("reasoning_content") or d.get("reasoning"):
                        if time.time() - last_ping > 5: ev("ping", {"type": "ping"}); last_ping = time.time()
                    if d.get("content"):
                        if open_kind != "text":
                            close(); idx += 1; open_kind = "text"
                            ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": d["content"]}})
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0); fn = tc.get("function") or {}
                        if i not in tools:
                            close(); idx += 1; open_kind = "tool"; tools[i] = idx
                            ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block":
                                {"type": "tool_use", "id": tc.get("id") or "toolu_%x" % time.time_ns(), "name": fn.get("name") or "", "input": {}}})
                        if fn.get("arguments"):
                            ev("content_block_delta", {"type": "content_block_delta", "index": tools[i], "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}})
            close()
            ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": STOP.get(finish, "end_turn"), "stop_sequence": None}, "usage": usage_of(usage)})
            ev("message_stop", {"type": "message_stop"})
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass

def serve(port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv

# ----------------------------------------------------------------------------- launcher
def ensure_claude():
    if shutil.which("claude"): return "claude"
    log("Installing Claude Code")
    if os.name == "nt":
        subprocess.check_call(["powershell", "-NoProfile", "-Command", "irm https://claude.ai/install.ps1 | iex"])
    else:
        subprocess.check_call("curl -fsSL https://claude.ai/install.sh | bash", shell=True)
    os.environ["PATH"] = os.pathsep.join([os.path.expanduser("~/.local/bin"), os.environ.get("PATH", "")])
    return shutil.which("claude") or die("claude still not on PATH; open a new shell and re-run")

def pick(ids, current):
    print("\nModels on build.nvidia.com (%d). Type a number, or text to filter, Enter for default.\n" % len(ids))
    shown = ids
    while True:
        for n, i in enumerate(shown, 1): print("  %3d) %s%s" % (n, i, "   (current)" if i == current else ""))
        ans = input("\n  choice [%s]: " % (current or shown[0])).strip()
        if not ans: return current or shown[0]
        if ans.isdigit() and 1 <= int(ans) <= len(shown): return shown[int(ans) - 1]
        shown = [i for i in ids if ans.lower() in i.lower()] or ids
        print()

SHORT = {"nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "super": "nvidia/nemotron-3-super-120b-a12b",
         "ultra": "nvidia/nemotron-3-ultra-550b-a55b", "lightning": "nvidia/nemotron-3.5-lightning-30b-a3b"}

def main():
    argv = sys.argv[1:]
    if argv[:1] in (["-h"], ["--help"], ["help"]): print(__doc__); return
    claude_args = []
    if "--" in argv: i = argv.index("--"); argv, claude_args = argv[:i], argv[i + 1:]
    cmd = argv[0] if argv else ""
    a = argparse.Namespace(list=cmd == "list", pick=cmd == "pick", serve_only=cmd == "serve", reset_key=cmd == "key",
                           model=SHORT.get(cmd) or (cmd if "/" in cmd else None),
                           port=int(os.environ.get("NVCLAUDE_PORT", 8787)), claude_args=claude_args)
    if cmd and not (a.list or a.pick or a.serve_only or a.reset_key or a.model):
        die("unknown command '%s'. Try: nvclaude | pick | nano | super | ultra | list | key | serve" % cmd)
    cfg = load_cfg()
    if a.list: print("\n".join(catalog())); return

    key = os.environ.get("NVIDIA_API_KEY") or ("" if a.reset_key else cfg.get("api_key", ""))
    if not key:
        import getpass
        print("Get a free key at https://build.nvidia.com/settings/api-keys")
        key = getpass.getpass("NVIDIA API key (nvapi-...): ").strip() or die("no key")
        cfg["api_key"] = key; save_cfg(cfg)
    STATE["api_key"] = key

    model = a.model or cfg.get("model", "")
    if a.pick or not model:
        model = pick(catalog(), model)
    if model != cfg.get("model"): cfg["model"] = model; save_cfg(cfg)
    STATE["model"] = model

    port = free_port(a.port); serve(port)
    if a.serve_only:
        log("Proxy on http://127.0.0.1:%d  (ANTHROPIC_BASE_URL)  default model: %s" % (port, model))
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: return

    claude = ensure_claude()
    env = dict(os.environ)
    env.update({
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % port,
        "ANTHROPIC_AUTH_TOKEN": "nvclaude", "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": PREFIX + model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": PREFIX + model, "ANTHROPIC_DEFAULT_SONNET_MODEL": PREFIX + model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": PREFIX + model, "CLAUDE_CODE_SUBAGENT_MODEL": PREFIX + model,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    })
    args = a.claude_args
    log("Claude Code on %s  (proxy :%d). Switch models with /model or `nvclaude pick`." % (model, port))
    sys.exit(subprocess.call([claude] + args, env=env))

if __name__ == "__main__":
    main()
