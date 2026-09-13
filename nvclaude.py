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

_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")   # terminal escape sequences (bracketed paste, arrow keys)
KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")

def clean_key(k):
    """Strip escape sequences, whitespace and non-printables that terminals inject into a pasted key."""
    k = _CSI.sub("", k or "")
    return "".join(c for c in k if c.isprintable() and not c.isspace())

def check_key(k):
    """Ask NVIDIA whether the key works (1-token request). Returns (ok, message)."""
    body = json.dumps({"model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    try:
        http(UPSTREAM + "/chat/completions", "POST", body, {"Content-Type": "application/json", "Authorization": "Bearer " + k}, timeout=30)
        return True, "key accepted by NVIDIA"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403): return False, "NVIDIA rejected the key (HTTP %d)" % e.code
        if e.code == 400: return False, "NVIDIA returned 400: the key looks malformed"
        return True, "could not fully verify (HTTP %d); continuing" % e.code
    except Exception as e:
        return True, "could not reach NVIDIA (%s); continuing" % e

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

def _read_err(e):
    """Read an HTTPError body defensively: NVIDIA sometimes closes a chunked error body early (IncompleteRead)."""
    try: msg = e.read().decode(errors="replace")
    except Exception as ex:
        msg = getattr(ex, "partial", b"").decode(errors="replace") if hasattr(ex, "partial") else ""
    msg = (msg or "").strip() or ("HTTP %s %s (empty error body from upstream)" % (e.code, getattr(e, "reason", "")))
    if os.environ.get("NVCLAUDE_DEBUG"): log("<- upstream %s: %s" % (e.code, msg[:1000]))
    return msg[:4000]

def usage_of(u):
    u = u or {}
    return {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}

# ----------------------------------------------------------------------------- tool-call repair
# Open models sometimes emit tool arguments that aren't valid JSON (trailing commas, single quotes,
# Python literals, markdown fences, truncation at max_tokens, duplicated objects, a bare value, or a
# Hermes-style {"name":..,"arguments":..} wrapper). Claude Code needs a clean dict, so we repair here.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_TRAIL = re.compile(r",\s*([}\]])")
_PYLIT = [(re.compile(r"(?<![\w\"'])True(?![\w\"'])"), "true"), (re.compile(r"(?<![\w\"'])False(?![\w\"'])"), "false"),
          (re.compile(r"(?<![\w\"'])None(?![\w\"'])"), "null")]

def _scan(s):
    """Walk s string-aware; return (first balanced {...} or None, list of unclosed closers, in_string_at_end)."""
    start, depth, stack, in_str, esc, first = s.find("{"), 0, [], False, False, None
    for i, c in enumerate(s):
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        elif c == '"': in_str = True
        elif c in "{[": stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
            if first is None and start >= 0 and i > start and not stack: first = s[start:i + 1]
    return first, stack, in_str

def _try(s):
    try: return json.loads(s)
    except Exception: return None

def _close(s):
    """Close an unterminated JSON prefix (output cut at max_tokens); drop a dangling partial token first."""
    for _ in range(6):
        _, stack, in_str = _scan(s)
        if in_str: return None                 # cut inside a string value: refuse to guess the rest
        v = _try(s + "".join(reversed(stack)))
        if v is not None: return v
        cut = max(s.rfind(","), s.rfind("{"), s.rfind("["))       # drop the trailing partial key/value
        if cut <= 0: return None
        s = s[:cut] if s[cut] == "," else s[:cut + 1]
    return None

def repair_args(raw, schema=None):
    """Parse a tool-call arguments string leniently. Returns (dict, how) or (None, reason)."""
    schema = schema or {}
    props = list((schema.get("properties") or {}).keys())
    raw = (raw or "").strip()
    if not raw: return {}, "empty"
    steps, s = [], raw
    v = _try(s)
    if v is None:
        t = _FENCE.sub("", s).strip()
        if t != s: s = t; steps.append("fence")
        first, _, _ = _scan(s)
        if first and first != s: s = first; steps.append("first-object")
        v = _try(s)
    if v is None:
        t = _TRAIL.sub(r"\1", s)
        for rx, rep in _PYLIT: t = rx.sub(rep, t)
        if t != s: s = t; steps.append("literals")
        v = _try(s)
    if v is None and "'" in s and s.count('"') < 2:
        t = s.replace("'", '"'); v = _try(t)
        if v is not None: s = t; steps.append("single-quotes")
    if v is None:
        v = _close(s)
        if v is not None: steps.append("truncated")
    ptype = (schema.get("properties") or {}).get(props[0], {}).get("type") if len(props) == 1 else None
    def bare_ok(val): return len(props) == 1 and (ptype is None or {"string": str, "number": (int, float), "integer": int,
                                                     "boolean": bool, "array": list, "object": dict}.get(ptype, object) is not None
                                                     and isinstance(val, {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}.get(ptype, object)))
    if v is None and not re.search(r"[{}\[\]]", raw) and bare_ok(raw):
        v = {props[0]: raw}; steps.append("bare-value")
    if v is None: return None, "unparseable"
    if isinstance(v, dict) and "arguments" in v and set(v) <= {"name", "arguments", "id", "type"} and isinstance(v["arguments"], (dict, str)):
        inner = v["arguments"]; v = inner if isinstance(inner, dict) else (_try(inner) or repair_args(inner, schema)[0]); steps.append("unwrapped")
    if isinstance(v, dict) and "parameters" in v and set(v) <= {"name", "parameters"} and isinstance(v["parameters"], dict):
        v = v["parameters"]; steps.append("unwrapped")
    if not isinstance(v, dict):
        if bare_ok(v): v = {props[0]: v}; steps.append("bare-value")
        else: return None, "not-an-object"
    return v, ",".join(steps) or "ok"

def tool_blocks(calls, schemas):
    """calls: [{"id","name","args"}] in order -> (content blocks, number of valid tool_use blocks)."""
    blocks, ok = [], 0
    names = {n.lower(): n for n in schemas}
    for c in calls:
        name = c.get("name") or ""
        if name not in schemas and name.lower() in names: name = names[name.lower()]     # fix casing
        args, how = repair_args(c.get("args") or "", schemas.get(name))
        if args is None or not name:
            blocks.append({"type": "text", "text": "\n[nvclaude] The model produced a malformed tool call%s that could not be repaired (%s):\n```\n%s\n```\n"
                           % (" to `%s`" % name if name else "", how, (c.get("args") or "")[:2000])})
            continue
        if how != "ok" and os.environ.get("NVCLAUDE_DEBUG"): log("repaired tool call to %s via %s" % (name, how))
        blocks.append({"type": "tool_use", "id": c.get("id") or "toolu_%x" % time.time_ns(), "name": name, "input": args}); ok += 1
    return blocks, ok

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
        try: self._post()
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e:
            import traceback; traceback.print_exc()
            try: self._err(500, "nvclaude proxy error: %r" % e)
            except Exception: pass

    def _upstream(self, body, hdr):
        if os.environ.get("NVCLAUDE_DEBUG"):
            o = json.loads(body); log("-> %s max_tokens=%s tools=%d stream=%s" % (o["model"], o.get("max_tokens"), len(o.get("tools", [])), o.get("stream")))
        return http(UPSTREAM + "/chat/completions", "POST", body, hdr, timeout=600)

    def _post(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: req = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._err(400, "bad json")
        path = self.path.split("?")[0]
        if path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": max(1, len(json.dumps(req)) // 4)})
        if path != "/v1/messages": return self._err(404, "not found")
        body = json.dumps(to_openai(req)).encode()
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + STATE["api_key"], "Accept": "text/event-stream"}
        try: up = self._upstream(body, hdr)
        except urllib.error.HTTPError as e:
            msg = _read_err(e)
            if e.code == 400 and "stream_options" in msg:      # some models reject it; retry without
                o = json.loads(body); o.pop("stream_options", None)
                try: up = self._upstream(json.dumps(o).encode(), hdr)
                except urllib.error.HTTPError as e2: return self._err(e2.code, _read_err(e2))
                except Exception as e2: return self._err(502, "upstream: %s" % e2)
            else: return self._err(e.code, msg)
        except Exception as e: return self._err(502, "upstream: %s" % e)
        schemas = {t["name"]: t.get("input_schema") or {} for t in req.get("tools", []) if t.get("name")}
        if req.get("stream"): self._stream(up, req, schemas)
        else: self._once(up, req, schemas)

    def _once(self, up, req, schemas):
        r = json.load(up); ch = (r.get("choices") or [{}])[0]; m = ch.get("message", {})
        content = []
        if m.get("content"): content.append({"type": "text", "text": m["content"]})
        calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"), "args": (tc.get("function") or {}).get("arguments")}
                 for tc in m.get("tool_calls") or []]
        blocks, ok = tool_blocks(calls, schemas); content += blocks
        stop = "tool_use" if ok else ("end_turn" if calls else STOP.get(ch.get("finish_reason"), "end_turn"))
        self._json(200, {"id": r.get("id", "msg_1"), "type": "message", "role": "assistant", "model": req.get("model"),
                         "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage_of(r.get("usage"))})

    def _stream(self, up, req, schemas):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def ev(t, d):
            chunk = ("event: %s\ndata: %s\n\n" % (t, json.dumps(d))).encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        ev("message_start", {"type": "message_start", "message": {"id": "msg_%x" % time.time_ns(), "type": "message", "role": "assistant",
            "model": req.get("model"), "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
        idx, text_open, calls, usage, finish, last_ping = -1, False, {}, None, None, time.time()
        # Text streams live. Tool calls are buffered until the stream ends so their arguments can be
        # validated/repaired as a whole (Claude Code only runs tools after the message completes anyway).
        def close_text():
            nonlocal text_open
            if text_open: ev("content_block_stop", {"type": "content_block_stop", "index": idx}); text_open = False
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
                        if not text_open:
                            idx += 1; text_open = True
                            ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": d["content"]}})
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0); fn = tc.get("function") or {}
                        c = calls.setdefault(i, {"id": None, "name": "", "args": ""})
                        if tc.get("id"): c["id"] = tc["id"]
                        if fn.get("name") and not c["name"]: c["name"] = fn["name"]
                        if fn.get("arguments"): c["args"] += fn["arguments"]
                        if time.time() - last_ping > 5: ev("ping", {"type": "ping"}); last_ping = time.time()
            close_text()
            blocks, ok = tool_blocks([calls[i] for i in sorted(calls)], schemas)
            for b in blocks:
                idx += 1
                if b["type"] == "text":
                    ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": b["text"]}})
                else:
                    ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
                ev("content_block_stop", {"type": "content_block_stop", "index": idx})
            stop = "tool_use" if ok else ("end_turn" if calls else STOP.get(finish, "end_turn"))
            ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": usage_of(usage)})
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

    key = clean_key(os.environ.get("NVIDIA_API_KEY") or ("" if a.reset_key else cfg.get("api_key", "")))
    if key and key != cfg.get("api_key") and not os.environ.get("NVIDIA_API_KEY"):
        log("Stored key contained stray characters; cleaned it."); cfg["api_key"] = key; save_cfg(cfg)
    if key and not KEY_RE.fullmatch(key):
        log("Stored key is not a valid NVIDIA key; please enter it again."); key = ""
    for attempt in range(3):
        if key: break
        import getpass
        print("Get a free key at https://build.nvidia.com/settings/api-keys  (paste it; input is hidden)")
        key = clean_key(getpass.getpass("NVIDIA API key (nvapi-...): "))
        if not KEY_RE.fullmatch(key):
            print("  That doesn't look like an NVIDIA key (expected nvapi-... with 20+ letters/digits). Try again."); key = ""; continue
        ok, why = check_key(key); print("  " + why)
        if not ok: key = ""; continue
        cfg["api_key"] = key; save_cfg(cfg); print("  Saved nvapi-…%s to %s" % (key[-4:], CONFIG))
    if not key: die("no valid NVIDIA API key after 3 attempts")
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
