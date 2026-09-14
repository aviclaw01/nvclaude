#!/usr/bin/env python3
"""
nvclaude - run Claude Code on any free model from build.nvidia.com.

One file, no dependencies (Python 3.9+). It:
  1. installs Claude Code if missing
  2. asks once for your NVIDIA API key (build.nvidia.com/settings/api-keys)
  3. lets you pick any chat model from build.nvidia.com (or --model)
  4. runs a local Anthropic->OpenAI translating proxy and launches `claude` on it

Usage:
  nvclaude                 # first run: picker; later: launches on the last model
  nvclaude pick            # choose a different model
  nvclaude ultra           # shortcuts: nano | super | ultra  (Nemotron 3)
  nvclaude nvidia/nemotron-3.5-lightning-30b-a3b   # or any exact model id
  nvclaude list            # print the catalog (add --refresh to bypass the 24h cache)
  nvclaude bench [model…]  # measure first-token latency, streaming and tool calling; feeds the picker badges
  nvclaude info [model]    # context window / output cap / vision for a model
  nvclaude status          # key, models, proxy, versions, and what to do next
  nvclaude fast <model>    # model for Claude Code's quick Haiku-tier work (default: Nemotron 3.5 Lightning); 'none' resets
  nvclaude subagent <model># model for spawned subagents (default: the main model); 'pick' opens the picker
  nvclaude key             # re-enter the NVIDIA API key
  nvclaude serve           # proxy only (for VS Code / other clients)
  nvclaude version         # nvclaude, Python and Claude Code versions
  nvclaude update          # fetch the latest nvclaude.py and replace this one
  nvclaude uninstall       # remove nvclaude (asks before deleting the config)
  nvclaude -- --continue   # anything after -- goes to claude

Inside Claude Code, /model lists every NVIDIA model too (gateway discovery).
Config lives in ~/.nvclaude.json (chmod 600).
"""
import argparse, json, os, re, secrets, shutil, socket, subprocess, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__version__ = "0.3.1"
SELF_URL = os.environ.get("NVCLAUDE_SRC", "https://raw.githubusercontent.com/aviclaw01/nvclaude/main/nvclaude.py")
UPSTREAM = os.environ.get("NVCLAUDE_UPSTREAM", "https://integrate.api.nvidia.com/v1")
CONFIG = os.path.join(os.path.expanduser("~"), ".nvclaude.json")
PREFIX = "nvclaude/"          # exposed model ids must contain "claude" for Claude Code's /model discovery
SKIP = re.compile(r"embed|rerank|reward|safety|guard|ocr|parse|whisper|parakeet|riva|-vl-|vision|fuyu|paligemma|neva|kosmos|florence|clip", re.I)
RUNFILE = os.path.join(os.path.expanduser("~"), ".nvclaude-run.json")   # who is serving on which port, with its token
STATE = {"api_key": "", "model": "", "token": ""}

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

CATALOG_TTL = 86400
KNOWN_SLOW = {"deepseek-ai/deepseek-v4-pro-0813": "3-4 min to first byte, 504 on full requests (measured 2026-09-13)"}

def fetch_catalog():
    data = json.load(http(UPSTREAM + "/models", timeout=20))["data"]
    ids = sorted(m["id"] for m in data if not SKIP.search(m["id"]))
    ids.sort(key=lambda i: (0 if "nemotron" in i else 1, i))   # NVIDIA's own models first
    return ids

def catalog(refresh=False, allow_stale=False):
    """Chat-capable models on build.nvidia.com, cached in the config for CATALOG_TTL. allow_stale: answer from cache
    immediately and refresh in the background (used by /v1/models, which Claude Code times out after 3s)."""
    cfg = load_cfg(); c = cfg.get("catalog") or {}
    age = time.time() - c.get("fetched_at", 0)
    if c.get("ids") and not refresh and (age < CATALOG_TTL or allow_stale):
        if age >= CATALOG_TTL: threading.Thread(target=lambda: catalog(refresh=True), daemon=True).start()
        return c["ids"]
    try:
        ids = fetch_catalog()
        cfg = load_cfg(); cfg["catalog"] = {"ids": ids, "fetched_at": time.time()}; save_cfg(cfg)
        return ids
    except Exception as e:
        if c.get("ids"):
            log("catalog fetch failed (%s); using the cached list from %s" % (e, time.strftime("%Y-%m-%d", time.localtime(c["fetched_at"]))))
            return c["ids"]
        raise

def unavailable():
    return set((load_cfg().get("unavailable") or {}).keys())

def mark_unavailable(model, why):
    cfg = load_cfg(); cfg.setdefault("unavailable", {})[model] = {"why": why[:200], "at": time.time()}; save_cfg(cfg)

def usable_catalog(**kw):
    bad = unavailable(); return [i for i in catalog(**kw) if i not in bad]

def validate_model(model, ids):
    """(ok, suggestions) for a model id typed on the command line."""
    if model in ids: return True, []
    import difflib
    return False, difflib.get_close_matches(model, ids, n=3, cutoff=0.4)

def badge(model, cfg=None):
    """Short annotation for the picker: latency, tool support, slowness, availability."""
    cfg = cfg if cfg is not None else load_cfg()
    if model in (cfg.get("unavailable") or {}): return "unavailable for this account"
    b = (cfg.get("bench") or {}).get(model)
    if b:
        if b.get("error"): return "bench: " + b["error"]
        parts = ["%.0fs to first token" % b["ttfb"]]
        if b["ttfb"] > 60: parts.append("SLOW")
        parts.append("tools ok" if b.get("tool") else "no tool call")
        if not b.get("streams"): parts.append("no streaming")
        return ", ".join(parts)
    if model in KNOWN_SLOW: return "SLOW: " + KNOWN_SLOW[model]
    return ""

def bench(models, timeout=None):
    """Measure time-to-first-token, streaming and tool calling for each model; store under config['bench']."""
    timeout = timeout or int(os.environ.get("NVCLAUDE_BENCH_TIMEOUT", 180))
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + STATE["api_key"], "Accept": "text/event-stream"}
    tool = [{"type": "function", "function": {"name": "ping", "description": "reply tool", "parameters":
             {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
    print("%-46s %10s %8s %6s %7s  %s" % ("model", "first tok", "total", "tools", "stream", "note"))
    results = {}
    for m in models:
        body = {"model": m, "messages": [{"role": "user", "content": "Call the ping tool with text=pong. TOOLTEST"}], "max_tokens": 64,
                "stream": True, "tools": tool, "tool_choice": "auto"}
        t0 = time.time(); first = None; last = None; lines = 0; tool_ok = False; err = None
        try:
            r = http(UPSTREAM + "/chat/completions", "POST", json.dumps(body).encode(), hdr, timeout=timeout)
            for raw in r:
                lines += 1; now = time.time()
                if first is None: first = now - t0
                last = now
                line = raw.decode(errors="replace").strip()
                if line.startswith("data:") and "ping" in line and "tool_calls" in line: tool_ok = True
        except urllib.error.HTTPError as e:
            msg = _read_err(e); err = "HTTP %d" % e.code
            if e.code == 404: mark_unavailable(m, msg); err = "unavailable for this account"
        except Exception as e: err = "timeout after %ds" % timeout if "timed out" in str(e) else str(e)[:40]
        total = time.time() - t0
        res = {"ttfb": round(first or total, 1), "total": round(total, 1), "tool": tool_ok, "streams": bool(first and last and last - first > 0.05 and lines > 3), "error": err, "at": time.time()}
        results[m] = res
        cfg = load_cfg(); cfg.setdefault("bench", {})[m] = res; save_cfg(cfg)
        print("%-46s %9.1fs %7.1fs %6s %7s  %s" % (m[:46], res["ttfb"], res["total"], "yes" if tool_ok else "no", "yes" if res["streams"] else "no", err or ""))
    return results

def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0: return p
    die("no free port")

# ----------------------------------------------------------------------------- Anthropic -> OpenAI
def blocks_text(c):
    if isinstance(c, str): return c
    return "\n".join(b.get("text", "") for b in (c or []) if b.get("type") == "text")

def image_part(b):
    src = b.get("source") or {}
    if src.get("type") == "base64": url = "data:%s;base64,%s" % (src.get("media_type", "image/png"), src.get("data", ""))
    elif src.get("type") == "url": url = src.get("url", "")
    else: return {"type": "text", "text": "[image omitted]"}
    return {"type": "image_url", "image_url": {"url": url}}

def strip_images(o):
    """Replace image parts with a note (for models without vision)."""
    for m in o.get("messages", []):
        if isinstance(m.get("content"), list):
            m["content"] = [p if p["type"] != "image_url" else {"type": "text", "text": "[image omitted: this model has no vision]"} for p in m["content"]]
            if all(p["type"] == "text" for p in m["content"]): m["content"] = "\n".join(p["text"] for p in m["content"])
    return o

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
            parts = []          # OpenAI content parts for the user turn: text and image_url
            def add_text(t):
                if parts and parts[-1]["type"] == "text": parts[-1]["text"] += "\n" + t
                else: parts.append({"type": "text", "text": t})
            for b in c or []:
                t = b.get("type")
                if t == "text": add_text(b.get("text", ""))
                elif t == "tool_result":
                    body = blocks_text(b.get("content")) or ""
                    if b.get("is_error"): body = "Error: " + body
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body or "(no output)"})
                    for ib in (b.get("content") if isinstance(b.get("content"), list) else []):     # screenshots inside tool results
                        if ib.get("type") == "image": parts.append(image_part(ib))
                elif t == "image": parts.append(image_part(b))
                elif t == "document": add_text("[document attachment omitted]")
            parts = [p for p in parts if p["type"] != "text" or p["text"].strip()]
            if parts:
                msgs.append({"role": "user", "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts})
    model = req.get("model", "")
    model = model[len(PREFIX):] if model.startswith(PREFIX) else (model if "/" in model else STATE["model"])
    out = {"model": model, "messages": msgs, "stream": bool(req.get("stream")),
           "max_tokens": min(int(req.get("max_tokens") or 8192), int(os.environ.get("NVCLAUDE_MAX_TOKENS", 32768)), context_for(model) // 4)}
    if out["stream"]: out["stream_options"] = {"include_usage": True}
    for k in ("temperature", "top_p"):
        if k in req: out[k] = req[k]
    if req.get("stop_sequences"): out["stop"] = req["stop_sequences"]
    oc = req.get("output_config") or {}
    if oc.get("effort"): out["reasoning_effort"] = {"max": "high"}.get(oc["effort"], oc["effort"])   # #21
    fmt = oc.get("format") or req.get("output_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"), dict):   # #22
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": fmt.get("name") or "output", "schema": fmt["schema"], "strict": False}}
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
              "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
             for t in req.get("tools", []) if t.get("input_schema") or t.get("type") in (None, "custom")]
    if tools:
        out["tools"] = tools
        tc = req.get("tool_choice") or {}
        out["tool_choice"] = {"auto": "auto", "any": "required", "none": "none",
                              "tool": {"type": "function", "function": {"name": tc.get("name")}}}.get(tc.get("type"), "auto")
    return out

# ----------------------------------------------------------------------------- model compatibility (#21 #22 #23)
_DROP_KEYS = {"$schema", "$id", "title", "examples", "default", "format", "pattern", "minLength", "maxLength", "minimum", "maximum",
              "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "uniqueItems", "additionalProperties", "$comment", "deprecated"}

def simplify_schema(sc, root=None):
    """Reduce a JSON Schema to the subset every OpenAI-compatible backend accepts."""
    root = root if root is not None else sc
    if isinstance(sc, list): return [simplify_schema(x, root) for x in sc]
    if not isinstance(sc, dict): return sc
    if "$ref" in sc:                                            # inline local refs (#/$defs/X or #/definitions/X)
        node = root
        for part in sc["$ref"].lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        merged = dict(node); merged.update({k: v for k, v in sc.items() if k != "$ref"}); return simplify_schema(merged, root)
    out = {}
    for k, v in sc.items():
        if k in _DROP_KEYS or k in ("$defs", "definitions"): continue
        out[k] = simplify_schema(v, root) if k in ("properties", "items", "anyOf", "oneOf", "allOf", "not", "additionalItems") or isinstance(v, (dict, list)) and k != "enum" else v
    if isinstance(out.get("properties"), dict): out["properties"] = {k: simplify_schema(v, root) for k, v in out["properties"].items()}
    if isinstance(out.get("type"), list):                        # ["string","null"] -> "string"
        t = [x for x in out["type"] if x != "null"]; out["type"] = t[0] if len(t) == 1 else (t or ["string"])[0]
    for key in ("anyOf", "oneOf"):                              # [X, {type: null}] -> X
        if isinstance(out.get(key), list):
            alts = [a for a in out[key] if not (isinstance(a, dict) and a.get("type") == "null")]
            if len(alts) == 1: base = out.pop(key); merged = dict(alts[0]); merged.update({k: v for k, v in out.items() if k != key}); out = merged
    if isinstance(out.get("allOf"), list):                      # merge object allOf into the parent
        for part in out.pop("allOf"):
            if isinstance(part, dict):
                out.setdefault("properties", {}).update(part.get("properties") or {})
                out["required"] = sorted(set(out.get("required") or []) | set(part.get("required") or []))
    if "properties" in out or out.get("type") == "object":
        out["type"] = "object"; out.setdefault("properties", {})
        if "required" in out: out["required"] = [r for r in out["required"] if r in out["properties"]] or None
        if out.get("required") is None: out.pop("required", None)
    return out

# Context windows as served on build.nvidia.com (verified 2026-09-13 via API errors / vendor pages). Unknown models: 128K.
CONTEXT = {
    "nvidia/nemotron-3-ultra-550b-a55b": 1_048_576, "nvidia/nemotron-3-super-120b-a12b": 262_144,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 1_048_576, "nvidia/nemotron-3.5-lightning-30b-a3b": 1_048_576,
    "deepseek-ai/deepseek-v4-flash-0731": 1_048_576, "deepseek-ai/deepseek-v4-pro-0813": 1_048_576, "z-ai/glm-5.3-flash": 1_048_576,
    "meta/llama-3.2-11b-vision-instruct": 131_072, "meta/llama-3.2-90b-vision-instruct": 131_072,
}
DEFAULT_CONTEXT = 131_072

def context_for(model):
    m = model[len(PREFIX):] if model.startswith(PREFIX) else model
    if m in CONTEXT: return CONTEXT[m]
    if m.startswith("nvidia/nemotron-3"): return 262_144          # conservative for other Nemotron 3.x variants
    return DEFAULT_CONTEXT

def compat_for(model):
    return STATE.setdefault("compat", {}).setdefault(model, {"drop": set(), "schema": 0, "json_object": False, "no_vision": False})

def apply_compat(o, compat):
    """Rewrite an OpenAI request according to what this model has already rejected."""
    for p in compat["drop"]: o.pop(p, None)
    rf = o.get("response_format")
    if compat["json_object"] and rf and rf.get("type") == "json_schema":
        o["response_format"] = {"type": "json_object"}
        note = "Respond only with a JSON object matching this schema: " + json.dumps(rf["json_schema"]["schema"])
        if o["messages"] and o["messages"][0]["role"] == "system": o["messages"][0]["content"] += "\n\n" + note
        else: o["messages"].insert(0, {"role": "system", "content": note})
    if compat["no_vision"]: strip_images(o)
    if compat["schema"] and o.get("tools"):
        for t in o["tools"]:
            t["function"]["parameters"] = simplify_schema(t["function"]["parameters"]) if compat["schema"] == 1 else {"type": "object", "properties": {}}
    return o

_PARAM_RX = re.compile(r"stream_options|reasoning_effort|parallel_tool_calls|response_format|top_p|temperature|stop\b")
_SCHEMA_RX = re.compile(r"tool|function|parameters|schema|\$ref|properties", re.I)

def adapt_after_400(o, msg, compat):
    """Mutate compat/o after a 400. Returns a description of the change, or None if nothing applies."""
    low = msg.lower()
    if not compat["no_vision"] and re.search(r"multimodal|image", low) and "image_url" in json.dumps(o):
        compat["no_vision"] = True; return "images stripped (model has no vision)"
    for p in _PARAM_RX.findall(low):
        if p == "response_format" and o.get("response_format", {}).get("type") == "json_schema" and not compat["json_object"]:
            compat["json_object"] = True; return "response_format -> json_object"
        if p in o and p != "response_format" or (p == "response_format" and p in o):
            compat["drop"].add(p); return "dropped " + p
    if o.get("tools") and compat["schema"] < 2 and _SCHEMA_RX.search(low):
        compat["schema"] += 1; return "tool schemas -> %s" % ("simplified" if compat["schema"] == 1 else "description-only (model rejects JSON Schema; tool inputs will be weak)")
    return None

STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn"}

def err_type(code):
    return {401: "authentication_error", 403: "authentication_error", 429: "rate_limit_error", 400: "invalid_request_error",
            404: "not_found_error", 502: "overloaded_error", 503: "overloaded_error", 504: "overloaded_error", 529: "overloaded_error"}.get(code, "api_error")

_CTX_RX = re.compile(r"maximum context length|context length|context window|too many tokens|max tokens must not exceed|exceeds the model|token limit|prompt is too long", re.I)

def friendly(code, msg):
    """Rewrite upstream error text so Claude Code's recovery logic recognizes it."""
    if code == 400 and _CTX_RX.search(msg or ""): return "prompt is too long: " + msg
    if code in (401, 403): return "NVIDIA rejected the API key. Run `nvclaude key`. Upstream said: " + msg
    if code == 404: return "Model not available for this account. Run `nvclaude pick`. Upstream said: " + msg
    return msg

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
        self._json(code, {"type": "error", "error": {"type": err_type(code), "message": friendly(code, msg)}})

    def do_HEAD(self): self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

    def _authed(self):
        """Only the Claude Code we launched (or a client given the token) may spend the NVIDIA key."""
        if not STATE["token"]: return True
        auth = self.headers.get("Authorization") or ""
        got = auth[7:] if auth.lower().startswith("bearer ") else self.headers.get("x-api-key") or ""
        if secrets.compare_digest(got, STATE["token"]): return True
        self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "nvclaude: bad or missing proxy token"}}); return False

    def do_GET(self):
        if self.path == "/nvclaude":                                  # unauthenticated liveness/identity probe
            return self._json(200, {"nvclaude": True, "model": STATE["model"], "pid": os.getpid()})
        if not self._authed(): return
        if self.path.split("?")[0] == "/v1/models":
            try: ids = usable_catalog(allow_stale=True)
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
        if not self._authed(): return
        n = int(self.headers.get("Content-Length") or 0)
        try: req = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._err(400, "bad json")
        path = self.path.split("?")[0]
        if path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": max(1, len(json.dumps(req)) // 4)})
        if path != "/v1/messages": return self._err(404, "not found")
        if os.environ.get("NVCLAUDE_DUMP"):                     # debug: append raw Anthropic requests to a file
            with open(os.environ["NVCLAUDE_DUMP"], "a") as f: f.write(json.dumps(req) + "\n")
        body = json.dumps(to_openai(req)).encode()
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + STATE["api_key"], "Accept": "text/event-stream"}
        schemas = {t["name"]: t.get("input_schema") or {} for t in req.get("tools", []) if t.get("name")}
        if req.get("stream"): return self._stream(lambda: self._call(body, hdr), req, schemas)
        try: up = self._call(body, hdr)
        except urllib.error.HTTPError as e: return self._err(e.code, getattr(e, "msg_text", None) or _read_err(e))
        except Exception as e: return self._err(502, "upstream: %s" % e)
        self._once(up, req, schemas)

    def _call(self, body, hdr):
        base = json.loads(body); compat = compat_for(base["model"])
        last = None
        for attempt in range(6):
            o = apply_compat(json.loads(body), compat)
            try: return self._upstream(json.dumps(o).encode(), hdr)
            except urllib.error.HTTPError as e:
                e.msg_text = _read_err(e); last = e
                if e.code == 404 and "not found" in e.msg_text.lower(): mark_unavailable(base["model"], e.msg_text)
                if e.code != 400: raise
                change = adapt_after_400(o, e.msg_text, compat)
                if not change: raise
                log("model %s rejected the request (%s); retrying with %s" % (base["model"], e.msg_text[:80].replace("\n", " "), change))
        raise last

    def _once(self, up, req, schemas):
        try: r = json.load(up)
        finally:
            try: up.close()
            except Exception: pass
        ch = (r.get("choices") or [{}])[0]; m = ch.get("message", {})
        content = []
        think = m.get("reasoning_content") or m.get("reasoning")
        if think and os.environ.get("NVCLAUDE_SHOW_THINKING", "1") != "0": content.append({"type": "thinking", "thinking": think, "signature": ""})
        if m.get("content"): content.append({"type": "text", "text": m["content"]})
        calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"), "args": (tc.get("function") or {}).get("arguments")}
                 for tc in m.get("tool_calls") or []]
        blocks, ok = tool_blocks(calls, schemas); content += blocks
        stop = "tool_use" if ok else ("end_turn" if calls else STOP.get(ch.get("finish_reason"), "end_turn"))
        self._json(200, {"id": r.get("id", "msg_1"), "type": "message", "role": "assistant", "model": req.get("model"),
                         "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage_of(r.get("usage"))})

    def _stream(self, get_up, req, schemas):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def ev(t, d):
            chunk = ("event: %s\ndata: %s\n\n" % (t, json.dumps(d))).encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        def end(): self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        ev("message_start", {"type": "message_start", "message": {"id": "msg_%x" % time.time_ns(), "type": "message", "role": "assistant",
            "model": req.get("model"), "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
        # NVIDIA's free tier can queue a request for minutes before the first byte. Claude Code aborts a stream
        # that is silent for 300s, so wait for upstream in a thread and ping meanwhile.
        box = {}
        def run():
            try: box["up"] = get_up()
            except Exception as e: box["err"] = e
        th = threading.Thread(target=run, daemon=True); th.start()
        ping_every = float(os.environ.get("NVCLAUDE_PING_SECS", 15))
        try:
            while th.is_alive():
                th.join(ping_every)
                if th.is_alive(): ev("ping", {"type": "ping"})
            if "err" in box:
                e = box["err"]
                code = getattr(e, "code", 502); msg = getattr(e, "msg_text", None) or (_read_err(e) if isinstance(e, urllib.error.HTTPError) else "upstream: %s" % e)
                ev("error", {"type": "error", "error": {"type": err_type(code), "message": friendly(code, msg)}}); return end()
        except (BrokenPipeError, ConnectionResetError): return
        up = box["up"]
        idx, cur, calls, usage, finish, last_ping = -1, None, {}, None, None, time.time()
        t0, first, dbg = time.time(), None, bool(os.environ.get("NVCLAUDE_DEBUG"))
        show_thinking = os.environ.get("NVCLAUDE_SHOW_THINKING", "1") != "0"
        # Text and thinking stream live. Tool calls are buffered until the stream ends so their arguments can be
        # validated/repaired as a whole (Claude Code only runs tools after the message completes anyway).
        def close_block():
            nonlocal cur
            if cur is not None:
                if cur == "thinking": ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "signature_delta", "signature": ""}})
                ev("content_block_stop", {"type": "content_block_stop", "index": idx}); cur = None
        def open_block(kind):
            nonlocal idx, cur
            if cur != kind:
                close_block(); idx += 1; cur = kind
                ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block":
                    {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": ""}})
        try:
            for raw in up:
                line = raw.decode(errors="replace").strip()
                if first is None:
                    first = time.time() - t0
                    if dbg: log("<- first upstream chunk after %.1fs" % first)
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: j = json.loads(data)
                except Exception: continue
                if j.get("usage"): usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ch.get("finish_reason"): finish = ch["finish_reason"]
                    r = d.get("reasoning_content") or d.get("reasoning")
                    if r:
                        if show_thinking:
                            open_block("thinking")
                            ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": r}})
                        elif time.time() - last_ping > 5: ev("ping", {"type": "ping"}); last_ping = time.time()
                    if d.get("content"):
                        open_block("text")
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": d["content"]}})
                    for tc in d.get("tool_calls") or []:
                        i = tc.get("index", 0); fn = tc.get("function") or {}
                        c = calls.setdefault(i, {"id": None, "name": "", "args": ""})
                        if tc.get("id"): c["id"] = tc["id"]
                        if fn.get("name") and not c["name"]: c["name"] = fn["name"]
                        if fn.get("arguments"): c["args"] += fn["arguments"]
                        if time.time() - last_ping > 5: ev("ping", {"type": "ping"}); last_ping = time.time()
            close_block()
            if dbg: log("<- stream done in %.1fs: finish=%s blocks=%d tool_calls=%d" % (time.time() - t0, finish, idx + 1, len(calls)))
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
            ev("message_stop", {"type": "message_stop"}); end()
        except (BrokenPipeError, ConnectionResetError): pass
        finally:
            try: up.close()
            except Exception: pass

def running_instance(port):
    """If an nvclaude proxy already serves on port, return its run info (port, token, model); else None."""
    try:
        info = json.load(http("http://127.0.0.1:%d/nvclaude" % port, timeout=2))
        if not info.get("nvclaude"): return None
        run = json.load(open(RUNFILE))
        return run if run.get("port") == port and run.get("token") else None
    except Exception: return None

def serve(port, token=None, write_runfile=False):
    STATE["token"] = token if token is not None else secrets.token_urlsafe(24)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if write_runfile:
        with open(RUNFILE, "w") as f: json.dump({"pid": os.getpid(), "port": port, "token": STATE["token"], "model": STATE["model"]}, f)
        try: os.chmod(RUNFILE, 0o600)
        except Exception: pass
    return srv

def stop(srv):
    try: srv.shutdown(); srv.server_close()
    except Exception: pass
    try:
        if json.load(open(RUNFILE)).get("pid") == os.getpid(): os.remove(RUNFILE)
    except Exception: pass

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

def launch_env(model, port, base=None, token="nvclaude", fast=None, subagent=None):
    """Environment for the claude process. Routing vars are forced; CLAUDE_CODE_* toggles keep any value the user already set.
    fast: model for Claude Code's Haiku tier (titles, quick classification); subagent: model for spawned agents."""
    env = dict(os.environ if base is None else base)
    env.update({"ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % port, "ANTHROPIC_AUTH_TOKEN": token, "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_MODEL": PREFIX + model, "ANTHROPIC_DEFAULT_OPUS_MODEL": PREFIX + model, "ANTHROPIC_DEFAULT_SONNET_MODEL": PREFIX + model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": PREFIX + (fast or model), "CLAUDE_CODE_SUBAGENT_MODEL": PREFIX + (subagent or model)})
    ctx = context_for(model)
    for k, v in {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx), "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(int(ctx * 0.9)),
                 "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1", "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
                 "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                 "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"}.items():
        env.setdefault(k, v)
    return env

# ----------------------------------------------------------------------------- lifecycle (#12)
def self_path():
    """Where this script lives: the installed file (bin/nvclaude or %LOCALAPPDATA%\\nvclaude\\nvclaude.py) or the checkout."""
    return os.path.realpath(sys.argv[0] if sys.argv[0].endswith((".py", "nvclaude")) else __file__)

def claude_version():
    try: return subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=15).stdout.strip() or "unknown"
    except Exception: return "not installed"

def cmd_version():
    print("nvclaude %s\npython %s\nclaude %s\nconfig %s" % (__version__, sys.version.split()[0], claude_version(), CONFIG))

def cmd_update():
    import py_compile, tempfile
    target = self_path()
    log("Downloading %s" % SELF_URL)
    data = open(SELF_URL, "rb").read() if os.path.exists(SELF_URL) else http(SELF_URL, timeout=30).read()
    m = re.search(rb'__version__ = "([^"]+)"', data); newv = m.group(1).decode() if m else "unknown"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), prefix=".nvclaude.", suffix=".py"); os.close(fd)
    with open(tmp, "wb") as f: f.write(data)
    try: py_compile.compile(tmp, doraise=True)
    except Exception as e:
        os.remove(tmp); die("downloaded file does not compile (%s); keeping the current version" % e)
    if data == open(target, "rb").read():
        os.remove(tmp); log("Already up to date (%s)" % __version__); return
    try: os.chmod(tmp, os.stat(target).st_mode)
    except Exception: pass
    os.replace(tmp, target)
    log("Updated %s: %s -> %s" % (target, __version__, newv))

def cmd_uninstall():
    target = self_path(); removed = []
    for f in (target, os.path.join(os.path.dirname(target), "nvclaude.cmd"), RUNFILE):
        if os.path.exists(f):
            try: os.remove(f); removed.append(f)
            except Exception as e: log("could not remove %s: %s" % (f, e))
    if os.name == "nt":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                "$d=Join-Path $env:LOCALAPPDATA 'nvclaude'; $p=[Environment]::GetEnvironmentVariable('Path','User'); "
                "[Environment]::SetEnvironmentVariable('Path', (($p -split ';') | Where-Object { $_ -and $_ -ne $d }) -join ';', 'User')"],
                capture_output=True, text=True, timeout=30)
            removed.append("user PATH entry")
        except Exception: pass
    else:
        for rc in (".zshrc", ".bashrc", ".profile"):
            path = os.path.join(os.path.expanduser("~"), rc)
            if os.path.exists(path):
                lines = open(path).read().split("\n"); kept = [l for l in lines if "# nvclaude" not in l]
                if kept != lines: open(path, "w").write("\n".join(kept)); removed.append("PATH line in " + rc)
    if os.path.exists(CONFIG):
        ans = input("Delete %s (your NVIDIA key, model choice, bench data)? [y/N] " % CONFIG).strip().lower()
        if ans == "y": os.remove(CONFIG); removed.append(CONFIG)
    log("Removed: %s" % (", ".join(removed) or "nothing"))
    log("Claude Code itself was left installed. Bye.")

def pick(ids, current):
    print("\nModels on build.nvidia.com (%d). Type a number, or text to filter, Enter for default.\n" % len(ids))
    shown = ids; cfg = load_cfg()
    while True:
        for n, i in enumerate(shown, 1):
            b = badge(i, cfg); print("  %3d) %-48s %s%s" % (n, i, "(current) " if i == current else "", ("[" + b + "]") if b else ""))
        ans = input("\n  choice [%s]: " % (current or shown[0])).strip()
        if not ans: return current or shown[0]
        if ans.isdigit() and 1 <= int(ans) <= len(shown): return shown[int(ans) - 1]
        shown = [i for i in ids if ans.lower() in i.lower()] or ids
        print()

SHORT = {"nano": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "super": "nvidia/nemotron-3-super-120b-a12b",
         "ultra": "nvidia/nemotron-3-ultra-550b-a55b", "lightning": "nvidia/nemotron-3.5-lightning-30b-a3b"}

FAST_DEFAULT = "nvidia/nemotron-3.5-lightning-30b-a3b"      # Haiku tier: quick background work (measured 0.8s, tools ok)
def _interactive():
    """A real keyboard is attached. On Windows isatty() is true for NUL, so require stdout to be a terminal as well."""
    if os.environ.get("NVCLAUDE_NONINTERACTIVE"): return False
    try: return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception: return False
INTERACTIVE = _interactive()

def mask(k): return ("nvapi-…" + k[-4:]) if k else "MISSING"

def ensure_key(cfg, reset=False):
    """Step 1: a valid NVIDIA key in STATE and on disk, or a clear exit telling the user what to do."""
    key = clean_key(os.environ.get("NVIDIA_API_KEY") or ("" if reset else cfg.get("api_key", "")))
    if key and key != cfg.get("api_key") and not os.environ.get("NVIDIA_API_KEY"):
        log("Stored key contained stray characters; cleaned it."); cfg["api_key"] = key; save_cfg(cfg)
    if key and not KEY_RE.fullmatch(key):
        log("Stored key is not a valid NVIDIA key; please enter it again."); key = ""
    if not key and not INTERACTIVE: die("No NVIDIA API key. Run `nvclaude key` in a terminal, or set NVIDIA_API_KEY.")
    for attempt in range(3):
        if key: break
        import getpass
        print("Get a free key at https://build.nvidia.com/settings/api-keys  (paste it; input is hidden)")
        key = clean_key(getpass.getpass("NVIDIA API key (nvapi-...): "))
        if not KEY_RE.fullmatch(key):
            print("  That doesn't look like an NVIDIA key (expected nvapi-... with 20+ letters/digits). Try again."); key = ""; continue
        ok, why = check_key(key); print("  " + why)
        if not ok: key = ""; continue
        cfg["api_key"] = key; save_cfg(cfg); print("  Saved %s to %s" % (mask(key), CONFIG))
    if not key: die("no valid NVIDIA API key after 3 attempts")
    STATE["api_key"] = key; return key

def resolve_model(requested):
    """Validate a model id or shortcut typed on the command line; exit with suggestions if it is wrong."""
    m = SHORT.get(requested, requested)
    try: ids = catalog()
    except Exception: ids = None
    if ids:
        ok, near = validate_model(m, ids)
        if not ok: die("model '%s' is not in the build.nvidia.com catalog.%s" % (m, ("  Did you mean:\n  " + "\n  ".join(near)) if near else ""))
    if m in unavailable(): die("model '%s' answered 'not found for this account' last time. Run `nvclaude pick`." % m)
    return m

def ensure_model(cfg, requested=None, force_pick=False):
    """Step 2: the main model, validated, chosen interactively if needed, saved."""
    model = resolve_model(requested) if requested else cfg.get("model", "")
    if force_pick or not model:
        if not INTERACTIVE: die("No model chosen. Run `nvclaude pick` in a terminal, or `nvclaude super`.")
        model = pick(usable_catalog(), model)
    if model != cfg.get("model"): cfg["model"] = model; save_cfg(cfg)
    STATE["model"] = model; return model

def set_tier(cfg, slot, value):
    """`nvclaude fast <model|shortcut|none>` / `nvclaude subagent …`: route Claude Code's Haiku tier / subagents elsewhere."""
    key = "model_" + slot
    if value in ("none", "off", "default"): cfg.pop(key, None); save_cfg(cfg); log("%s model cleared (uses %s)" % (slot, "the built-in default" if slot == "fast" else "the main model")); return
    if value == "pick":
        if not INTERACTIVE: die("`nvclaude %s pick` needs a terminal" % slot)
        value = pick(usable_catalog(), cfg.get(key, ""))
    else: value = resolve_model(value)
    cfg[key] = value; save_cfg(cfg); log("%s model: %s" % (slot, value))

def tiers(cfg, model):
    return cfg.get("model_fast") or FAST_DEFAULT, cfg.get("model_subagent") or model

def pid_alive(pid):
    try: os.kill(int(pid), 0); return True
    except Exception: return False

def ensure_proxy(port, serve_only):
    """Step 3: start this session's own proxy. A proxy lives and dies with the nvclaude process that started it, so a
    launch never borrows another session's (that left Claude Code talking to a closed port when the other session ended).
    Two nvclaude sessions simply get two proxies on two ports. Returns (port, token, srv)."""
    try:
        run = json.load(open(RUNFILE))
        if not pid_alive(run.get("pid", -1)): os.remove(RUNFILE)          # stale run file from a session that was killed
    except Exception: pass
    p = free_port(port)
    if p != port: log("Port %d is busy; using :%d for this session" % (port, p))
    srv = serve(p, os.environ.get("NVCLAUDE_TOKEN") or None, write_runfile=True)
    return p, STATE["token"], srv

def cmd_status(cfg, port):
    key = clean_key(os.environ.get("NVIDIA_API_KEY") or cfg.get("api_key", ""))
    model = cfg.get("model", ""); fast, sub = tiers(cfg, model or "(main model)")
    cat = cfg.get("catalog") or {}
    run = running_instance(port)
    if run and not pid_alive(run.get("pid", -1)): run = None
    rows = [("NVIDIA key", mask(key) + ("" if not key or KEY_RE.fullmatch(key) else "  (INVALID format)") + ("  (from NVIDIA_API_KEY env)" if os.environ.get("NVIDIA_API_KEY") else "")),
            ("model", (model + ("  [" + badge(model, cfg) + "]" if badge(model, cfg) else "")) if model else "not chosen  -> nvclaude pick"),
            ("fast tier (Haiku)", fast + ("" if cfg.get("model_fast") else "  (default)")),
            ("subagents", sub + ("" if cfg.get("model_subagent") else "  (= main model)")),
            ("context window", format(context_for(model), ",") + " tokens" if model else "-"),
            ("proxy", ("running on :%d (pid %s)" % (run["port"], run.get("pid"))) if run else "not running (starts with nvclaude)"),
            ("catalog cache", ("%d models, fetched %s" % (len(cat.get("ids", [])), time.strftime("%Y-%m-%d %H:%M", time.localtime(cat.get("fetched_at", 0))))) if cat else "empty"),
            ("unavailable models", str(len(cfg.get("unavailable") or {}))), ("benchmarked models", str(len(cfg.get("bench") or {}))),
            ("claude", claude_version()), ("python", sys.version.split()[0]), ("nvclaude", __version__), ("config", CONFIG)]
    for k, v in rows: print("  %-20s %s" % (k, v))
    if not key: print("\nNext: run `nvclaude key`")
    elif not model: print("\nNext: run `nvclaude pick`")

def main():
    argv = sys.argv[1:]
    if argv[:1] in (["-h"], ["--help"], ["help"]): print(__doc__); return
    claude_args = []
    if "--" in argv: i = argv.index("--"); argv, claude_args = argv[:i], argv[i + 1:]
    cmd = argv[0] if argv else ""; arg = argv[1] if len(argv) > 1 else ""
    port = int(os.environ.get("NVCLAUDE_PORT", 8787))
    cfg = load_cfg()
    KNOWN = {"pick", "list", "info", "bench", "key", "serve", "status", "version", "update", "uninstall", "fast", "subagent"}
    if cmd and cmd not in KNOWN and cmd not in SHORT and "/" not in cmd:
        die("unknown command '%s'. Try: nvclaude | pick | nano | super | ultra | lightning | list | info | bench | status | fast | subagent | key | serve | version | update | uninstall" % cmd)

    # ---- commands that need no key
    if cmd == "version": return cmd_version()
    if cmd == "update": return cmd_update()
    if cmd == "uninstall": return cmd_uninstall()
    if cmd == "status": return cmd_status(cfg, port)
    if cmd == "info":
        m = SHORT.get(arg, arg) or cfg.get("model", "") or die("nvclaude info <model-id>")
        print("%s\n  context window: %s tokens%s\n  max output sent: %d tokens\n  vision: %s" % (m, format(context_for(m), ","),
              "" if m in CONTEXT else " (default; not verified)", min(int(os.environ.get("NVCLAUDE_MAX_TOKENS", 32768)), context_for(m) // 4),
              "yes" if any(k in m for k in ("vision", "omni", "-vl", "gemma-3", "phi-3-vision", "neva")) else "unknown/no")); return
    if cmd == "list":
        for i in catalog(refresh="--refresh" in argv):
            b = badge(i, cfg); print("%-48s %s" % (i, ("[" + b + "]") if b else ""))
        return
    if cmd in ("fast", "subagent"): return set_tier(cfg, cmd, arg or "pick")

    # ---- step 1: key
    ensure_key(cfg, reset=(cmd == "key"))
    if cmd == "key": log("Key ready. Run `nvclaude` to launch."); return
    if cmd == "bench":
        targets = [SHORT.get(x, x) for x in argv[1:]] or [cfg.get("model") or die("nvclaude bench <model…>")]
        bench(targets); return

    # ---- step 2: model
    requested = cmd if (cmd in SHORT or "/" in cmd) else None
    model = ensure_model(cfg, requested, force_pick=(cmd == "pick" and arg == "") )
    if cmd == "pick" and arg in ("fast", "subagent"): return set_tier(cfg, arg, "pick")
    fast, sub = tiers(cfg, model)

    # ---- step 3: proxy
    p, token, srv = ensure_proxy(port, cmd == "serve")
    if cmd == "serve":
        log("Proxy on http://127.0.0.1:%d  (ANTHROPIC_BASE_URL)  default model: %s" % (p, model))
        log("Proxy token (ANTHROPIC_AUTH_TOKEN): %s" % token)
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: stop(srv); return

    # ---- step 4: Claude Code
    b = badge(model)
    if "SLOW" in b or "unavailable" in b: log("Warning: %s is marked %s. `nvclaude super` is a fast alternative." % (model, b))
    claude = ensure_claude()
    env = launch_env(model, p, token=token, fast=fast, subagent=sub)
    log("Claude Code on %s  (fast tier: %s%s; proxy :%d). Switch with /model or `nvclaude pick`." % (model, fast, "" if sub == model else "; subagents: " + sub, p))
    try: rc = subprocess.call([claude] + claude_args, env=env)
    except KeyboardInterrupt: rc = 130
    finally: stop(srv)
    sys.exit(rc)

if __name__ == "__main__":
    main()
