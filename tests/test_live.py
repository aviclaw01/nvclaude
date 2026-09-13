"""Live smoke test against build.nvidia.com. Skipped unless NVIDIA_API_KEY is set.

    NVIDIA_API_KEY=nvapi-... python3 -m unittest tests.test_live -v
    NVCLAUDE_LIVE_MODELS="nvidia/nemotron-3-super-120b-a12b,other/model" ...   # override the list

Each model gets one streaming request offering a single tool that the prompt asks it to call. The test asserts a
parseable tool call arrives (or records the failure), prints a summary table, and writes docs/models.live.json.
"""
import json, os, sys, time, unittest, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("NVIDIA_API_KEY", "")
DEFAULT_MODELS = ["nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"]
MODELS = [m for m in os.environ.get("NVCLAUDE_LIVE_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m]
UPSTREAM = "https://integrate.api.nvidia.com/v1"
TIMEOUT = int(os.environ.get("NVCLAUDE_LIVE_TIMEOUT", 120))


def stream_tool_call(model):
    body = {"model": model, "max_tokens": 64, "stream": True, "tool_choice": "auto",
            "messages": [{"role": "user", "content": "Call the ping tool with text=pong."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "echo", "parameters":
                       {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]}
    req = urllib.request.Request(UPSTREAM + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "Accept": "text/event-stream"})
    t0 = time.time(); first = None; args = ""; name = None; lines = 0; retry_after = None
    try:
        r = urllib.request.urlopen(req, timeout=TIMEOUT)
        for raw in r:
            lines += 1
            if first is None: first = time.time() - t0
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"): continue
            for ch in json.loads(line[5:]).get("choices", []):
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    fn = tc.get("function") or {}; name = name or fn.get("name"); args += fn.get("arguments") or ""
        parsed = json.loads(args) if args else None
        return {"ttfb": round(first or 0, 1), "total": round(time.time() - t0, 1), "lines": lines, "tool": name, "args": parsed, "status": 200}
    except urllib.error.HTTPError as e:
        retry_after = e.headers.get("Retry-After")
        try: body = e.read().decode(errors="replace")[:200]
        except Exception: body = ""
        return {"ttfb": None, "total": round(time.time() - t0, 1), "status": e.code, "error": body, "retry_after": retry_after}
    except Exception as e:
        return {"ttfb": None, "total": round(time.time() - t0, 1), "status": None, "error": type(e).__name__}


@unittest.skipUnless(KEY, "NVIDIA_API_KEY not set")
class LiveTests(unittest.TestCase):
    results = {}

    def test_tool_calls_on_each_model(self):
        failures = []
        for m in MODELS:
            res = stream_tool_call(m); LiveTests.results[m] = res
            ok = res.get("tool") == "ping" and isinstance(res.get("args"), dict) and res["args"].get("text")
            print("%-48s status=%s first=%ss total=%ss tool=%s %s" % (m, res.get("status"), res.get("ttfb"), res["total"], "ok" if ok else "NO", res.get("error", "") or ""))
            if not ok: failures.append(m)
        self.assertEqual(failures, [], "models that did not produce a usable tool call: %s" % failures)

    @classmethod
    def tearDownClass(cls):
        out = os.path.join(os.path.dirname(HERE), "docs", "models.live.json")
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump({"measured_at": time.strftime("%Y-%m-%d"), "results": cls.results}, open(out, "w"), indent=1)
        except Exception: pass


if __name__ == "__main__":
    unittest.main()
