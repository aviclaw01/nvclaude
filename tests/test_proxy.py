"""End-to-end tests for the nvclaude proxy against a mock OpenAI-compatible upstream.

Run:  python3 -m unittest discover -s tests -v
"""
import json, os, sys, threading, unittest, urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ["NVCLAUDE_UPSTREAM"] = "http://127.0.0.1:8799"
import mock_openai  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("nvclaude", os.path.join(os.path.dirname(HERE), "nvclaude.py"))
nv = importlib.util.module_from_spec(spec); spec.loader.exec_module(nv)
PROXY = "http://127.0.0.1:8797"


def post(path, body, stream=False):
    req = urllib.request.Request(PROXY + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=10)
    return r if stream else json.load(r)


def events(resp):
    out = []
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data:"): out.append(json.loads(line[5:]))
    return out


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = ThreadingHTTPServer(("127.0.0.1", 8799), mock_openai.H)
        threading.Thread(target=cls.mock.serve_forever, daemon=True).start()
        nv.STATE.update(api_key="test", model="nvidia/nemotron-3-super-120b-a12b")
        cls.proxy = nv.serve(8797)

    @classmethod
    def tearDownClass(cls):
        cls.mock.shutdown(); cls.proxy.shutdown()

    def test_models_are_prefixed_for_discovery(self):
        data = json.load(urllib.request.urlopen(PROXY + "/v1/models?limit=1000"))["data"]
        ids = [m["id"] for m in data]
        self.assertTrue(all(i.startswith(nv.PREFIX) for i in ids))
        self.assertNotIn(nv.PREFIX + "nvidia/nemotron-3-embed-1b", ids)  # non-chat models filtered

    def test_count_tokens(self):
        self.assertGreater(post("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "hi"}]})["input_tokens"], 0)

    def test_non_streaming_text(self):
        r = post("/v1/messages", {"model": nv.PREFIX + "nvidia/nemotron-3-super-120b-a12b", "max_tokens": 50,
                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual([b for b in r["content"] if b["type"] == "text"][0]["text"], "Hello from mock.")
        self.assertEqual(r["stop_reason"], "end_turn")
        self.assertEqual(mock_openai.LAST["req"]["model"], "nvidia/nemotron-3-super-120b-a12b")

    def test_unknown_claude_model_maps_to_default(self):
        post("/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(mock_openai.LAST["req"]["model"], "nvidia/nemotron-3-super-120b-a12b")

    def test_streaming_with_tool_call_and_history(self):
        body = {"model": "claude-sonnet-5", "max_tokens": 50, "stream": True, "system": [{"type": "text", "text": "sys"}],
                "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
                          {"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "TOOLTEST"}]},
                             {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "ok"},
                                                               {"type": "tool_use", "id": "call_0", "name": "Bash", "input": {"command": "ls"}}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": "file.txt"},
                                                          {"type": "text", "text": "now TOOLTEST"}]}]}
        ev = events(post("/v1/messages", body, stream=True))
        types = [e["type"] for e in ev]
        self.assertEqual(types[0], "message_start"); self.assertEqual(types[-1], "message_stop")
        starts = [e["content_block"]["type"] for e in ev if e["type"] == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text", "tool_use"])   # mock emits reasoning_content first
        args = "".join(e["delta"]["partial_json"] for e in ev if e.get("delta", {}).get("type") == "input_json_delta")
        self.assertEqual(json.loads(args), {"command": "echo hi"})
        self.assertEqual([e for e in ev if e["type"] == "message_delta"][0]["delta"]["stop_reason"], "tool_use")
        up = mock_openai.LAST["req"]
        self.assertEqual([m["role"] for m in up["messages"]], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(len(up["tools"]), 1)  # server-side web_search tool dropped
        self.assertNotIn("thinking", json.dumps(up))

    # ---- issue #9: malformed tool-call arguments are repaired or refused safely
    TOOL = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]

    def _tool_turn(self, key, stream):
        body = {"model": "claude-sonnet-5", "max_tokens": 50, "stream": stream, "tools": self.TOOL,
                "messages": [{"role": "user", "content": "please run it ARGS:%s" % key}]}
        if not stream:
            r = post("/v1/messages", body); return r["content"], r["stop_reason"]
        ev = events(post("/v1/messages", body, stream=True))
        blocks, cur = [], None
        for e in ev:
            if e["type"] == "content_block_start": cur = dict(e["content_block"]); blocks.append(cur)
            elif e["type"] == "content_block_delta":
                d = e["delta"]
                if d["type"] == "text_delta": cur["text"] += d["text"]
                elif d["type"] == "input_json_delta": cur["input"] = json.loads(d["partial_json"])
        stop = [e for e in ev if e["type"] == "message_delta"][0]["delta"]["stop_reason"]
        return blocks, stop

    def test_repairable_variants_yield_clean_tool_use(self):
        for key in ("good", "trailing", "single", "trunc_key", "bare", "dup", "fence", "wrapped"):
            for stream in (True, False):
                blocks, stop = self._tool_turn(key, stream)
                tools = [b for b in blocks if b["type"] == "tool_use"]
                self.assertEqual(len(tools), 1, (key, stream, blocks))
                self.assertEqual(tools[0]["input"], {"command": "echo hi"}, (key, stream))
                self.assertEqual(tools[0]["name"], "Bash", (key, stream))          # wrong-case name fixed
                self.assertEqual(stop, "tool_use", (key, stream))

    def test_unrecoverable_variants_become_text_not_broken_tool_use(self):
        for key in ("garbage", "trunc_str"):
            for stream in (True, False):
                blocks, stop = self._tool_turn(key, stream)
                self.assertFalse([b for b in blocks if b["type"] == "tool_use"], (key, stream))
                self.assertIn("malformed tool call", "".join(b.get("text", "") for b in blocks), (key, stream))
                self.assertEqual(stop, "end_turn", (key, stream))

    def test_truncated_400_error_body_gets_a_json_error_not_a_dropped_socket(self):   # issue #17
        req = urllib.request.Request(PROXY + "/v1/messages", data=json.dumps({"model": nv.PREFIX + "fail/400-truncated", "max_tokens": 5,
                                     "messages": [{"role": "user", "content": "hi"}]}).encode(), method="POST", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm: urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 400)
        body = json.load(cm.exception)
        self.assertEqual(body["type"], "error"); self.assertIn("400", body["error"]["message"]); self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_clean_key_strips_terminal_paste_artifacts(self):   # issue #17 root cause
        raw = "\x1b[200~nvapi-abcDEF123456789012345678\x1b[201~\n"
        self.assertEqual(nv.clean_key(raw), "nvapi-abcDEF123456789012345678")
        self.assertTrue(nv.KEY_RE.fullmatch(nv.clean_key(raw)))
        self.assertIsNone(nv.KEY_RE.fullmatch(nv.clean_key("nvapi-short")))

    def test_streaming_pings_while_upstream_is_slow(self):
        os.environ["NVCLAUDE_PING_SECS"] = "0.5"
        try:
            ev = events(post("/v1/messages", {"model": nv.PREFIX + "slow/2s", "max_tokens": 5, "stream": True,
                                              "messages": [{"role": "user", "content": "hi"}]}, stream=True))
        finally: del os.environ["NVCLAUDE_PING_SECS"]
        types = [e["type"] for e in ev]
        self.assertEqual(types[0], "message_start")
        self.assertGreaterEqual(types.count("ping"), 2)                       # kept alive while waiting
        self.assertEqual(types[-1], "message_stop")
        self.assertIn("Hello from mock.", "".join(e["delta"]["text"] for e in ev if e.get("delta", {}).get("type") == "text_delta"))

    def test_streaming_upstream_failure_is_an_in_stream_error_event(self):
        ev = events(post("/v1/messages", {"model": nv.PREFIX + "fail/504", "max_tokens": 5, "stream": True,
                                          "messages": [{"role": "user", "content": "hi"}]}, stream=True))
        self.assertEqual([e["type"] for e in ev], ["message_start", "error"])
        self.assertEqual(ev[1]["error"]["type"], "overloaded_error"); self.assertIn("timed out", ev[1]["error"]["message"])

    def test_error_types(self):
        for code, t in ((401, "authentication_error"), (429, "rate_limit_error"), (400, "invalid_request_error"), (504, "overloaded_error"), (418, "api_error")):
            self.assertEqual(nv.err_type(code), t)

    # ---- #21 #22 #23 #24
    REF_TOOL = [{"name": "Lookup", "description": "look", "input_schema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                 "properties": {"q": {"$ref": "#/$defs/Q"}, "n": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": 5},
                                "t": {"type": ["string", "null"], "format": "uri"}},
                 "required": ["q", "gone"], "additionalProperties": False, "$defs": {"Q": {"type": "string", "maxLength": 10, "title": "Q"}}}}]

    def test_effort_and_format_are_mapped(self):
        post("/v1/messages", {"model": nv.PREFIX + "nvidia/nemotron-3-super-120b-a12b", "max_tokens": 5, "output_config": {"effort": "max",
              "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}},
              "thinking": {"type": "adaptive"}, "messages": [{"role": "user", "content": "hi"}]})
        up = mock_openai.LAST["req"]
        self.assertEqual(up["reasoning_effort"], "high")
        self.assertEqual(up["response_format"]["type"], "json_schema"); self.assertEqual(up["response_format"]["json_schema"]["schema"]["properties"]["a"]["type"], "string")
        self.assertNotIn("thinking", up); self.assertNotIn("output_config", up)

    def test_unsupported_reasoning_effort_is_dropped_and_remembered(self):
        body = {"model": nv.PREFIX + "fail/reasoning_effort", "max_tokens": 5, "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "hi"}]}
        r = post("/v1/messages", body); self.assertEqual([b for b in r["content"] if b["type"] == "text"][0]["text"], "Hello from mock.")
        self.assertNotIn("reasoning_effort", mock_openai.LAST["req"])
        n = mock_openai.LAST["n"]; post("/v1/messages", body)
        self.assertEqual(mock_openai.LAST["n"], n + 1)                     # second call: no failed attempt, compat remembered

    def test_json_schema_falls_back_to_json_object_with_schema_in_system(self):
        r = post("/v1/messages", {"model": nv.PREFIX + "fail/response_format", "max_tokens": 5, "system": "sys",
                                  "output_config": {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"z": {"type": "number"}}}}},
                                  "messages": [{"role": "user", "content": "hi"}]})
        up = mock_openai.LAST["req"]
        self.assertEqual(up["response_format"], {"type": "json_object"})
        self.assertIn('"z"', up["messages"][0]["content"]); self.assertTrue(up["messages"][0]["content"].startswith("sys"))

    def test_rejected_tool_schema_is_simplified_then_description_only(self):
        for model, expect_props in (("fail/schema", True), ("fail/schema-hard", False)):
            r = post("/v1/messages", {"model": nv.PREFIX + model, "max_tokens": 5, "tools": self.REF_TOOL, "messages": [{"role": "user", "content": "hi"}]})
            params = mock_openai.LAST["req"]["tools"][0]["function"]["parameters"]
            self.assertNotIn("$ref", json.dumps(params), model)
            self.assertEqual(bool(params.get("properties")), expect_props, model)
            self.assertEqual(r["stop_reason"], "end_turn")

    def test_simplify_schema_unit(self):
        sc = nv.simplify_schema(self.REF_TOOL[0]["input_schema"])
        self.assertEqual(sc["properties"]["q"], {"type": "string"})                 # $ref inlined, maxLength/title dropped
        self.assertEqual(sc["properties"]["n"], {"type": "integer"})                # nullable anyOf collapsed, minimum/default dropped
        self.assertEqual(sc["properties"]["t"], {"type": "string"})                 # type list collapsed, format dropped
        self.assertEqual(sc["required"], ["q"])                                     # unknown required entry removed
        for k in ("$schema", "$defs", "additionalProperties"): self.assertNotIn(k, sc)

    def test_launch_env_defaults_and_overrides(self):
        env = nv.launch_env("nvidia/x", 8787, base={"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "0", "ANTHROPIC_API_KEY": "sk-real"})
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8787"); self.assertEqual(env["ANTHROPIC_API_KEY"], "")   # routing is forced
        self.assertEqual(env["ANTHROPIC_MODEL"], nv.PREFIX + "nvidia/x"); self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], nv.PREFIX + "nvidia/x")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"], "0")                                                   # user value kept
        self.assertEqual(env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"], "1")

    def test_reasoning_becomes_thinking_blocks(self):        # #6
        body = {"model": nv.PREFIX + "nvidia/nemotron-3-super-120b-a12b", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
        ev = events(post("/v1/messages", dict(body, stream=True), stream=True))
        starts = [e["content_block"]["type"] for e in ev if e["type"] == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text"])
        deltas = [e["delta"]["type"] for e in ev if e["type"] == "content_block_delta"]
        self.assertEqual(deltas[:2], ["thinking_delta", "signature_delta"]); self.assertIn("text_delta", deltas)
        self.assertEqual("".join(e["delta"].get("thinking", "") for e in ev if e["type"] == "content_block_delta"), "thinking...")
        r = post("/v1/messages", body)
        self.assertEqual([b["type"] for b in r["content"]], ["thinking", "text"]); self.assertEqual(r["content"][0]["thinking"], "thinking...")
        os.environ["NVCLAUDE_SHOW_THINKING"] = "0"
        try:
            ev = events(post("/v1/messages", dict(body, stream=True), stream=True))
            self.assertEqual([e["content_block"]["type"] for e in ev if e["type"] == "content_block_start"], ["text"])
        finally: del os.environ["NVCLAUDE_SHOW_THINKING"]

    def test_repair_args_unit(self):
        S = self.TOOL[0]["input_schema"]
        self.assertEqual(nv.repair_args('{"command": "echo hi", "run_in_background": True}', S)[0]["run_in_background"], True)
        self.assertEqual(nv.repair_args("[1, 2]", S), (None, "not-an-object"))
        self.assertEqual(nv.repair_args('{"command": "echo hi}', S), (None, "unparseable"))   # unterminated string: never guessed
        self.assertEqual(nv.repair_args("", S), ({}, "empty"))


if __name__ == "__main__":
    unittest.main()
