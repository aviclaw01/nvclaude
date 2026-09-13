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
        self.assertEqual(r["content"][0]["text"], "Hello from mock.")
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
        self.assertEqual(starts, ["text", "tool_use"])
        args = "".join(e["delta"]["partial_json"] for e in ev if e.get("delta", {}).get("type") == "input_json_delta")
        self.assertEqual(json.loads(args), {"command": "echo hi"})
        self.assertEqual([e for e in ev if e["type"] == "message_delta"][0]["delta"]["stop_reason"], "tool_use")
        up = mock_openai.LAST["req"]
        self.assertEqual([m["role"] for m in up["messages"]], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(len(up["tools"]), 1)  # server-side web_search tool dropped
        self.assertNotIn("thinking", json.dumps(up))


if __name__ == "__main__":
    unittest.main()
