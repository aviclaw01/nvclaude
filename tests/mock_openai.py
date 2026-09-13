import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LAST = {}
# Malformed tool-argument variants, selected by an "ARGS:<key>" marker in the last user message.
VARIANTS = {
    "good": '{"command": "echo hi"}',
    "trailing": '{"command": "echo hi",}',
    "single": "{'command': 'echo hi'}",
    "trunc_key": '{"command": "echo hi", "timeout"',
    "trunc_str": '{"command": "echo h',
    "bare": 'echo hi',
    "dup": '{"command": "echo hi"}{"command": "echo hi"}',
    "fence": '```json\n{"command": "echo hi"}\n```',
    "wrapped": '{"name": "bash", "arguments": {"command": "echo hi"}}',
    "garbage": 'this is not json }{',
}
def variant(req):
    import re
    m = re.search(r"ARGS:(\w+)", json.dumps(req["messages"]))
    return VARIANTS[m.group(1)] if m else None
def chunks(s, n=3):
    k = max(1, len(s) // n); return [s[i:i + k] for i in range(0, len(s), k)]
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        b=json.dumps({"data":[{"id":"nvidia/nemotron-3-super-120b-a12b"},{"id":"nvidia/nemotron-3-embed-1b"},{"id":"meta/llama-3.3-70b-instruct"}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        req=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LAST["n"] = LAST.get("n", 0) + 1
        def bad(msg):
            b = json.dumps({"error": {"message": msg, "type": "invalid_request_error"}}).encode()
            self.send_response(400); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
        m = req.get("model", "")
        if m == "fail/reasoning_effort" and "reasoning_effort" in req: return bad("Unsupported parameter: 'reasoning_effort' is not supported with this model.")
        if m == "fail/response_format" and req.get("response_format", {}).get("type") == "json_schema": return bad("response_format of type json_schema is not supported")
        if m == "fail/schema" and "$ref" in json.dumps(req.get("tools", [])): return bad("Invalid schema for function 'Lookup': $ref is not supported in parameters")
        if m == "fail/schema-hard" and any(t["function"]["parameters"].get("properties") for t in req.get("tools", [])): return bad("tool parameters schema not supported")
        if req.get("model") == "fail/504":
            b=b"upstream timed out"; self.send_response(504); self.send_header("Content-Type","text/plain"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        if req.get("model") == "slow/2s": time.sleep(2)
        if req.get("model") == "fail/400-truncated":   # issue #17: 400 whose chunked body is cut off
            self.send_response(400); self.send_header("Content-Type","application/json"); self.send_header("Transfer-Encoding","chunked"); self.end_headers()
            self.wfile.flush(); self.close_connection = True; self.wfile.write(b""); return
        LAST["req"] = req
        want_tool = bool(req.get("tools")) and "TOOLTEST" in json.dumps(req["messages"])
        if not req.get("stream"):
            msg={"role":"assistant","content":"Hello from mock.","reasoning_content":"thinking..."}
            v = variant(req)
            if want_tool or v: msg["tool_calls"]=[{"id":"call_1","type":"function","function":{"name":"Bash","arguments": v or "{\"command\":\"echo hi\"}"}}]
            b=json.dumps({"id":"chatcmpl-1","choices":[{"message":msg,"finish_reason":"tool_calls" if want_tool else "stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Transfer-Encoding","chunked"); self.end_headers()
        def send(o):
            c=("data: "+json.dumps(o)+"\n\n").encode(); self.wfile.write(b"%x\r\n%s\r\n"%(len(c),c)); self.wfile.flush()
        send({"choices":[{"delta":{"reasoning_content":"thinking..."}}]})
        for w in ["Hello"," from"," mock."]: send({"choices":[{"delta":{"content":w}}]})
        v = variant(req)
        if v is not None:
            name = "bash" if "wrapped" in json.dumps(req["messages"]) else "Bash"   # wrong-case name for the wrapped variant
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":name,"arguments":""}}]}}]})
            for part in chunks(v): send({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":part}}]}}]})
            send({"choices":[{"delta":{},"finish_reason":"length" if "trunc" in v or v.startswith("{\"command\": \"echo h") else "tool_calls"}]})
        elif want_tool:
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":""}}]}}]})
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]})
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]})
        else:
            send({"choices":[{"delta":{},"finish_reason":"stop"}]})
        send({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}})
        c=b"data: [DONE]\n\n"; self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n"%(len(c),c)); self.wfile.flush()
if __name__ == "__main__": ThreadingHTTPServer(("127.0.0.1",8799),H).serve_forever()
