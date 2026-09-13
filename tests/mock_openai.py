import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LAST = {}
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        b=json.dumps({"data":[{"id":"nvidia/nemotron-3-super-120b-a12b"},{"id":"nvidia/nemotron-3-embed-1b"},{"id":"meta/llama-3.3-70b-instruct"}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        req=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LAST["req"] = req
        want_tool = bool(req.get("tools")) and "TOOLTEST" in json.dumps(req["messages"])
        if not req.get("stream"):
            msg={"role":"assistant","content":"Hello from mock."}
            if want_tool: msg["tool_calls"]=[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"echo hi\"}"}}]
            b=json.dumps({"id":"chatcmpl-1","choices":[{"message":msg,"finish_reason":"tool_calls" if want_tool else "stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Transfer-Encoding","chunked"); self.end_headers()
        def send(o):
            c=("data: "+json.dumps(o)+"\n\n").encode(); self.wfile.write(b"%x\r\n%s\r\n"%(len(c),c)); self.wfile.flush()
        send({"choices":[{"delta":{"reasoning_content":"thinking..."}}]})
        for w in ["Hello"," from"," mock."]: send({"choices":[{"delta":{"content":w}}]})
        if want_tool:
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":""}}]}}]})
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]}}]})
            send({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]})
        else:
            send({"choices":[{"delta":{},"finish_reason":"stop"}]})
        send({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}})
        c=b"data: [DONE]\n\n"; self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n"%(len(c),c)); self.wfile.flush()
if __name__ == "__main__": ThreadingHTTPServer(("127.0.0.1",8799),H).serve_forever()
