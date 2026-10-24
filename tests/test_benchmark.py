import io
import json
import unittest
from unittest.mock import patch

from scripts.agent_benchmark import complete


class StreamTests(unittest.TestCase):
    def test_tool_fragments_and_usage_are_preserved(self):
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_", "arguments": "{\"name\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": "\"solution.py\"}"}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 12}},
        ]
        body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
            message, timing = complete("http://localhost:8008", {})
        call = message["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"], {"name": "read_file", "arguments": '{"name":"solution.py"}'})
        self.assertEqual(timing["usage"]["completion_tokens"], 12)
        self.assertEqual(timing["finish_reason"], "tool_calls")
        self.assertIsNotNone(timing["ttft_seconds"])
