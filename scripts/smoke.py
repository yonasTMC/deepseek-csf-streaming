import json
from pathlib import Path
from agent_benchmark import complete

message, timing = complete("http://127.0.0.1:8008", dict(
    model="DeepSeek-V4.1-Flash-streaming",
    messages=[dict(role="user", content="What is 17 + 25? Reply with just the number.")],
    temperature=0, max_tokens=16, chat_template_kwargs=dict(thinking=False)))
result = dict(message=message, timing=timing)
Path(__file__).resolve().parents[1].joinpath("evidence/smoke.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
assert "42" in message.get("content", ""), result
