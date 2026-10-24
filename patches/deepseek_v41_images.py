"""Build-time patch: let DeepSeek-V4.1-Flash-1017's encoder see image blocks.

vLLM's V4.1 wrapper (vllm/tokenizers/deepseek_v41.py) flattens image parts into
the literal IMAGE_PLACEHOLDER text before calling encode_messages. The 1017
encoder (required for 1017 tool calling) rejects that token in message text as
user injection, so every image request failed with "Message content contains
image special token". For messages that contain images, pass the parts through
as ordered text/image blocks instead; the encoder then inserts the placeholder
itself and joins parts with "\\n\\n", producing the same prompt bytes, while
user text that types the token is still rejected. Text-only messages are unchanged.
"""
from pathlib import Path
import sys

path = Path(sys.argv[1])
src = path.read_text()
old = '''        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\\n\\n".join(parts)
'''
new = '''        content = message.get("content")
        if isinstance(content, list):
            parts = []
            blocks = []
            has_image = False
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    parts.append(block.get("text", ""))
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                    url = block.get("image_url") or block.get("url") or "vllm-image"
                    if isinstance(url, dict):
                        url = url.get("url") or "vllm-image"
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
                    has_image = True
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            # csf-stream patch: image messages go to the 1017 encoder as blocks
            message["content"] = blocks if has_image else "\\n\\n".join(parts)
'''
if src.count(old) != 1:
    sys.exit(f"patch target not found exactly once in {path}; vLLM changed, re-check the patch")
path.write_text(src.replace(old, new))
print("patched", path)
