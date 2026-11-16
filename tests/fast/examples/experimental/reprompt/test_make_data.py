import json
import sys

from examples.experimental.reprompt import make_data
from examples.experimental.reprompt.reward import ASSISTANT_PREFIX


class _Tokenizer:
    """Renders messages the way a ChatML template does, and refuses anything but an open final assistant turn."""

    def apply_chat_template(self, messages, *, tokenize, continue_final_message):
        assert not tokenize and continue_final_message
        assert messages[-1] == {"role": "assistant", "content": ASSISTANT_PREFIX}
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)[
            : -len("<|im_end|>\n")
        ]


def test_each_prompt_line_becomes_an_open_assistant_turn(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a photo of a cat \n\na photo of two dogs\n")
    out = tmp_path / "train.jsonl"
    monkeypatch.setattr(make_data.AutoTokenizer, "from_pretrained", lambda name: _Tokenizer())
    monkeypatch.setattr(sys, "argv", ["make_data", str(prompts), "tok", str(out)])

    make_data.main()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["metadata"]["ori_prompt"] for row in rows] == ["a photo of a cat", "", "a photo of two dogs"]
    assert rows[0]["text"].startswith(f"<|im_start|>system\n{make_data.SYSTEM_PROMPT}<|im_end|>\n")
    assert rows[0]["text"].endswith(
        "<|im_start|>user\na photo of a cat<|im_end|>\n<|im_start|>assistant\n" + ASSISTANT_PREFIX
    )
