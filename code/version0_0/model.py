"""Two local inference backends using the same text action protocol."""

import json
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Completion:
    text: str
    output_tokens: int
    seconds: float


class ScriptedModel:
    label = "scripted-smoke"
    token_unit = "regex units (smoke only)"

    def count(self, messages: list[dict]) -> int:
        return sum(self.count_output(item["content"]) + 4 for item in messages) + 3

    def count_output(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\s\w]", text))

    def complete(self, messages: list[dict], limit: int, purpose: str) -> Completion:
        if purpose == "compact":
            text = "Previously collected public tests for the supplied candidate solutions."
        else:
            state = json.loads(messages[-1]["content"])
            step = state["turn"]
            default_test = state["public_tests"][0]
            if step == 0:
                action = {"action": "verify", "candidate": "A", "test": default_test}
            elif step == 1:
                test = "preserve" if "preserve" in state["public_tests"] else default_test
                action = {"action": "verify", "candidate": "B", "test": test}
            else:
                action = {"action": "submit" if state["phase"] == 2 else "continue",
                          "candidate": "A" if state["last_result"].get("status") == "assertion_failed" else "B"}
            text = json.dumps(action)
        return Completion(text, min(limit, self.count_output(text)), 0.0)

    def peak_memory(self) -> int | None:
        return None


class TokenizerSmoke(ScriptedModel):
    token_unit = "chat-template tokens (scripted actions)"

    def __init__(self, tokenizer_dir: Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.label = f"scripted:{tokenizer_dir.resolve()}"

    def count(self, messages: list[dict]) -> int:
        return len(chat_ids(self.tokenizer, messages))

    def count_output(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
    )


def check_snapshot(model_dir: Path) -> dict:
    """Check index and shard structure without loading weights or using CUDA."""
    root = model_dir.resolve(strict=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = root / "model.safetensors.index.json"
    expected = {}
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        for tensor, filename in weights.items():
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"Invalid shard filename: {filename}")
            expected.setdefault(filename, set()).add(tensor)
        if not expected:
            raise ValueError("Empty weight index")
    else:
        expected["model.safetensors"] = None
    sizes = {}
    for filename, names in expected.items():
        path = root / filename
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete shard: {filename}")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= min(100_000_000, size - 8):
                raise ValueError(f"Invalid Safetensors header: {filename}")
            tensors = json.loads(stream.read(length))
        tensors.pop("__metadata__", None)
        if not tensors or (names is not None and names != tensors.keys()):
            raise ValueError(f"Shard tensors mismatch: {filename}")
        if 8 + length + max(value["data_offsets"][1] for value in tensors.values()) != size:
            raise ValueError(f"Incomplete or malformed shard: {filename}")
        sizes[filename] = size
    return {"path": str(root), "model_type": config["model_type"], "shards": sizes}


class LocalModel:
    token_unit = "chat-template tokens"

    def __init__(self, model_dir: Path, device: str):
        snapshot = check_snapshot(model_dir)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

        self.torch = torch
        self.label = snapshot["path"]
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this environment")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Selected CUDA device does not support BF16")
        self.metadata = {"model_type": snapshot["model_type"], "transformers": __version__,
                         "torch": torch.__version__, "thinking": False, "decoding": "greedy"}
        self.tokenizer = AutoTokenizer.from_pretrained(self.label, local_files_only=True)
        model_class = AutoModelForCausalLM
        if snapshot["model_type"] == "qwen3_5":
            # Official checkpoint contains both text and vision weights.
            # Supplying text IDs alone bypasses image processing in forward().
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        self.model, loading = model_class.from_pretrained(
            self.label, local_files_only=True, use_safetensors=True,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map={"": str(self.device)}, attn_implementation="sdpa", output_loading_info=True,
        )
        problems = {key: loading[key] for key in ("missing_keys", "mismatched_keys", "error_msgs")
                    if loading.get(key)}
        if problems:
            raise RuntimeError(f"Checkpoint loading failed: {problems}")
        self.model.eval()

    def ids(self, messages: list[dict]) -> list[int]:
        return chat_ids(self.tokenizer, messages)

    def count(self, messages: list[dict]) -> int:
        return len(self.ids(messages))

    def complete(self, messages: list[dict], limit: int, purpose: str) -> Completion:
        torch = self.torch
        ids = self.ids(messages)
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        start = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                max_new_tokens=limit, pad_token_id=pad_id,
            )
        output_ids = outputs[0, len(ids):].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return Completion(text, len(output_ids), time.monotonic() - start)

    def peak_memory(self) -> int | None:
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device)
        return None
