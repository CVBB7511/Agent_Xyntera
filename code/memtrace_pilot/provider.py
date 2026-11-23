"""Local non-streaming Responses subset used by both mini-swe-agent arms."""

import http.server
import json
import os
import re
import socketserver
import time
import uuid
from pathlib import Path

from common import (
    MODEL,
    OUT,
    OUTPUT_RESERVE,
    ROOT,
    context_window,
    result_directory,
    write,
)


def messages(items):
    result = []
    names = {}
    for item in items:
        kind = item.get("type", "message")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            name = item["name"]
            names[item["call_id"]] = name
            if not result or result[-1]["role"] != "assistant":
                result.append({"role": "assistant", "content": ""})
            result[-1].setdefault("tool_calls", []).append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.loads(item["arguments"]),
                    },
                }
            )
        elif kind == "function_call_output":
            result.append(
                {
                    "role": "tool",
                    "name": names[item["call_id"]],
                    "content": item["output"],
                }
            )
        elif kind == "message":
            value = item["content"]
            if isinstance(value, list):
                value = "".join(x["text"] for x in value)
            role = item["role"]
            result.append(
                {"role": "system" if role == "developer" else role, "content": value}
            )
        else:
            raise ValueError(f"unsupported Responses item {kind}")
    return result


def parse_answer(answer, schemas=()):
    definitions = {
        s["name"]: s.get("parameters", {}).get("properties", {}) for s in schemas
    }
    output = []
    for tool in re.finditer(
        r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>",
        answer,
        re.S,
    ):
        args = {
            key: value.strip("\n")
            for key, value in re.findall(
                r"<parameter=([^>\n]+)>(.*?)</parameter>", tool[2], re.S
            )
        }
        for key, value in list(args.items()):
            field = definitions.get(tool[1], {}).get(key, {})
            if field.get("type") in ("array", "object", "integer", "boolean", "number"):
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    pass  # The harness handles malformed model arguments.
        output.append(
            {
                "type": "function_call",
                "id": "fc_" + uuid.uuid4().hex,
                "call_id": "call_" + uuid.uuid4().hex,
                "name": tool[1],
                "arguments": json.dumps(args),
                "status": "completed",
            }
        )
    if not output:
        output.append(
            {
                "type": "message",
                "id": "msg_" + uuid.uuid4().hex,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }
        )
    return output


class ContextLengthExceeded(ValueError):
    def __init__(self, input_tokens, output_reserve, context_limit):
        self.input_tokens = input_tokens
        self.output_reserve = output_reserve
        self.context_limit = context_limit
        super().__init__(
            f"context_length_exceeded: {input_tokens} input + "
            f"{output_reserve} reserved > {context_limit}"
        )


class Engine:
    def __init__(self):
        import sys

        sys.path.insert(0, str(ROOT / "code/version0_0"))
        from model import LocalModel

        self.context_limit = context_window()
        fraction = os.environ.get("PILOT_CUDA_MEMORY_FRACTION")
        if fraction is not None:
            import torch

            torch.cuda.set_per_process_memory_fraction(float(fraction), 0)
        self.backend = LocalModel(MODEL, "cuda:0")
        if (
            self.context_limit
            > self.backend.model.config.text_config.max_position_embeddings
        ):
            raise ValueError(
                "context window exceeds the installed model's context limit"
            )
        self.backend.torch.manual_seed(20261122)

    def prepare_request(self, request):
        if request.get("model", "").split("/")[-1] != "Qwen3.5-9B":
            raise ValueError("unsupported model")
        chat = messages(request["input"])
        tools = [
            {
                "type": "function",
                "function": {
                    k: t[k] for k in ("name", "description", "parameters") if k in t
                },
            }
            for t in request.get("tools", [])
        ]
        ids = self.backend.tokenizer.apply_chat_template(
            chat,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
            return_dict=False,
        )
        requested = request.get("max_output_tokens")
        limit = (
            OUTPUT_RESERVE if requested is None else min(OUTPUT_RESERVE, int(requested))
        )
        if limit <= 0:
            raise ValueError("max_output_tokens must be positive")
        if len(ids) + limit > self.context_limit:
            raise ContextLengthExceeded(len(ids), limit, self.context_limit)
        return ids, limit

    def preflight(self, request):
        ids, limit = self.prepare_request(request)
        return {
            "input_tokens": len(ids),
            "output_reserve": limit,
            "context_window": self.context_limit,
        }

    def respond(self, request):
        ids, limit = self.prepare_request(request)
        b = self.backend
        tensor = b.torch.tensor([ids], device=b.device)
        before = time.monotonic()
        with b.torch.inference_mode():
            ends = {b.tokenizer.eos_token_id, b.model.config.text_config.eos_token_id}
            params = dict(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_new_tokens=limit,
                eos_token_id=sorted(ends),
                pad_token_id=b.tokenizer.pad_token_id,
            )
            generated = b.model.generate(
                input_ids=tensor, attention_mask=b.torch.ones_like(tensor), **params
            )
        tokens = generated[0, len(ids) :].tolist()
        stop = b.tokenizer.convert_tokens_to_ids("</think>")
        closed = stop in tokens
        n_reason = tokens.index(stop) if closed else len(tokens)
        answer = (
            b.tokenizer.decode(tokens[n_reason + 1 :], skip_special_tokens=True).strip()
            if closed
            else ""
        )
        output = parse_answer(answer, request.get("tools", []))
        usage = {
            "input_tokens": len(ids),
            "output_tokens": len(tokens),
            "total_tokens": len(ids) + len(tokens),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": n_reason},
        }
        event = dict(
            usage=usage,
            seconds=time.monotonic() - before,
            reasoning_closed=closed,
            tools=[t["name"] for t in output if t["type"] == "function_call"],
        )
        with (result_directory(OUT) / "provider.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        return {
            "id": "resp_" + uuid.uuid4().hex,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed" if closed and len(tokens) < limit else "incomplete",
            "error": None,
            "incomplete_details": None
            if len(tokens) < limit
            else {"reason": "max_output_tokens"},
            "model": "Qwen3.5-9B",
            "output": output,
            "usage": usage,
            "parallel_tool_calls": False,
            "tools": request.get("tools", []),
            "tool_choice": "auto",
        }


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            path = self.path.rstrip("/")
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if path == "/v1/pilot/preflight":
                data = self.server.engine.preflight(request)
                print(json.dumps({"recovery_preflight": data}), flush=True)
            elif path == "/v1/responses":
                data = self.server.engine.respond(request)
            else:
                raise ValueError("unknown Responses endpoint")
            code = 200
        except ContextLengthExceeded as exc:
            print(str(exc), flush=True)
            data = {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                    "input_tokens": exc.input_tokens,
                    "output_reserve": exc.output_reserve,
                    "context_window": exc.context_limit,
                }
            }
            code = 400
        except Exception as exc:
            import traceback

            traceback.print_exc()
            data = {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "request_failed",
                }
            }
            code = 400
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def smoke(engine):
    import tempfile
    import subprocess

    receipt = {}
    with tempfile.TemporaryDirectory(dir=ROOT / ".conda/tmp") as tmp:
        folder = Path(tmp)
        subprocess.run(["git", "init", "-q", tmp], check=True)
        nonce = "f61b3a90"
        (folder / "input.txt").write_text(nonce)
        tools = [
            {
                "type": "function",
                "name": "read_input",
                "description": "Read input.txt",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "write_result",
                "description": "Write result.txt",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        ]
        history = [
            {
                "role": "user",
                "content": "Use read_input first, then write_result to copy the value exactly.",
            }
        ]
        for expected in ("read_input", "write_result"):
            resp = engine.respond(
                {
                    "model": "Qwen3.5-9B",
                    "input": history,
                    "tools": tools,
                    "max_output_tokens": 1024,
                }
            )
            calls = [x for x in resp["output"] if x["type"] == "function_call"]
            if (
                len(calls) != 1
                or calls[0]["name"] != expected
                or not resp["usage"]["output_tokens_details"]["reasoning_tokens"]
            ):
                raise RuntimeError(f"thinking/tool smoke failed at {expected}")
            call = calls[0]
            if expected == "write_result":
                (folder / "result.txt").write_text(
                    json.loads(call["arguments"])["value"]
                )
            history.extend(
                [
                    call,
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": nonce,
                    },
                ]
            )
            receipt[expected] = resp["usage"]
        if (folder / "result.txt").read_text() != nonce:
            raise RuntimeError("tool observation was not used")
        write(result_directory(OUT) / "smoke-tools.json", receipt)
    context_smoke(engine)


def context_smoke(engine):
    # Near-limit prefill is confined to this smoke and excluded from formal trajectories.
    tokenizer = engine.backend.tokenizer
    sources = sorted((ROOT / "vendor/MemTrace/src").rglob("*.py"))
    material = "\n".join(p.read_text() for p in sources)
    target = int(
        os.environ.get("PILOT_SMOKE_INPUT_TOKENS", engine.context_limit - 3072)
    )
    tokens = tokenizer.encode(material, add_special_tokens=False)[:target]
    prompt = (
        "Reference source code follows.\n"
        + tokenizer.decode(tokens)
        + "\nEnd of source. Reply with the single word READY to acknowledge receipt."
    )
    resp = engine.respond(
        {
            "model": "Qwen3.5-9B",
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": 2048,
        }
    )
    answer = resp["output"][0].get("content", [{}])[0].get("text", "")
    if (
        resp["usage"]["input_tokens"] < target - 500
        or resp["status"] != "completed"
        or "READY" not in answer.upper()
    ):
        raise RuntimeError(f"invalid long-context probe {resp['usage']}")
    write(
        result_directory(OUT) / "smoke-context.json",
        {
            "usage": resp["usage"],
            "status": resp["status"],
            "context_window": engine.context_limit,
        },
    )


if __name__ == "__main__":
    import sys

    engine = Engine()
    if sys.argv[1] == "smoke":
        smoke(engine)
    elif sys.argv[1] == "context-smoke":
        context_smoke(engine)
    elif sys.argv[1] == "recovery-check":
        from recovery_check import replay

        replay(engine, generate=True)
    else:
        with socketserver.UnixStreamServer(sys.argv[1], Handler) as server:
            server.engine = engine
            server.serve_forever()
