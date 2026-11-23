"""Replay one replacement request using the previous Milestone failure as input."""

import json
import sqlite3
import sys
from contextlib import closing
from types import SimpleNamespace

from common import (
    MODEL,
    OUT,
    OUTPUT_RESERVE,
    ROOT,
    SK,
    context_window,
    result_directory,
    write,
)
from provider import ContextLengthExceeded, Engine

FAILED_TRIAL = OUT / "runs" / SK / "memtrace" / "trace/memtrace"


def replay(engine, *, generate=False):
    sys.path.insert(0, str(ROOT / "vendor/MemTrace/src"))
    from memtrace.context_runtime.lifecycle import ContextLifecycle
    from memtrace.contracts import digest
    from memtrace.harness.context_transport import CodexContextTransport
    from memtrace.harness.mini_five_stage import (
        _SYSTEM_TEMPLATE,
        lite_llm_memory_tools,
        mini_response_tools,
    )
    from memtrace.orchestration.execution_coordinator import (
        _render_epoch_replacement_context,
    )
    from memtrace.page_store.synopsis import memory_ref_for_page

    db = FAILED_TRIAL / "v2-state.sqlite3"
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        checkpoint = json.loads(
            conn.execute(
                "select checkpoint_json from v2_continuity_checkpoints order by rowid desc limit 1"
            ).fetchone()[0]
        )
        image = ContextLifecycle._decode_image(
            json.dumps(checkpoint["resident_context_image"])
        )
        original = _render_epoch_replacement_context(checkpoint, image)
        packed = _render_epoch_replacement_context(checkpoint, image, compact_mini=True)
        estimate = CodexContextTransport.replacement_provider_token_estimate
        safe_budget = engine.context_limit - 8_000 - 16_000 - OUTPUT_RESERVE - 8_000
        use_refs = estimate(packed) > safe_budget
        if use_refs:
            packed = _render_epoch_replacement_context(
                checkpoint,
                image,
                compact_mini=True,
                references_only=True,
            )
        if estimate(packed) > safe_budget:
            raise RuntimeError("recovery payload still exceeds the coordinator budget")
        facts = json.loads(packed)["execution_handoff"]
        pages = {
            p["memory_ref"]: p["page_id"]
            for p in facts["direct_page_recovery"]
            if p.get("memory_ref")
        }
        addresses = {
            ref: conn.execute(
                "select payload_digest, storage_digest, storage_path from v2_pages where page_id=?",
                (page_id,),
            ).fetchone()
            for ref, page_id in pages.items()
        }
    page_root = (FAILED_TRIAL / "page-store").resolve()
    verified_pages = {}
    for ref, address in addresses.items():
        if address is None or memory_ref_for_page(pages[ref], address[0]) != ref:
            raise RuntimeError(
                "recovered MemoryRef does not resolve to its archived Page"
            )
        stored = (page_root / address[2]).resolve()
        if not stored.is_relative_to(page_root):
            raise RuntimeError("recovered Page path is outside the archived Page Store")
        content = stored.read_bytes()
        if digest({"bytes": content.hex()}) != address[1]:
            raise RuntimeError("recovered Page failed the storage integrity check")
        verified_pages[ref] = len(json.loads(content)["groups"])
    if not verified_pages:
        raise RuntimeError("recovered handoff has no verifiable continuity reference")

    envelope = CodexContextTransport.render_replacement_payload(
        delivery_id="delivery_" + "0" * 32,
        context_digest=image.image_digest,
        rendered_content=packed,
    )
    task = (
        "This is a single-turn recovery check. Locate a memory_ref in direct_page_recovery "
        "in the injected checkpoint. Call recall_memory once with that exact memory_ref, "
        "an entity_refs value from the same entry, and temporal_purpose=memory_ref_detail. "
        "Ask for the previous observation behind that anchor. Do not edit or run commands."
    )
    request = {
        "model": "Qwen3.5-9B",
        "input": [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": envelope},
            {"role": "user", "content": task},
        ],
        "tools": mini_response_tools(lite_llm_memory_tools()),
    }
    receipt = {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "old_estimated_tokens": estimate(original),
        "new_estimated_tokens": estimate(packed),
        "coordinator_budget": safe_budget,
        "references_only": use_refs,
        "verified_pages": verified_pages,
        "exact_request": engine.preflight(request),
    }
    if not generate:
        too_large = {
            **request,
            "input": [
                *request["input"],
                {"role": "user", "content": " replay" * engine.context_limit},
            ],
        }
        try:
            engine.preflight(too_large)
        except ContextLengthExceeded as exc:
            receipt["overflow_rejected_input_tokens"] = exc.input_tokens
        else:
            raise RuntimeError(
                "exact tokenizer accepted an intentionally oversized replay"
            )
    results = result_directory(OUT)
    if not generate:
        write(results / "recovery-preflight.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if generate:
        response = engine.respond({**request, "max_output_tokens": 2048})
        calls = [
            {
                "name": item["name"],
                "memory_ref": json.loads(item["arguments"]).get("memory_ref"),
            }
            for item in response["output"]
            if item["type"] == "function_call"
        ]
        found = {
            c["memory_ref"]: verified_pages[c["memory_ref"]]
            for c in calls
            if c["name"] == "recall_memory" and c["memory_ref"] in verified_pages
        }
        passed = len(calls) == 1 and bool(found)
        result = {
            "passed": passed,
            "calls": calls,
            "resolved_page_groups": found,
            "usage": response["usage"],
        }
        write(results / "recovery-generation.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not passed:
            raise RuntimeError(
                "first recovery response did not recall a checkpoint MemoryRef"
            )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    engine = Engine.__new__(Engine)
    engine.context_limit = context_window()
    engine.backend = SimpleNamespace(
        tokenizer=AutoTokenizer.from_pretrained(MODEL, local_files_only=True),
    )
    replay(engine)
