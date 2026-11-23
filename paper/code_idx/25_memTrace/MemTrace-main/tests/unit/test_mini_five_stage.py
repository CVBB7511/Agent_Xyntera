from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memtrace.cli import main
from memtrace.config import load_config
from memtrace.harness.contracts import HarnessEventType
from memtrace.harness.mini_five_stage import (
    MiniSweAgentHarnessAdapter,
    RecoveryPreflightError,
    sanitize_mini_messages_for_api,
)
from memtrace.orchestration.execution_coordinator import _project_mini_epoch_handoff


class _FakeModel:
    def format_message(self, *, role: str, content: str) -> dict[str, str]:
        return {"role": role, "content": content}


class _FakeAgent:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.messages: list[dict[str, str]] = []
        self.config = type("Config", (), {"output_path": None})()
        self.n_calls = 0

    def add_messages(self, *messages: dict[str, str]) -> None:
        self.messages.extend(messages)

    def query(self) -> dict[str, object]:
        return {
            "content": "1. Inspect the failing test\n2. Implement the fix\n3. Run the tests",
            "extra": {"actions": []},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.messages), encoding="utf-8")


def test_recovery_wire_projects_repeated_route_but_preserves_test_provenance() -> None:
    milestone = {"target_outcome": "large task specification\n" * 6000}
    evidence = {
        "summary": "pytest tests/test_parser.py::test_escaping passed",
        "workspace_revision_id": "r1",
        "raw_detail_memory_ref": "memoryref_a",
        "entity_refs": ["test:tests/test_parser.py::test_escaping", "file:parser.py"],
    }
    record = {
        "knowledge_map": {
            "semantic_route": {"current_milestone": milestone, "goal": "repair parser"},
            "page_table": {"memory_ref": "memoryref_a"},
        },
        "working_route": {"current_milestone": milestone, "goal": "repair parser"},
        "direct_page_recovery": [{
            "memory_ref": "memoryref_a", "rendered_content": "complete pytest stdout",
            "status": "RESTORED", "entity_refs": evidence["entity_refs"],
        }],
        "established_facts": [evidence],
    }
    projected = _project_mini_epoch_handoff(record, references_only=True)
    assert json.dumps(projected).count("large task specification") == 6000
    assert "page_table" not in projected["knowledge_map"]
    assert projected["direct_page_recovery"][0]["memory_ref"] == "memoryref_a"
    assert projected["direct_page_recovery"][0]["status"] == "ADDRESS_ONLY"
    assert "rendered_content" not in projected["direct_page_recovery"][0]
    assert projected["established_facts"] == [evidence]
    assert record["direct_page_recovery"][0]["status"] == "RESTORED"


@pytest.mark.parametrize(
    ("replacement_accepted", "proactive"), [(True, False), (False, False), (None, False), (False, True)]
)
def test_physical_context_overflow_only_retries_a_fresh_history(
    tmp_path: Path, replacement_accepted: bool | None, proactive: bool,
) -> None:
    profile = (
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    agent = _FakeAgent()
    adapter = MiniSweAgentHarnessAdapter(
        repository_path=tmp_path,
        model="Qwen3.5-9B",
        run_root=tmp_path / "run",
        provider=load_config(profile).provider,
        runtime_factory=lambda **_: agent,
    )
    adapter.plan(
        user_task="Fix an input parser", planning_context="",
        run_id="fixture", revision_id="r1", workspace_receipt=None, inject=False,
    )
    adapter._runtime.pending_response = None
    agent.config.output_path = tmp_path / "current.json"
    agent.add_messages({"role": "user", "content": "stale overflowing history"})
    requests = []

    def query():
        requests.append(list(agent.messages))
        agent.n_calls += 1
        if (agent.n_calls == 1 and proactive) or (agent.n_calls == 2 and replacement_accepted):
            return {"content": "continuing", "extra": {"actions": []}}
        raise ValueError("context_length_exceeded: provider rejected the request")

    agent.query = query
    driver = adapter.build_driver(native_compaction_timeout_seconds=180, native_compaction_enabled=False)
    transport = driver.context_transport()
    stream = driver.events(
        user_task="fixture", run_id="fixture", branch_id="main",
        revision_tracker=SimpleNamespace(current=lambda: "r1"),
    )
    first = next(stream)
    assert first.event_type is HarnessEventType.TURN_STARTED
    if proactive:
        assert next(stream).event_type is HarnessEventType.ITEM_COMPLETED
        usage = next(stream)
        assert usage.event_type is HarnessEventType.TOKEN_USAGE_UPDATED
        transport.request_turn_fence(
            turn_id=first.turn_id,
            reason="PROVIDER_CONTEXT_LIMIT",
            source_event_id=usage.source_event_id,
        )
    else:
        overflow = next(stream)
        assert overflow.event_type is HarnessEventType.PHYSICAL_CONTEXT_FAILURE
        assert overflow.thread_id == first.thread_id
        assert overflow.payload["will_retry"] is False
        assert "context_length_exceeded" in overflow.payload["provider_error"]
    assert len(requests) == 1

    terminal = next(stream)
    assert terminal.event_type is HarnessEventType.TURN_COMPLETED
    assert terminal.payload["quiescence_reason"] == "PROVIDER_CONTEXT_LIMIT"
    if replacement_accepted is None:
        assert list(stream) == []
        assert len(requests) == 1
        return

    replacement = transport.start_replacement_thread()
    transport.submit_replacement(
        delivery_id="delivery", thread_id=replacement, context_digest="digest",
        rendered_content="Verified continuity checkpoint fixture",
    )
    second = next(stream)
    assert second.event_type is HarnessEventType.TURN_STARTED
    assert second.thread_id == replacement != first.thread_id
    transport.observe(second)
    event = next(stream)
    assert len(requests) == 2
    assert "stale overflowing history" not in json.dumps(requests[1])
    assert "Verified continuity checkpoint fixture" in json.dumps(requests[1])
    archived, = (tmp_path / "run/mini-swe-agent/epochs").glob("*.json")
    assert "stale overflowing history" in archived.read_text(encoding="utf-8")
    if replacement_accepted:
        assert event.event_type is HarnessEventType.ITEM_COMPLETED
        transport.observe(event)
        assert next(stream).event_type is HarnessEventType.TOKEN_USAGE_UPDATED
        assert next(stream).event_type is HarnessEventType.PHYSICAL_CONTEXT_FAILURE
        assert next(stream).event_type is HarnessEventType.TURN_COMPLETED
        assert list(stream) == []
        assert len(requests) == 3
    else:
        assert event.event_type is HarnessEventType.TURN_COMPLETED
        assert event.payload["quiescence_reason"] == "PROVIDER_CONTEXT_LIMIT"
        assert list(stream) == []
        assert len(requests) == 2


def test_real_mini_adapter_refuses_unfit_recovery_before_model_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("minisweagent")
    httpx = pytest.importorskip("httpx")
    litellm = pytest.importorskip("litellm")
    profile = (
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    provider = load_config(profile).provider
    monkeypatch.setenv(provider.api_key_env, "local-fixture")
    monkeypatch.setenv("MEMTRACE_MINI_TOKEN_PREFLIGHT", "1")
    adapter = MiniSweAgentHarnessAdapter(
        repository_path=tmp_path, model="Qwen3.5-9B", run_root=tmp_path / "run", provider=provider,
    )
    runtime = adapter._default_runtime_factory(task="fixture", trajectory_path=tmp_path / "trajectory.json")
    adapter._check_replacement_input = True
    preflights = []

    def preflight(url, *, json, **kwargs):
        preflights.append((url, json))
        return httpx.Response(400, json={"error": {"message": "context_length_exceeded: fixture"}})

    def generate(**kwargs):
        pytest.fail("rejected recovery reached model generation")

    monkeypatch.setattr(httpx, "post", preflight)
    monkeypatch.setattr(litellm, "responses", generate)
    with pytest.raises(RecoveryPreflightError, match="context_length_exceeded"):
        runtime.agent.query()
    assert len(preflights) == 1
    assert preflights[0][0].endswith("/pilot/preflight")
    assert preflights[0][1]["input"][0]["role"] == "system"
    assert any(tool["name"] == "recall_memory" for tool in preflights[0][1]["tools"])


def test_deepswe_five_stage_profile_matches_best_campaign() -> None:
    root = Path(__file__).parents[2]
    config = load_config(
        root / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    assert config.provider.model == "deepseek-v4-flash-0731"
    assert config.provider.api_key_env == "MEMTENSOR_DOMESTIC_API_KEY"
    assert config.provider.base_url == "https://api-int.memtensor.cn/v1"
    assert config.provider.model_context_window == 200_000
    assert config.provider.native_compaction_enabled is False
    assert config.codex_sandbox_mode == "danger-full-access"
    assert config.stages.planning is True
    assert config.stages.page_store is True
    assert config.stages.semantic_memory is True
    assert config.stages.recall is True
    assert config.stages.context_runtime is True
    assert config.stages.rich_graph is True
    assert config.page_policy.min_tokens == 2048
    assert config.page_policy.target_tokens == 6144
    assert config.page_policy.nominal_max_tokens == 8192
    assert config.page_policy.absolute_max_tokens == 16384
    assert config.context_budget.model_limit == 200_000
    assert config.recall_max_pages == 8
    assert config.recall_max_tokens == 8192
    assert config.rich_prefetch_budget == 4
    assert config.run_budget.max_execution_turns == 200
    assert config.run_budget.wall_clock_seconds == 14_400
    assert dict(config.acceptance_budgets) == {
        "weak_progress": 4,
        "no_progress": 2,
        "semantic_review_rounds": 2,
        "unclaimed_boundaries": 3,
    }
    assert config.engagement.mode == "full"


def test_mini_five_stage_projects_one_behavioral_plan(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    config = load_config(
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    adapter = MiniSweAgentHarnessAdapter(
        repository_path=repository,
        model="deepseek-v4-flash-0731",
        run_root=tmp_path / "run",
        provider=config.provider,
        reasoning_effort="high",
        runtime_factory=lambda **_: _FakeAgent(),
    )
    result = adapter.plan(
        user_task="Fix the reported failure and leave the tests passing.",
        planning_context="",
        run_id="run-fixture",
        revision_id="planning-revision",
        workspace_receipt=None,
        inject=False,
    )
    assert result.thread_id
    criteria = [
        criterion
        for milestone in result.plan.milestones
        for criterion in milestone.criteria
    ]
    assert criteria
    claim_types = {criterion.claim_type for criterion in criteria}
    assert "BEHAVIORAL" in claim_types
    assert "STRUCTURAL" not in claim_types
    driver = adapter.build_driver(
        native_compaction_timeout_seconds=180,
        native_compaction_enabled=False,
    )
    assert driver.capabilities().supports_plan_mode is True
    assert driver.capabilities().supports_native_compaction is False
    assert driver.capabilities().supports_context_replacement is True


def test_mini_harness_rejects_codex_only_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("Fix the reported failure.\n", encoding="utf-8")
    config = (
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    code = main(
        [
            "run",
            "--harness",
            "mini_swe_agent",
            "--multilang-plan",
            "--repository",
            str(repository),
            "--task-file",
            str(task),
            "--run-root",
            str(tmp_path / "run"),
            "--config",
            str(config),
        ]
    )
    assert code == 2
    assert "only available with the Codex Harness" in capsys.readouterr().err


def test_memtensor_replay_drops_rejected_response_fields() -> None:
    cleaned = sanitize_mini_messages_for_api(
        [
            {
                "object": "response",
                "created_at": 1,
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "look around"}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": "{}",
                        "caller": {"type": "direct"},
                        "namespace": "functions",
                    },
                ],
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
    )
    assert all(item.get("object") != "response" and "created_at" not in item for item in cleaned)
    calls = [item for item in cleaned if item.get("type") == "function_call"]
    assert len(calls) == 1
    assert "caller" not in calls[0]
    assert "namespace" not in calls[0]
    assert calls[0]["call_id"] == "call_1"
    assert any(item.get("role") == "assistant" for item in cleaned)
    assistant = next(item for item in cleaned if item.get("role") == "assistant")
    assert assistant["content"][0]["type"] == "input_text"
