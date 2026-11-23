"""Regression tests for compaction, validity, budgets and grading."""

import sys
import tempfile
import unittest
import json
import struct
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from byl_gpu import idle_a800
from experiment import Limits, Run, prepare_reference
from fixtures import make_demo
from model import Completion, ScriptedModel, chat_ids, check_snapshot
from task import Task, parse_result


class SubmitBModel(ScriptedModel):
    """Replay the observed early-B decision independently of the new policy."""

    def __init__(self):
        self.final_state = None
        self.final_summary = None

    def complete(self, messages, limit, purpose):
        if purpose == "compact":
            return super().complete(messages, limit, purpose)
        state = json.loads(messages[-1]["content"])
        if state["phase"] == 1:
            return Completion('{"action":"continue"}', 8, 0)
        self.final_state = state
        self.final_summary = next(json.loads(item["content"])["working_summary"] for item in messages
                                  if item["role"] == "user" and item["content"].startswith('{"working_summary":'))
        return Completion('{"action":"submit","candidate":"B"}', 14, 0)


class PilotTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.limits = Limits(8192, 6144, 1024, 110_000, 12, 90, 10, 8)

    def fixture(self, scenario):
        return Task(make_demo(self.root / scenario, scenario), sys.executable)

    def test_two_boundaries_share_initial_prefix_and_cache_avoids_execution(self):
        task = self.fixture("unrelated")
        initial, truth = prepare_reference(task, self.limits)
        summary, cache = [
            Run(task, arm, ScriptedModel(), self.limits, self.root / arm, initial).execute(truth)
            for arm in ("summary", "cache")
        ]
        for result in (summary, cache):
            replay = [c for c in result["compactions"] if c["reason"] == "length_replay"]
            self.assertEqual([c["phase"] for c in replay], [1, 2])
            self.assertTrue(all(c["before_tokens"] >= 6144 and c["raw_messages_kept"] == 0 for c in replay))
            self.assertTrue(result["passed"])
        self.assertEqual(summary["compactions"][0], cache["compactions"][0])
        self.assertEqual(summary["repeat_executions"], 4)
        self.assertEqual(cache["physical_tests"], 0)

    def test_independent_blocker_changes_which_test_is_recommended(self):
        for scenario in ("fail-to-pass", "independent-blocker"):
            with self.subTest(scenario=scenario):
                task = self.fixture(scenario)
                initial, _ = prepare_reference(task, self.limits)
                run = Run(task, "selective", ScriptedModel(), self.limits, self.root / f"run-{scenario}", initial)
                run.epoch = "E1"
                self.assertIsNone(run.current("A", "offset"))
                frontier = run.frontier()
                triage = {(o["candidate"], o["test"]) for o in frontier["checks_for_unblocked_candidates"]}
                stale = {(o["candidate"], o["test"]) for o in frontier["stale_checks"]}
                self.assertIn(("A", "offset"), stale)
                self.assertEqual(("A", "offset") in triage, scenario == "fail-to-pass")
                if scenario == "independent-blocker":
                    self.assertTrue(any(row["candidate"] == "A" and row["test"] == "offset"
                                        and row["reason"] == "independent_failure"
                                        for row in frontier["deferred_checks"]))

    def test_no_test_budget_still_permits_valid_cache_hits(self):
        task = self.fixture("fail-to-pass")
        initial, truth = prepare_reference(task, self.limits)
        limits = Limits(8192, 6144, 1024, 110_000, 0, 90, 10, 8)
        result = Run(task, "cache", ScriptedModel(), limits, self.root / "run-zero", initial).execute(truth)
        self.assertTrue(result["passed"])
        self.assertEqual(result["physical_tests"], 0)
        self.assertEqual(result["cache_hits"], 3)
        self.assertEqual(result["reopening_opportunities_e1"], 1)
        self.assertEqual(result["reopening_checks_e1"], 0)

    def test_submission_gate_reopens_preferred_candidate_from_public_checks(self):
        task = self.fixture("fail-to-pass")
        initial, truth = prepare_reference(task, self.limits)
        model = SubmitBModel()
        visible = Run(task, "eager-visible", model, self.limits, self.root / "visible", initial).execute(truth)
        self.assertEqual(model.final_state["current_candidate_evidence"]["A"]["passed"]["offset"], "E1:A:offset")
        self.assertEqual(visible["candidate"], "B")
        gate = Run(task, "guarded", SubmitBModel(), self.limits, self.root / "gate", initial)
        guarded = gate.execute(truth)
        self.assertEqual(guarded["submission_gate"]["proposal"], "B")
        self.assertEqual(guarded["candidate"], "A")
        self.assertTrue(guarded["selected_preferred_viable"])
        self.assertEqual(guarded["reopening_checks_missed_e1"], 0)
        self.assertEqual([(o.candidate, o.test) for o in gate.executions], [("A", "offset"), ("A", "preserve")])
        for arm in (visible, guarded):
            self.assertEqual([c["phase"] for c in arm["compactions"] if c["reason"] == "length_replay"], [1, 2])

    def test_submission_gate_preserves_independent_blocker_and_budget(self):
        task = self.fixture("independent-blocker")
        initial, truth = prepare_reference(task, self.limits)
        result = Run(task, "guarded", SubmitBModel(), self.limits, self.root / "blocked", initial).execute(truth)
        self.assertEqual(result["candidate"], "B")
        self.assertEqual(result["independent_blocker_retests_e1"], 0)

        task = self.fixture("fail-to-pass")
        initial, truth = prepare_reference(task, self.limits)
        # The captured B submission cited all three checks, including an E0
        # check collected during the original run rather than at startup.
        initial.append(task.run("E0", "B", "empty", self.limits.test_timeout))
        limits = replace(self.limits, max_tests=0)
        fallback = Run(task, "guarded", SubmitBModel(), limits, self.root / "fallback", initial).execute(truth)
        self.assertEqual(fallback["physical_tests"], 0)
        self.assertEqual(fallback["submission_gate"]["reason"], "verified_fallback")
        self.assertEqual(fallback["submission_gate"]["unresolved_higher_priority"], ["A"])
        self.assertEqual(fallback["candidate"], "B")

    def test_full_public_evidence_can_be_shown_without_changing_the_models_choice(self):
        task = self.fixture("fail-to-pass")
        initial, truth = prepare_reference(task, self.limits)
        model = SubmitBModel()
        run = Run(task, "verified-visible", model, self.limits, self.root / "verified-visible", initial)
        result = run.execute(truth)
        self.assertEqual(model.final_state["current_candidate_evidence"]["A"]["status"], "verified")
        self.assertEqual(result["candidate"], "B")
        self.assertIsNone(result["submission_gate"])

        explicit_model = SubmitBModel()
        explicit = Run(task, "verified-explicit", explicit_model, self.limits, self.root / "verified-explicit", initial)
        explicit_result = explicit.execute(truth)
        state = explicit_model.final_state["current_candidate_evidence"]
        self.assertEqual(state["A"]["current_pass_evidence"]["offset"], "E1:A:offset")
        self.assertIn("empty", state["B"]["tests_without_current_result"])
        self.assertNotIn("pending", state["B"])
        self.assertEqual([(o.candidate, o.test) for o in explicit.executions],
                         [(o.candidate, o.test) for o in run.executions])
        self.assertEqual(explicit_result["candidate"], "B")
        self.assertIsNone(explicit_result["submission_gate"])

        scoped_model = SubmitBModel()
        scoped = Run(task, "verified-scoped", scoped_model, self.limits, self.root / "verified-scoped", initial)
        scoped_result = scoped.execute(truth)
        self.assertEqual(scoped_model.final_state["current_candidate_evidence"], state)
        self.assertTrue(scoped_model.final_summary.startswith("Historical summary."))
        self.assertTrue(scoped_model.final_summary.endswith(explicit_model.final_summary))
        self.assertEqual(scoped_result["candidate"], "B")
        self.assertIsNone(scoped_result["submission_gate"])

    def test_submission_gate_withholds_when_no_candidate_can_be_verified(self):
        task = self.fixture("pass-to-fail")
        initial, truth = prepare_reference(task, self.limits)
        limits = replace(self.limits, max_tests=0)
        result = Run(task, "guarded", SubmitBModel(), limits, self.root / "withheld", initial).execute(truth)
        self.assertEqual(result["stop"], "submission_blocked")
        self.assertIsNone(result["candidate"])
        self.assertEqual(result["submission_gate"]["physical_tests"], 0)

    def test_same_submission_rule_stops_after_confirming_a_blocker(self):
        task = self.fixture("relevant-stable")
        initial, _ = prepare_reference(task, self.limits)
        initial.extend(task.run("E0", c, t, 10) for c, t in (("A", "preserve"), ("B", "empty")))
        gates = []
        for arm in ("guarded", "guarded-eager"):
            run = Run(task, arm, ScriptedModel(), self.limits, self.root / arm, initial)
            run.epoch, run.phase = "E1", 2
            self.assertEqual(run.check_submission("B"), "B")
            gates.append(run.submission_gate)
        self.assertEqual(gates[0]["incoming_sha256"], gates[1]["incoming_sha256"])
        self.assertEqual([gate["physical_tests"] for gate in gates], [1, 2])

    def test_silent_shared_boundary_restores_observations_and_consumed_budget(self):
        class PrefixModel(SubmitBModel):
            def __init__(self):
                super().__init__()
                self.compaction_inputs = []

            def complete(self, messages, limit, purpose):
                if purpose == "compact":
                    self.compaction_inputs.extend(item["content"] for item in messages)
                else:
                    state = json.loads(messages[-1]["content"])
                    if (state["phase"], state["turn"]) == (1, 0):
                        return Completion('{"action":"verify","candidate":"B","test":"empty"}', 15, 0)
                return super().complete(messages, limit, purpose)

        task = self.fixture("fail-to-pass")
        limits = replace(self.limits, max_tests=1)
        initial, truth = prepare_reference(task, limits)
        prefix_model = PrefixModel()
        prefix = Run(task, "cache", prefix_model, limits, self.root / "prefix", initial, announce_change=False)
        prefix.prepare_boundary()
        checkpoint = json.dumps(prefix.save_boundary())
        self.assertNotIn("The environment is now E1", "\n".join(prefix_model.compaction_inputs))
        for arm in ("verified-explicit", "verified-scoped"):
            with self.subTest(arm=arm):
                model = SubmitBModel()
                branch = Run(task, arm, model, limits, self.root / arm, initial, announce_change=False)
                state = json.loads(checkpoint)
                result = branch.execute(truth, boundary=state, boundary_digest="shared", prefix_seconds=1)
                self.assertEqual(result["boundary_sha256"], "shared")
                self.assertEqual(result["physical_tests"], 1)
                self.assertGreater(result["model_input_tokens"], state["token_usage"][0])
                self.assertGreaterEqual(result["run_wall_seconds"], 1)
                self.assertNotIn("epoch", model.final_state)
                self.assertEqual(model.final_state["physical_tests_remaining"], 0)
                self.assertEqual(branch.current("B", "empty").epoch, "E0")
                self.assertEqual(result["candidate"], "B")

        prompts = []
        for notify in (False, True):
            model = SubmitBModel()
            branch = Run(task, "verified-explicit", model, limits, self.root / f"notice-{notify}", initial,
                         announce_change=False)
            result = branch.execute(truth, boundary=json.loads(checkpoint), boundary_digest="shared",
                                    post_boundary_notice=notify)
            prompts.append(result["first_decision_prompt"])
            self.assertEqual(prompts[-1]["notice_present"], notify)
            self.assertEqual(result["continuation_physical_tests"], 0)
            self.assertNotIn("epoch", model.final_state)
        self.assertEqual(prompts[0]["without_notice_sha256"], prompts[1]["without_notice_sha256"])
        self.assertNotEqual(prompts[0]["sha256"], prompts[1]["sha256"])

    def test_pruning_skips_only_checks_on_a_candidate_with_an_independent_blocker(self):
        for scenario in ("independent-blocker", "fail-to-pass"):
            with self.subTest(scenario=scenario):
                task = self.fixture(scenario)
                initial = [task.run("E0", candidate, test, self.limits.test_timeout)
                           for candidate, test in task.initial]
                run = Run(task, "guarded-pruned", ScriptedModel(), self.limits, self.root / f"pruned-{scenario}", initial)
                run.epoch, run.phase = "E1", 2
                response = run.action({"action": "verify", "candidate": "A", "test": "offset"})
                if scenario == "independent-blocker":
                    self.assertEqual(response["status"], "verification_skipped")
                    self.assertEqual(response["blockers"], {"empty": "E0:A:empty"})
                    self.assertEqual(len(run.executions), 0)
                else:
                    self.assertEqual(response["status"], "passed")
                    self.assertEqual(len(run.executions), 1)

    def test_guard_does_not_reorder_candidates_without_explicit_priority(self):
        task = self.fixture("fail-to-pass")
        del task.config["candidate_preference"]
        initial = [task.run("E0", candidate, test, self.limits.test_timeout) for candidate, test in task.initial]
        run = Run(task, "guarded", ScriptedModel(), self.limits, self.root / "no-priority", initial)
        run.epoch, run.phase = "E1", 2
        self.assertEqual(run.check_submission("B"), "B")
        self.assertEqual(run.submission_gate["reason"], "proposal_verified")
        self.assertEqual([(o.candidate, o.test) for o in run.executions], [("B", "empty")])

    def test_result_contract(self):
        self.assertEqual(parse_result(0, '{"status":"not_collected"}')[0], "not_collected")
        self.assertEqual(parse_result(0, '{"status":[]}')[0], "invalid_result")
        self.assertEqual(parse_result(1, '{"status":"passed"}')[0], "execution_error")

    def test_weight_check_rejects_truncated_payload(self):
        (self.root / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
        name = "model-00001.safetensors"
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": name}}), encoding="utf-8",
        )
        header = b'{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}'
        shard = self.root / name
        shard.write_bytes(struct.pack("<Q", len(header)) + header + b"AB")
        self.assertEqual(check_snapshot(self.root)["shards"][name], shard.stat().st_size)
        with shard.open("r+b") as stream:
            stream.truncate(shard.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            check_snapshot(self.root)

    def test_chat_template_uses_flat_token_ids_and_disables_thinking(self):
        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, return_dict,
                                    add_generation_prompt, enable_thinking):
                self.options = tokenize, return_dict, add_generation_prompt, enable_thinking
                return [1, 2, 3]

        tokenizer = Tokenizer()
        self.assertEqual(chat_ids(tokenizer, [{"role": "user", "content": "hi"}]), [1, 2, 3])
        self.assertEqual(tokenizer.options, (True, False, True, False))

    def test_byl_gate_stops_with_one_idle_card_or_unknown_occupancy(self):
        hardware = "0, GPU-a, NVIDIA A800, 8, 0\n1, GPU-b, NVIDIA A800, 4, 0\n"
        with patch("byl_gpu.query", side_effect=[hardware, ""]):
            self.assertEqual(idle_a800(), "GPU-b")
        with patch("byl_gpu.query", side_effect=[hardware, "GPU-b\n"]):
            with self.assertRaisesRegex(ValueError, "at least two idle GPUs"):
                idle_a800()
        with patch("byl_gpu.query", side_effect=[hardware, "Not Supported\n"]):
            with self.assertRaisesRegex(ValueError, "compute occupancy"):
                idle_a800()


if __name__ == "__main__":
    unittest.main()
