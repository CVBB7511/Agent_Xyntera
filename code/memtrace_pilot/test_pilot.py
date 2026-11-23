import http.client
import http.server
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pilot
import reporting
from benchdata import prefix
from common import (
    CONTEXT_WINDOWS,
    DEEP,
    LEGACY_CONTEXT_WINDOW,
    OUTPUT_RESERVE,
    SK,
    write,
)
from evo_evaluator import cached_evaluation_image
from gpu import GPUUnavailable, select_recovery_gpu, watch_recovery
from provider import Engine, Handler, messages, parse_answer


class PilotTests(unittest.TestCase):
    def test_official_image_alias_uses_only_the_frozen_local_image(self):
        images = Mock()
        source = "example/evaluator"
        pinned = source + "@sha256:" + "a" * 64
        self.assertIsNone(
            cached_evaluation_image(
                images, source, "other", source=source, pinned=pinned
            )
        )
        self.assertIsNone(
            cached_evaluation_image(
                images, "example/unrelated", None, source=source, pinned=pinned
            )
        )
        images.get.assert_not_called()
        self.assertIs(
            cached_evaluation_image(images, source, None, source=source, pinned=pinned),
            images.get.return_value,
        )
        images.get.assert_called_once_with(pinned)
        images.pull.assert_not_called()

    def test_recovery_counts_and_selects_cards_by_memory(self):
        inventory = "\n".join(
            [
                "0, GPU-a, A800, 12000, 80000",
                "1, GPU-b, A800, 1100, 80000",
                "2, GPU-c, A800, 60000, 80000",
                "3, GPU-d, A800, 59000, 80000",
            ]
        )
        query = Mock(return_value=inventory)
        self.assertEqual(select_recovery_gpu(query).uuid, "GPU-b")
        query.assert_called_once_with(
            "--query-gpu=index,uuid,name,memory.used,memory.total"
        )
        with self.assertRaisesRegex(ValueError, "found 1"):
            select_recovery_gpu(
                Mock(
                    return_value=inventory.replace("12000", "60000").replace(
                        "59000", "60000"
                    ),
                )
            )
        with self.assertRaisesRegex(
            ValueError, "no A800 with enough allocator headroom"
        ):
            select_recovery_gpu(
                Mock(
                    return_value=inventory.replace("12000", "58000").replace(
                        "1100", "58000"
                    ),
                )
            )

    def test_recovery_monitors_only_memory_below_threshold(self):
        process = Mock()
        process.poll.side_effect = [None, None, 0]
        record = {}
        with patch("gpu.subprocess.check_output", return_value="61439, 81920") as query:
            watch_recovery(process, "GPU-fixture", record)
        query.assert_called_once()
        self.assertIn("--query-gpu=memory.used,memory.total", query.call_args.args[0])
        self.assertEqual(record, {"peak_memory_mib": 61439})

    def test_job_stops_at_75_percent_of_gpu_memory(self):
        process = Mock()
        process.poll.return_value = None
        record = {}
        with patch("gpu.subprocess.check_output", return_value="61440, 81920"):
            with self.assertRaisesRegex(
                GPUUnavailable, "GPU memory guard stopped the job"
            ):
                watch_recovery(process, "GPU-fixture", record)
        self.assertIn("resource_stop", record)
        process.wait.assert_not_called()

    def test_memory_timeout_and_empty_sample_are_retried(self):
        process = Mock()
        process.poll.return_value = None
        stop = threading.Event()
        record = {}

        def finish_after_sample(*args, **kwargs):
            if "peak_memory_mib" in record:
                stop.set()
            else:
                raise subprocess.TimeoutExpired("provider", 0.25)

        process.wait.side_effect = finish_after_sample
        with patch(
            "gpu.subprocess.check_output",
            side_effect=[
                subprocess.TimeoutExpired("nvidia-smi", 2),
                "",
                "20783, 81920",
            ],
        ) as query:
            watch_recovery(process, "GPU-fixture", record, stop=stop)
        self.assertEqual(
            [call.kwargs["timeout"] for call in query.call_args_list], [2, 3, 4]
        )
        self.assertEqual(record, {"memory_query_failures": 2, "peak_memory_mib": 20783})

    def test_consecutive_missing_samples_are_recorded_as_telemetry_loss(self):
        process = Mock()
        process.poll.return_value = None
        record = {}
        with patch("gpu.subprocess.check_output", return_value="") as query:
            with self.assertRaisesRegex(GPUUnavailable, "after 3 consecutive checks"):
                watch_recovery(process, "GPU-fixture", record)
        self.assertEqual(query.call_count, 3)
        self.assertEqual(record["resource_stop_kind"], "memory_telemetry_unavailable")

    def test_archive_only_unsubmitted_gpu_telemetry_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dest = root / "runs/dask/native"
            attempt = {"id": "a" * 32}
            write(dest / "attempt.json", attempt)
            write(
                dest / "gpu.json",
                {"state": "finished", "resource_stop": "nvidia-smi timed out"},
            )
            write(dest / "grade.json", {"metrics": {"Resolved": 0}})
            with self.assertRaisesRegex(RuntimeError, "submission or score"):
                pilot.archive_monitor_interruption(root, "dask", "native")
            (dest / "grade.json").unlink()
            pilot.archive_monitor_interruption(root, "dask", "native")
            archive = root / "attempt-history/dask/native" / attempt["id"]
            self.assertTrue((archive / "attempt.json").is_file())
            self.assertFalse(dest.exists())

    def test_provider_rejects_overflow_before_inference_with_structured_error(self):
        engine = Engine.__new__(Engine)
        engine.context_limit = CONTEXT_WINDOWS[-1]
        engine.backend = SimpleNamespace(
            tokenizer=SimpleNamespace(
                apply_chat_template=lambda *a, **kw: range(137_676)
            )
        )
        with http.server.HTTPServer(("127.0.0.1", 0), Handler) as server:
            server.engine = engine

            def request(path, **options):
                thread = threading.Thread(target=server.handle_request, daemon=True)
                thread.start()
                conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                try:
                    body = {"model": "Qwen3.5-9B", "input": [], **options}
                    conn.request("POST", path, json.dumps(body))
                    response = conn.getresponse()
                    return response.status, json.load(response)
                finally:
                    conn.close()
                    thread.join(timeout=5)

            status, count = request("/v1/pilot/preflight")
            self.assertEqual(status, 200)
            self.assertEqual(count["input_tokens"], 137_676)
            self.assertEqual(count["context_window"], CONTEXT_WINDOWS[-1])
            engine.context_limit = LEGACY_CONTEXT_WINDOW
            status, body = request("/v1/pilot/preflight")
            self.assertEqual(status, 400)
            error = body["error"]
            self.assertEqual(error["code"], "context_length_exceeded")
            status, body = request("/v1/responses")
            error = body["error"]
            self.assertEqual(status, 400)
            self.assertEqual(error["code"], "context_length_exceeded")
            self.assertEqual(error["input_tokens"], 137_676)
            self.assertEqual(error["context_window"], LEGACY_CONTEXT_WINDOW)
            self.assertEqual(error["output_reserve"], OUTPUT_RESERVE)
            with patch("traceback.print_exc"):
                status, body = request("/v1/responses", max_output_tokens=0)
            self.assertEqual((status, body["error"]["code"]), (400, "request_failed"))

    def test_only_documented_unsubmitted_overflow_is_archived_once(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pilot, "OUT", Path(d)):
            dest = Path(d) / "runs" / SK / "memtrace"
            stamp = {"id": pilot.LEGACY_OVERFLOW_ATTEMPT, "seconds": 487}
            write(dest / "attempt.json", stamp)
            write(
                dest / "trace/memtrace/result.json",
                {
                    "completion_verdict": "INCOMPLETE",
                    "metrics": {
                        "provider_turn_failure_retry_count": 5,
                        "provider_turn_failures_exhausted_count": 1,
                    },
                },
            )
            submitted = dest / "nodes/M06/snapshot.json"
            write(submitted, {})
            with self.assertRaisesRegex(RuntimeError, "has official submissions"):
                pilot.archive_legacy_overflow()
            submitted.unlink()
            pilot.archive_legacy_overflow()
            archive = Path(d) / "attempt-history" / SK / "memtrace" / stamp["id"]
            self.assertTrue((archive / "attempt.json").exists())
            self.assertFalse(dest.exists())
            write(dest / "attempt.json", {"id": "replacement"})
            pilot.archive_legacy_overflow()
            self.assertTrue((dest / "attempt.json").exists())

    def test_prefix_retains_ungraded_prerequisite(self):
        self.assertEqual(
            prefix({"A", "B", "C", "D"}, [("D", "A"), ("A", "B")], {"D"}),
            ["C", "D", "A", "B"],
        )

    def test_tool_roundtrip_preserves_shell_command_and_structured_argument(self):
        schema = [
            {
                "name": "recall",
                "parameters": {"properties": {"pages": {"type": "array"}}},
            }
        ]
        call = parse_answer(
            "<tool_call><function=recall><parameter=pages>[1,2]</parameter></function></tool_call>",
            schema,
        )[0]
        turns = messages(
            [
                {"role": "user", "content": "lookup"},
                call,
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "found",
                },
            ]
        )
        self.assertEqual(
            turns[1]["tool_calls"][0]["function"]["arguments"]["pages"], [1, 2]
        )
        self.assertEqual(
            turns[2], {"role": "tool", "name": "recall", "content": "found"}
        )

    @patch.dict(os.environ, {"PILOT_CONTEXT_WINDOW": str(LEGACY_CONTEXT_WINDOW)})
    def test_absent_judge_is_distinct_from_zero(self):
        import csv

        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(reporting, "OUT", Path(d)),
        ):
            root = Path(d)
            write(root / "inputs/evo-selection.json", ["x", "y"])
            dest = root / "runs" / DEEP[0] / "native"
            write(dest / "attempt.json", {"seconds": 20})
            write(dest / "grade.json", {"metrics": {"pass@1": 0}})
            reporting.report()
            with (root / "summary.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(
                (rows[0]["scored_runs"], rows[0]["subset_pct"]), ("1", "0.0")
            )
            self.assertEqual((rows[1]["scored_runs"], rows[1]["subset_pct"]), ("0", ""))

    @patch.dict(os.environ, {"PILOT_CONTEXT_WINDOW": str(LEGACY_CONTEXT_WINDOW)})
    def test_report_counts_only_activated_replacements(self):
        import csv

        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(reporting, "OUT", Path(d)),
        ):
            root = Path(d)
            write(root / "inputs/evo-selection.json", [])
            trace = root / "runs" / SK / "memtrace" / "trace"
            trace.mkdir(parents=True)
            with closing(sqlite3.connect(trace / "run.sqlite")) as db, db:
                db.execute(
                    "create table context_epochs (ordinal int, activated_at text)"
                )
                db.executemany(
                    "insert into context_epochs values (?, ?)",
                    [(0, "initial"), (1, "observed"), (2, None)],
                )
            reporting.report()
            with (root / "evidence.csv").open(encoding="utf-8-sig") as f:
                row = next(
                    r
                    for r in csv.DictReader(f)
                    if r["task"] == SK and r["arm"] == "memtrace"
                )
            self.assertEqual(row["context_refreshes"], "1")


if __name__ == "__main__":
    unittest.main()
