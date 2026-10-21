"""验证 RSL 的边界及数据隔离；假模型使测试不依赖联网或显卡。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsl import (
    DEFAULT_HOLDOUT,
    DEFAULT_TRAIN,
    ExpressionError,
    MemoryStore,
    OllamaClient,
    RSLSystem,
    benchmark,
    build_parser,
    fraction_text,
    parse_answer,
    parse_task,
)


def response(answer, reflection="重新检查符号和进位。"):
    return json.dumps({"answer": answer, "reflection": reflection}, ensure_ascii=False)


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, schema=None):
        self.calls.append([dict(message) for message in messages])
        return self.responses.pop(0)


class CorrectClient:
    """这是测试桩，仅供验证流程；真正推理使用 OllamaClient。"""
    model = "deterministic-test-stub"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, schema=None):
        self.calls.append([dict(message) for message in messages])
        if schema is not None:
            allowed = json.loads(messages[1]["content"])["allowed_rules"]
            return json.dumps({"lesson_ids": list(allowed)})
        expression = messages[1]["content"].split("计算表达式：", 1)[1]
        return response(fraction_text(parse_task(expression).value))


class RSLTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parent / ".work" / "test-rsl"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.base = Path(self.temporary.name)
        self.assertTrue(self.base.resolve().is_relative_to(temporary_root.resolve()))
        self.memory = MemoryStore(self.base / "memory.json")
        self.traces = self.base / "traces.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def system(self, client):
        return RSLSystem(client, self.memory, self.traces)

    def test_fraction_and_precedence_are_exact(self):
        self.assertEqual(fraction_text(parse_task("(81 / 7) - (5 / 3)").value), "208/21")
        self.assertEqual(fraction_text(parse_task("2 + 3 * 4").value), "14")
        self.assertEqual(fraction_text(parse_task("-23 * (14 - 31) + 7").value), "398")

    def test_unsafe_and_oversized_expressions_rejected(self):
        expressions = [
            '__import__("os").system("echo unsafe")', "open('a')", "x + 1",
            "True + 2", "1.2 + 3", "2 ** 100", "[1][0]", "1 << 2",
            "9" * 101, "1+" * 300 + "1", "-" * 20 + "1",
        ]
        client = QueueClient([])
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ExpressionError):
                self.system(client).solve(expression)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.memory.experiences, [])

    def test_zero_division_rejected_before_model(self):
        client = QueueClient([])
        for expression in ("1 / 0", "4 / (2 - 2)"):
            with self.assertRaises(ExpressionError):
                self.system(client).solve(expression)
        with self.assertRaises(ValueError):
            parse_answer("1/0")
        self.assertFalse(client.calls)

    def test_reject_wrong_or_invalid_answer_and_bound_calls(self):
        client = QueueClient(["not json", response(5), response("5")])
        result = self.system(client).solve("2 + 2", max_attempts=3)
        self.assertFalse(result.success)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(self.memory.experiences, [])
        self.assertFalse(self.memory.path.exists())
        traces = [json.loads(line) for line in self.traces.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(traces[0]["attempts"]), 3)

    def test_recursion_feedback_does_not_reveal_expected_answer(self):
        client = QueueClient([response("42"), response("49"), '{"lesson_ids":["multiplication"]}'])
        result = self.system(client).solve("7 * 7")
        self.assertTrue(result.success)
        self.assertEqual(len(client.calls), 3)
        feedback = client.calls[1][-1]["content"]
        self.assertIn("反思", feedback)
        self.assertNotIn("49", feedback)
        self.assertTrue(result.memory_added)
        self.assertEqual(self.memory.experiences[0]["attempts"], 2)

    def test_attempt_count_boundaries(self):
        for limit in (0, 4, -1):
            with self.assertRaises(ValueError):
                self.system(QueueClient([])).solve("2 + 2", max_attempts=limit)

    def test_persistence_retrieval_and_duplicate(self):
        result = self.system(QueueClient([response("21"), '{"lesson_ids":["multiplication"]}'])).solve("3 * 7")
        self.assertTrue(result.success)
        loaded = MemoryStore(self.memory.path)
        self.assertEqual(len(loaded.experiences), 1)
        self.assertTrue(loaded.retrieve(parse_task("4 * 8")))
        client = QueueClient([response("21")])
        repeated = RSLSystem(client, loaded, self.traces).solve(" (3 * 7) ")
        self.assertFalse(repeated.memory_added)
        self.assertEqual(len(loaded.experiences), 1)
        saved = json.loads(self.memory.path.read_text(encoding="utf-8"))
        self.assertNotIn("answer", saved["experiences"][0])
        self.assertNotIn("reflection", saved["experiences"][0])
        process = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("rsl.py")), "memory", "--memory", str(self.memory.path)],
            text=True, encoding="utf-8", capture_output=True, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(json.loads(process.stdout)["experiences"]), 1)

    def test_benchmark_modes_and_no_holdout_leakage(self):
        client = CorrectClient()
        metrics = benchmark(self.system(client))
        self.assertEqual(metrics["holdout_writes"], 0)
        self.assertEqual(len(client.calls), 20)
        self.assertEqual(len(self.memory.experiences), 4)
        holdout_ids = {parse_task(expression).task_id for expression in DEFAULT_HOLDOUT}
        self.assertFalse(holdout_ids.intersection(record["task_id"] for record in self.memory.experiences))
        self.assertTrue(all(record["source"] == "train" for record in self.memory.experiences))
        self.assertEqual(self.memory.protected, holdout_ids)
        self.assertEqual(metrics["baseline"]["model_calls"], 4)
        self.assertEqual(metrics["training"]["answer_calls"], 4)
        self.assertEqual(metrics["training"]["extraction_calls"], 4)
        self.assertEqual(metrics["training"]["model_calls"], 8)
        self.assertGreaterEqual(metrics["training"]["wall_seconds"], 0)
        self.assertTrue(all("通用经验" not in call[0]["content"] for call in client.calls[:4]))
        self.assertTrue(all("通用经验" in call[0]["content"] for call in client.calls[12:]))
        loaded = MemoryStore(self.memory.path)
        with self.assertRaises(ValueError):
            RSLSystem(QueueClient([]), loaded, self.traces).solve(DEFAULT_HOLDOUT[0])
        safe_result = RSLSystem(CorrectClient(), loaded, self.traces).solve(DEFAULT_HOLDOUT[0], learn=False)
        self.assertTrue(safe_result.success)
        self.assertEqual(len(loaded.experiences), 4)

    def test_train_holdout_overlap_rejected_before_calls_or_writes(self):
        client = CorrectClient()
        with self.assertRaises(ValueError):
            benchmark(self.system(client), train=["2 + 3"], holdout=["(2+3)"])
        self.assertFalse(client.calls)
        self.assertFalse(self.memory.path.exists())

    def test_existing_memory_overlap_and_evaluation_write_rejected(self):
        self.system(QueueClient([response("5"), '{"lesson_ids":["basic"]}'])).solve("2+3")
        client = CorrectClient()
        with self.assertRaises(ValueError):
            benchmark(self.system(client), train=["4+5"], holdout=["(2+3)"])
        self.assertFalse(client.calls)
        with self.assertRaises(ValueError):
            self.system(client).solve("4+5", source="benchmark-memory", learn=True)
        self.assertFalse(client.calls)

    def test_cli_options_before_and_after_command(self):
        parser = build_parser()
        args = parser.parse_args(["--host", "http://localhost:9999", "solve", "--expression", "2+2",
                                  "--memory", str(self.memory.path)])
        self.assertEqual(args.host, "http://localhost:9999")
        self.assertEqual(args.memory, self.memory.path)
        args = parser.parse_args(["demo", "--model", "custom:4b"])
        self.assertEqual(args.command, "demo")
        self.assertEqual(args.model, "custom:4b")

    def test_ollama_payload_disables_thinking_and_limits_context(self):
        from unittest.mock import patch

        class FakeHTTP:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, *args):
                return json.dumps({"message": {"content": response("4")}}).encode()

        with patch("urllib.request.urlopen", return_value=FakeHTTP()) as request_mock:
            raw = OllamaClient().chat([{"role": "user", "content": "2+2"}])
        self.assertEqual(json.loads(raw)["answer"], "4")
        payload = json.loads(request_mock.call_args.args[0].data)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertFalse(payload["stream"])
        self.assertEqual(list(payload["format"]["properties"]), ["reflection", "answer"])
        self.assertEqual(payload["format"]["required"], ["reflection", "answer"])

    def test_bare_host_is_normalized(self):
        self.assertEqual(OllamaClient("127.0.0.1:11435").host, "http://127.0.0.1:11435")
        self.assertEqual(OllamaClient("http://localhost:11435/").host, "http://localhost:11435")

    def test_invalid_or_unmatched_lesson_does_not_enter_memory(self):
        for lesson in ('{"lesson_ids":["injected_rule"]}', '{"lesson_ids":["division"]}',
                       '{"lesson_ids":[]}', 'not json',
                       '{"lesson_ids":["multiplication"],"answer":"49"}'):
            client = QueueClient([response("49"), lesson])
            with self.subTest(lesson=lesson):
                result = self.system(client).solve("7*7")
                self.assertTrue(result.success)
                self.assertFalse(result.memory_added)
                self.assertEqual(result.lesson_extraction["status"], "rejected")
                self.assertFalse(self.memory.experiences)

    def test_memory_has_only_selected_model_rules_and_trace_reference(self):
        client = QueueClient([response("24"), '{"lesson_ids":["precedence"]}'])
        result = self.system(client).solve("2 * 9 + 6")
        record = self.memory.experiences[0]
        self.assertEqual(record["rule_ids"], ["precedence"])
        self.assertEqual(record["successful_trace_id"], result.trace_id)
        self.assertEqual(record["selection_basis"]["successful_attempt"], 1)
        self.assertNotIn("answer", record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
