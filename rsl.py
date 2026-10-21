"""一个用验证、反思和经验记忆实现的最小 RSL 作业；不更新模型权重。

只处理整数、括号及 + - * /。模型输出不会作为 Python 代码执行。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "qwen3:4b-q4_K_M"
MAX_EXPRESSION_LENGTH = 512
MAX_NODES = 64
MAX_DEPTH = 16
MAX_LITERAL_DIGITS = 100
MAX_RESULT_BITS = 4096
MAX_ATTEMPTS = 3
ANSWER_PATTERN = re.compile(r"[+-]?\d+(?:/[+-]?\d+)?\Z", re.ASCII)

# 这些规则由程序限定；未经审核的模型反思不会直接变成未来提示词。
RULES = {
    "parentheses": "从内到外计算括号，并在替换子表达式时保留其正负号。",
    "precedence": "先乘除后加减；同级运算从左到右计算。",
    "multiplication": "整数乘法可分拆为高位和低位，重新相加并检查位数。",
    "division": "使用约分后的精确分数，避免浮点近似；除数必须非零。",
    "subtraction": "逐项检查减号及负数，减去负数等于加上该数的绝对值。",
    "unary": "先确定一元正负号，再处理外围的二元运算。",
    "large_integer": "大整数计算逐位核对进位，可用末位及数量级进行交叉检查。",
    "basic": "输出一个整数或最简分数，再重新代回原式检查。",
}

DEFAULT_TRAIN = [
    "12345 * 6789",
    "(17 + 25) * 9 - 16",
    "(81 / 7) - (5 / 3)",
    "-23 * (14 - 31) + 7",
]
DEFAULT_HOLDOUT = [
    "23456 * 7891",
    "(19 + 34) * 8 - 27",
    "(94 / 9) - (7 / 5)",
    "-29 * (16 - 37) + 11",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExpressionError(ValueError):
    """不属于受支持算术任务，或者超过资源边界。"""


@dataclass(frozen=True)
class Task:
    expression: str
    value: Fraction
    task_id: str
    features: tuple[str, ...]


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_task(expression: str) -> Task:
    """限制 AST 的节点、深度和数字大小，用 Fraction 作确定性裁判。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("表达式不能为空。")
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"表达式长度不能超过 {MAX_EXPRESSION_LENGTH} 个字符。")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ExpressionError("表达式语法错误。") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError(f"表达式 AST 节点不能超过 {MAX_NODES} 个。")
    features: set[str] = set()
    operators: set[type] = set()

    def bounded(value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RESULT_BITS:
            raise ExpressionError("计算中间结果过大。")
        return value

    def evaluate(node: ast.AST, depth: int) -> Fraction:
        if depth > MAX_DEPTH:
            raise ExpressionError(f"表达式嵌套深度不能超过 {MAX_DEPTH}。")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if len(str(abs(node.value))) > MAX_LITERAL_DIGITS:
                raise ExpressionError(f"整数最多允许 {MAX_LITERAL_DIGITS} 位。")
            if abs(node.value) >= 10000:
                features.add("large_integer")
            return bounded(Fraction(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            features.add("unary")
            value = evaluate(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            operators.add(type(node.op))
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return bounded(left + right)
            if isinstance(node.op, ast.Sub):
                features.add("subtraction")
                return bounded(left - right)
            if isinstance(node.op, ast.Mult):
                features.add("multiplication")
                return bounded(left * right)
            features.add("division")
            if right == 0:
                raise ExpressionError("分母或除数不能为 0。")
            return bounded(left / right)
        raise ExpressionError("只允许整数、括号及 + - * /；禁止函数、变量、幂和代码。")

    value = evaluate(tree.body, 0)
    if "(" in expression:
        features.add("parentheses")
    if operators.intersection({ast.Add, ast.Sub}) and operators.intersection({ast.Mult, ast.Div}):
        features.add("precedence")
    if not features:
        features.add("basic")
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    task_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return Task(expression, value, task_id, tuple(sorted(features)))


def parse_answer(answer: Any) -> Fraction:
    if not isinstance(answer, str):
        raise ValueError("answer 必须是字符串。")
    answer = answer.strip()
    if len(answer) > 512 or not ANSWER_PATTERN.fullmatch(answer):
        raise ValueError("answer 只能是整数或分数，例如 12 或 -7/3。")
    try:
        return Fraction(answer)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("answer 分数无效，分母必须非零。") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary_name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


class MemoryStore:
    """跨进程持久化已验证经验；只保存通用规则 ID，不保存模型答案。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "experiences": [],
            "protected_holdout_task_ids": [],
        }
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("schema_version") != 1:
                    raise ValueError("不支持的记忆版本。")
                experiences = data.get("experiences")
                protected = data.get("protected_holdout_task_ids")
                if not isinstance(experiences, list) or not isinstance(protected, list):
                    raise ValueError("记忆字段格式错误。")
                if any(not isinstance(item, str) for item in protected):
                    raise ValueError("留出题指纹格式错误。")
                for record in experiences:
                    if not isinstance(record, dict) or record.get("verified") is not True:
                        raise ValueError("记忆包含未经验证的记录。")
                    if record.get("source") not in {"solve", "train"}:
                        raise ValueError("记忆不能包含评估记录。")
                    if not isinstance(record.get("task_id"), str):
                        raise ValueError("经验指纹格式错误。")
                    if not isinstance(record.get("rule_ids"), list) or any(
                            rule not in RULES for rule in record["rule_ids"]):
                        raise ValueError("经验只能包含预定义通用规则。")
                    if not isinstance(record.get("features"), list) or any(
                            feature not in RULES for feature in record["features"]):
                        raise ValueError("经验特征格式错误。")
                if {record["task_id"] for record in experiences}.intersection(protected):
                    raise ValueError("记忆中的训练题与受保护留出题重叠。")
                self.data = data
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"无法读取记忆文件 {self.path}：{exc}") from exc

    @property
    def experiences(self) -> list[dict[str, Any]]:
        return self.data["experiences"]

    @property
    def protected(self) -> set[str]:
        return set(self.data["protected_holdout_task_ids"])

    def check_can_learn(self, task: Task) -> None:
        if task.task_id in self.protected:
            raise ValueError("此题是受保护留出题，禁止写入训练记忆；请使用 --no-learn。")

    def reserve_holdout(self, tasks: list[Task]) -> None:
        ids = {task.task_id for task in tasks}
        if ids.intersection(record["task_id"] for record in self.experiences):
            raise ValueError("留出集与已有训练记忆重叠，请更换留出题或使用新的记忆文件。")
        self.data["protected_holdout_task_ids"] = sorted(self.protected | ids)
        atomic_json(self.path, self.data)

    def retrieve(self, task: Task, limit: int = 3) -> list[str]:
        ranked = sorted(
            self.experiences,
            key=lambda record: len(set(record["features"]).intersection(task.features)),
            reverse=True,
        )
        selected: list[str] = []
        for record in ranked[:limit]:
            if not set(record["features"]).intersection(task.features):
                continue
            for rule_id in record["rule_ids"]:
                if rule_id in task.features and rule_id not in selected:
                    selected.append(rule_id)
        return [RULES[rule_id] for rule_id in selected]

    def add_verified(self, task: Task, attempts: int, source: str, *, rule_ids: list[str],
                     model: str, trace_id: str) -> bool:
        self.check_can_learn(task)
        if source not in {"solve", "train"}:
            raise ValueError("只允许 solve 或 train 来源写入记忆。")
        if any(record["task_id"] == task.task_id for record in self.experiences):
            return False
        if not rule_ids or any(rule not in RULES or rule not in task.features for rule in rule_ids):
            raise ValueError("经验提炼必须选择已审核且匹配题型的规则。")
        self.experiences.append({
            "task_id": task.task_id,
            "features": list(task.features),
            "rule_ids": list(dict.fromkeys(rule_ids)),
            "verified": True,
            "source": source,
            "attempts": attempts,
            "model_source": model,
            "successful_trace_id": trace_id,
            "selection_basis": {
                "candidate_rule_ids": list(task.features),
                "successful_attempt": attempts,
                "method": "model_selected_allowed_rules_after_exact_verification",
            },
            "created_at": now_iso(),
        })
        atomic_json(self.path, self.data)
        return True


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11435", model: str = DEFAULT_MODEL,
                 timeout: float = 120):
        host = host.strip().rstrip("/")
        self.host = host if "://" in host else "http://" + host
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema or {
                "type": "object",
                "properties": {
                    "reflection": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["reflection", "answer"],
                "additionalProperties": False,
            },
            "options": {"num_ctx": 4096, "temperature": 0, "seed": 42, "num_predict": 256},
        }
        request = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            content = result["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("模型响应 content 不是字符串。")
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Ollama 请求失败：{exc}") from exc


@dataclass
class SolveResult:
    task_id: str
    expression: str
    success: bool = False
    answer: str | None = None
    expected_answer: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    memory_added: bool = False
    retrieved_rules: list[str] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "solve"
    started_at: str = field(default_factory=now_iso)
    lesson_extraction: dict[str, Any] | None = None
    wall_seconds: float = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class RSLSystem:
    def __init__(self, client: Any, memory: MemoryStore, traces: Path | str):
        self.client = client
        self.memory = memory
        self.traces = Path(traces)

    def _write_trace(self, result: SolveResult) -> None:
        self.traces.parent.mkdir(parents=True, exist_ok=True)
        with self.traces.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")

    def _learn(self, task: Task, result: SolveResult) -> None:
        if any(record["task_id"] == task.task_id for record in self.memory.experiences):
            result.lesson_extraction = {"status": "duplicate_skipped", "model_calls": 0}
            return
        extraction: dict[str, Any] = {"status": "rejected", "model_calls": 1}
        result.lesson_extraction = extraction
        candidate_rules = {rule_id: RULES[rule_id] for rule_id in task.features}
        schema = {
            "type": "object",
            "properties": {"lesson_ids": {
                "type": "array", "items": {"type": "string", "enum": list(candidate_rules)},
                "minItems": 1, "maxItems": len(candidate_rules),
            }},
            "required": ["lesson_ids"],
            "additionalProperties": False,
        }
        messages = [
            {"role": "system", "content": (
                "从已通过精确验证的计算轨迹中提炼可复用经验。"
                "你只能从提供的审核规则目录中选择最有帮助的规则ID。"
                "只输出 JSON，例如 {\"lesson_ids\":[\"precedence\"]}；"
                "不要保存题目数字、答案或自由文本。"
            )},
            {"role": "user", "content": json.dumps({
                "expression": task.expression,
                "verified_attempts": [
                    {key: attempt[key] for key in ("number", "answer", "reflection", "verified") if key in attempt}
                    for attempt in result.attempts
                ],
                "allowed_rules": candidate_rules,
            }, ensure_ascii=False)},
        ]
        started = time.perf_counter()
        try:
            raw = self.client.chat(messages, schema=schema)
            extraction["raw_response"] = raw
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != {"lesson_ids"}:
                raise ValueError("经验提炼必须仅包含 lesson_ids。")
            selected = payload["lesson_ids"]
            if not isinstance(selected, list) or not selected or len(selected) > len(candidate_rules):
                raise ValueError("经验规则必须是非空且有界的列表。")
            if any(not isinstance(rule, str) or rule not in candidate_rules for rule in selected):
                raise ValueError("经验包含未经审核或不匹配题型的规则，拒绝写入。")
            selected = list(dict.fromkeys(selected))
            extraction.update(status="accepted", lesson_ids=selected)
            result.memory_added = self.memory.add_verified(
                task, len(result.attempts), result.source, rule_ids=selected,
                model=getattr(self.client, "model", "test-client"), trace_id=result.trace_id,
            )
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            extraction["feedback"] = str(exc)
            extraction["error_type"] = "model_error" if isinstance(exc, RuntimeError) else "invalid_lesson"
        finally:
            extraction["wall_seconds"] = round(time.perf_counter() - started, 6)

    def solve(self, expression: str, *, max_attempts: int = 3, learn: bool = True,
              use_memory: bool = True, source: str = "solve") -> SolveResult:
        if not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise ValueError("max_attempts 必须在 1 到 3 之间（包括首次作答）。")
        started = time.perf_counter()
        task = parse_task(expression)
        if learn:
            if source not in {"solve", "train"}:
                raise ValueError("评估任务不能写入训练记忆。")
            self.memory.check_can_learn(task)
        rules = self.memory.retrieve(task) if use_memory else []
        result = SolveResult(task_id=task.task_id, expression=task.expression,
                             expected_answer=fraction_text(task.value),
                             retrieved_rules=rules, source=source)
        system_prompt = (
            "你是精确算术助手。只输出 JSON 对象，按顺序先写字符串 reflection，再写字符串 answer。"
            "reflection 必须先用简短算式列出实际计算的关键中间数值，"
            "检查括号、优先级、符号和进位，然后根据这些中间结果确定最终答案。"
            "answer 只能包含最终整数或最简分数字符串（例如 14 或 -7/3）；"
            "不得包含小数、句子、等号、解释、单位、链接或代码。"
            "格式示例（与当前题无关）：计算 2*(3+4)，"
            '输出 {"reflection":"3+4=7;2*7=14","answer":"14"}。'
        )
        if rules:
            system_prompt += "\n来自已通过验证任务的通用经验：\n" + "\n".join("- " + rule for rule in rules)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "计算表达式：" + task.expression},
        ]
        for number in range(1, max_attempts + 1):
            attempt: dict[str, Any] = {"number": number, "verified": False}
            try:
                raw = self.client.chat(messages)
            except RuntimeError as exc:
                attempt.update(error_type="model_error", feedback=str(exc))
                result.attempts.append(attempt)
                break
            attempt["raw_response"] = raw
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("输出必须是 JSON 对象。")
                answer = payload.get("answer")
                attempt["answer"] = answer
                reflection = payload.get("reflection", "")
                attempt["reflection"] = reflection if isinstance(reflection, str) else ""
                candidate = parse_answer(answer)
                if candidate != task.value:
                    attempt.update(error_type="wrong_answer", feedback="结果未通过精确算术验证。")
                else:
                    result.success = True
                    result.answer = fraction_text(candidate)
                    attempt["verified"] = True
            except (json.JSONDecodeError, ValueError) as exc:
                attempt.update(error_type="invalid_output", feedback=str(exc))
            result.attempts.append(attempt)
            if result.success:
                if learn:
                    self._learn(task, result)
                break
            # 只反馈错误类型，不把裁判的正确答案提供给模型。
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "上一轮反馈：" + attempt["feedback"] +
                    "请反思上一轮错误，重新核对原表达式的括号、运算顺序、符号和进位。"
                    "先在 reflection 中用简短算式列出重新检查后的实际中间数值，"
                    "再根据这些数值给出修正后的 answer。answer 只能是整数或最简分数字符串；"
                    "不得写句子、解释或链接。仍然只输出 JSON，字段顺序为 reflection、answer。"
                )},
            ])
        result.wall_seconds = round(time.perf_counter() - started, 6)
        self._write_trace(result)
        return result


def check_split(train: list[str], holdout: list[str], memory: MemoryStore) -> tuple[list[Task], list[Task]]:
    if not train or not holdout:
        raise ValueError("训练集和留出集都不能为空。")
    train_tasks = [parse_task(expression) for expression in train]
    holdout_tasks = [parse_task(expression) for expression in holdout]
    train_ids = {task.task_id for task in train_tasks}
    holdout_ids = {task.task_id for task in holdout_tasks}
    if len(train_ids) != len(train_tasks) or len(holdout_ids) != len(holdout_tasks):
        raise ValueError("数据集内包含重复题目（忽略空格及冗余括号）。")
    if train_ids.intersection(holdout_ids):
        raise ValueError("训练集与留出集重叠，拒绝进行评估。")
    if train_ids.intersection(memory.protected):
        raise ValueError("训练集包含此前保留的留出题，拒绝学习。")
    if holdout_ids.intersection(record["task_id"] for record in memory.experiences):
        raise ValueError("留出集与已有训练记忆重叠，拒绝进行评估。")
    return train_tasks, holdout_tasks


def summarize(results: list[SolveResult]) -> dict[str, Any]:
    successes = sum(result.success for result in results)
    return {
        "tasks": len(results),
        "successes": successes,
        "accuracy": successes / len(results) if results else 0,
        "answer_calls": sum(len(result.attempts) for result in results),
        "extraction_calls": sum((result.lesson_extraction or {}).get("model_calls", 0) for result in results),
        "model_calls": sum(len(result.attempts) + (result.lesson_extraction or {}).get("model_calls", 0)
                           for result in results),
        "model_errors": sum(attempt.get("error_type") == "model_error"
                            for result in results for attempt in result.attempts) +
                        sum((result.lesson_extraction or {}).get("error_type") == "model_error"
                            for result in results),
        "wall_seconds": round(sum(result.wall_seconds for result in results), 6),
        "mean_attempts": sum(len(result.attempts) for result in results) / len(results) if results else 0,
        "results": [{"task_id": result.task_id, "expression": result.expression,
                     "success": result.success, "answer": result.answer,
                     "attempts": len(result.attempts), "trace_id": result.trace_id}
                    for result in results],
    }


def benchmark(system: RSLSystem, *, train: list[str] | None = None,
              holdout: list[str] | None = None, max_attempts: int = 3) -> dict[str, Any]:
    started = time.perf_counter()
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError("max_attempts 必须在 1 到 3 之间。")
    train_tasks, holdout_tasks = check_split(
        DEFAULT_TRAIN if train is None else train,
        DEFAULT_HOLDOUT if holdout is None else holdout,
        system.memory,
    )
    # 留出题指纹持久化保护，未来 solve 也不能误把这些题写入经验库。
    system.memory.reserve_holdout(holdout_tasks)
    before = len(system.memory.experiences)

    def run_group(tasks: list[Task], **kwargs: Any) -> dict[str, Any]:
        group_started = time.perf_counter()
        results = [system.solve(task.expression, **kwargs) for task in tasks]
        summary = summarize(results)
        summary["wall_seconds"] = round(time.perf_counter() - group_started, 6)
        return summary

    baseline = run_group(holdout_tasks, max_attempts=1, learn=False, use_memory=False,
                         source="benchmark-baseline")
    training = run_group(train_tasks, max_attempts=max_attempts, learn=True, use_memory=True,
                         source="train")
    after_training = len(system.memory.experiences)
    memory_results = run_group(holdout_tasks, max_attempts=1, learn=False, use_memory=True,
                               source="benchmark-memory")
    recursive = run_group(holdout_tasks, max_attempts=max_attempts, learn=False, use_memory=True,
                          source="benchmark-recursive")
    if len(system.memory.experiences) != after_training:
        raise RuntimeError("评估意外改变了训练经验数量。")
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "model": getattr(system.client, "model", "test-client"),
        "learning": "verified_experience_memory_only_no_weight_updates",
        "holdout_writes": 0,
        "memory_records_before": before,
        "memory_records_after_training": after_training,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "training": training,
        "baseline": baseline,
        "memory": memory_results,
        "recursive": recursive,
        "interpretation": "仅展示本次小样本结果；经验记忆和递归纠错不保证提高准确率。",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小 RSL：本地小模型 + 算术验证 + 递归修正 + 经验记忆")

    def common(target: argparse.ArgumentParser, suppress: bool = False) -> None:
        def default(value: Any) -> Any:
            return argparse.SUPPRESS if suppress else value
        target.add_argument("--host", default=default(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")),
                            help="Ollama 地址")
        target.add_argument("--model", default=default(DEFAULT_MODEL), help="Ollama 模型名")
        target.add_argument("--memory", type=Path, default=default(BASE_DIR / "memory.json"), help="经验 JSON 路径")
        target.add_argument("--traces", type=Path, default=default(BASE_DIR / "traces.jsonl"), help="过程 JSONL 路径")
        target.add_argument("--timeout", type=float, default=default(120), help="单次请求超时秒数")

    common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    solve_parser = commands.add_parser("solve", help="求解一题，只有验证成功才可保存经验")
    common(solve_parser, True)
    solve_parser.add_argument("--expression", required=True)
    solve_parser.add_argument("--max-attempts", type=int, default=3, choices=range(1, 4), help="总轮数，包含首次作答")
    solve_parser.add_argument("--no-learn", action="store_true", help="禁止写入经验")
    solve_parser.add_argument("--no-memory", action="store_true", help="不检索已有经验")
    bench_parser = commands.add_parser("benchmark", aliases=["demo"], help="分别测评无记忆、记忆和递归修正")
    common(bench_parser, True)
    bench_parser.add_argument("--max-attempts", type=int, default=3, choices=range(1, 4))
    bench_parser.add_argument("--metrics", type=Path, default=BASE_DIR / "metrics.json")
    bench_parser.add_argument("--dataset", type=Path, help='JSON 对象，含 train 和 holdout 两个表达式列表')
    memory_parser = commands.add_parser("memory", help="查看经验和留出题保护信息，不调用模型")
    common(memory_parser, True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout 必须大于 0。")
        memory = MemoryStore(args.memory)
        if args.command == "memory":
            print(json.dumps(memory.data, ensure_ascii=False, indent=2))
            return 0
        if args.memory.resolve() == args.traces.resolve():
            raise ValueError("memory 和 traces 必须使用不同文件。")
        system = RSLSystem(OllamaClient(args.host, args.model, args.timeout), memory, args.traces)
        if args.command == "solve":
            result = system.solve(args.expression, max_attempts=args.max_attempts,
                                  learn=not args.no_learn, use_memory=not args.no_memory)
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
            return 0 if result.success else 1
        if args.metrics.resolve() in {args.memory.resolve(), args.traces.resolve()}:
            raise ValueError("metrics、memory 和 traces 必须使用不同文件。")
        dataset: dict[str, Any] = {}
        if args.dataset:
            dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
            if not isinstance(dataset, dict) or set(dataset) != {"train", "holdout"}:
                raise ValueError("dataset 必须含 train 和 holdout 两个字段。")
            if any(not isinstance(dataset[key], list) for key in ("train", "holdout")):
                raise ValueError("train 和 holdout 必须是表达式列表。")
        metrics = benchmark(system, max_attempts=args.max_attempts, **dataset)
        atomic_json(args.metrics, metrics)
        for name in ("baseline", "memory", "recursive"):
            section = metrics[name]
            print(f"{name:10s}：{section['successes']}/{section['tasks']}，"
                  f"准确率 {section['accuracy']:.1%}，模型调用 {section['model_calls']} 次")
        print(f"训练经验：{metrics['memory_records_before']} → {metrics['memory_records_after_training']}")
        print("仅学习经验证的通用经验，不更新模型权重；不保证准确率提升。")
        print(f"指标文件：{args.metrics.resolve()}\n过程记录：{args.traces.resolve()}")
        return 1 if any(metrics[name]["model_errors"] for name in
                        ("training", "baseline", "memory", "recursive")) else 0
    except (ExpressionError, ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
