# Simple RSL Homework

作业二 **b：自己做一个简单的 Recursive Self-Learning 系统**。
用本地小模型完成“生成 → 验证 → 失败反思与递归修正 → 提炼经验 → 后续检索”的闭环。
任务限定为可精确判分的算术题，模型为 `qwen3:4b-q4_K_M`。

本项目学习的是**持久化经验记忆，不更新模型权重**。模型从审核规则库选择规则 ID，不自由发明长期规则；若课程要求参数学习，还需增加 LoRA 等训练模块。

## 快速开始

准备 Git、Python 3.10+ 和可下载模型的网络。Python 核心仅用标准库，无需 `pip install`。

```sh
git clone https://github.com/wneutralcola/simple-rsl-homework.git
cd simple-rsl-homework
```

### Windows

在 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1 solve --expression '(18 + 27) * 4 - 13' --memory .work/demo_memory.json --traces .work/demo_traces.jsonl
powershell -ExecutionPolicy Bypass -File .\run.ps1 memory --memory .work/demo_memory.json
powershell -ExecutionPolicy Bypass -File .\run.ps1 benchmark --dataset dataset.json --memory .work/rerun_memory.json --traces .work/rerun_traces.jsonl --metrics .work/rerun_metrics.json
python -m unittest -v test_rsl
```

`setup.ps1` 下载并校验官方 Ollama 0.40.3 独立包，放入仓库内 `.runtime/ollama`，拉取模型并启动 `http://127.0.0.1:11435` 服务。无需管理员安装或全局 Ollama `PATH`。
之后恢复服务运行 `powershell -ExecutionPolicy Bypass -File .\start.ps1`。`run.ps1` 查找 Python，也可通过 `RSL_PYTHON` 指定解释器。

### macOS / Linux

先按 [Ollama 官方说明](https://docs.ollama.com/quickstart)安装 Ollama，在另一个终端运行 `ollama serve`，再执行：

```sh
ollama pull qwen3:4b-q4_K_M
python3 rsl.py --host http://127.0.0.1:11434 benchmark --dataset dataset.json --memory .work/rerun_memory.json --traces .work/rerun_traces.jsonl --metrics .work/rerun_metrics.json
python3 -m unittest -v test_rsl
```

PowerShell 部署脚本仅面向 Windows；Python 核心使用标准库。此项目的实测环境为 Windows、RTX 5060 Laptop 8GB，未声称已在其他平台实测。
每次从零实验请使用尚不存在的 `--memory` 文件名；复用记忆会带入旧经验。新运行写入 `.work/`，保留版本化的 `results/` 证据。

## 方法

1. 按题型检索已有通用经验，调用本地 `/api/chat`。
2. 模型先在 JSON 的 `reflection` 写关键中间数值，再输出 `answer` 整数或最简分数。
3. 受限 AST 与 `Fraction` 独立精确计算并判分，不执行模型代码，不向纠错提示提供裁判答案。
4. 失败时给出反馈，请模型反思并重算；最多 3 次作答，包含首次。
5. 成功后另调用模型，从审核目录选择匹配题型的规则 ID，校验后写入 JSON；提炼失败不写经验。
6. 后续任务按题型读取经验。长期记忆不保存题目答案或自由反思文本，完整过程另写 JSONL。

只支持整数、括号、一元正负号和 `+ - * /`；分数写作 `81 / 7`。不支持小数、幂、变量或函数。
推理配置为 `think=false`、`num_ctx=4096`、`temperature=0`、`seed=42`、`num_predict=256`。固定种子不保证跨设备结果完全一致。

## 文件

| 文件 | 用途 |
| --- | --- |
| `rsl.py` / `test_rsl.py` | RSL 实现与 15 项流程、边界和隔离测试 |
| `setup.ps1` / `start.ps1` / `run.ps1` | Windows 部署、恢复服务和命令入口 |
| `verify_deployment.py` | 可选的真实推理与 GPU 状态检查 |
| `dataset.json` | 最终对照实验的 4 道训练题、4 道留出题 |
| `results/metrics.json` | 最终对照指标；对应轨迹和经验也在 `results/` |
| `results/metrics_initial.json` | 初版失败实验，保留供复核 |
| `results/demo_traces.jsonl` / `results/demo_memory.json` | 真实递归演示与独立经验 |
| `REPORT.md` | 简短实验报告 |

## 真实结果

训练与留出题分离，留出阶段禁止学习。baseline 不读经验、只答一次；memory 读训练经验、只答一次；recursive 读同一份经验、最多答三次。

| 组别 | 留出正确数 | 答题调用 | 耗时（秒） |
| --- | --- | --- | --- |
| baseline | 1/4（25%） | 4 | 3.176611 |
| memory | 1/4（25%） | 4 | 3.064344 |
| recursive | 2/4（50%） | 9 | 7.419466 |

训练通过 2/4，产生 2 条经验；训练用 9 次答题、2 次提炼调用，耗时 6.879271 秒。全流程 20.540531 秒，留出经验写入数为 0。
真实单题演示中，答案从错误的 **179** 经反馈修正为 **167**，另一次模型调用提炼 `parentheses`、`precedence` 并保存；早期连续三次失败的日志也保留。演示记忆与对照实验隔离。15 项单元测试在独立项目目录通过。

**memory 与 baseline 相同，本次未观察到经验本身提高准确率。** recursive 多答对一题，也多用了调用；4 题单次结果不能证明稳定提升。初版提示下三组均为 0/4，未形成训练经验；调整提示后使用新的留出题集，因此两次实验不能直接作为同一设置下的提升证明。

## 常用选项

`python rsl.py --help` 查看完整参数。`solve --max-attempts 1` 限制为单次，`--no-learn` 禁止写经验，`--no-memory` 禁止读取经验；`--host`、`--model` 可切换服务与模型，`demo` 是 `benchmark` 别名。

参考：[模型页面](https://ollama.com/library/qwen3:4b-q4_K_M)、[Ollama Chat API](https://docs.ollama.com/api/chat)。
