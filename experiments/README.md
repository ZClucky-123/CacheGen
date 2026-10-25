# 公共实验框架 v3

TinyLlama 与 Mistral 共用配置校验、Engine、数据选择、任务、检查点、分析和归档。四组实验默认展开为七任务：`quality`、`ttft`、四组倍率/SLO 的 `adaptive` 和 `ablation`。

| 模块 | 职责 |
| --- | --- |
| `config.py`、`configs/` | 适配器、精确矩阵和参数校验 |
| `engine.py`、`stable.py`、`numerics.py` | 模型/KV 接口、执行配置及正确性 |
| `datasets.py`、`scoring.py` | 数据选择、完整续写 NLL 和 QA |
| `chunks.py`、`ttft.py`、`feedback.py` | 分块、三模式传输与历史反馈 |
| `mechanisms.py`、`ablation.py` | 共用量化中间结果及机制消融 |
| `suite.py`、`records.py`、`artifacts.py` | 调度、SQLite 检查点、指纹和归档 |
| `analysis.py`、`analysis_v3.py` | 分组、配对、时间线和图表 |
| `background.py`、`workspace.py` | tmux 后台、路径、注册与状态 |

```bash
python3 -m experiments list
bash experiments/run.sh --model mistral --config experiments/configs/minimal.json --dry-run
python3 -m experiments.background start --name mistral-v3-new -- \
  --model mistral --config experiments/configs/minimal.json --run-id v3-new
python3 -m experiments status mistral
python3 -m experiments.background status --name mistral-v3-new
```

结果为 `results/<model>/<scale>/<run-id>/`。注册入口默认 small，状态查询覆盖该模型全部规模。重复 run ID 拒绝覆盖，`--resume` 读取原配置并核对指纹；同 GPU 入口共用锁。

完整协议和命令见 [框架指南](../docs/reproduction/SHARED_FRAMEWORK.md)，模块和路径见 [工作区指南](../docs/reproduction/WORKSPACE_GUIDE.md)，实现边界见 [论文对照](../docs/reproduction/PAPER_ALIGNMENT.md)。新增模型复用适配器；`mistral/`、`mistral_repro` 和 `mini_repro` 保留兼容与历史入口，不另写正式算法。
