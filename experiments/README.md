# 公共实验框架

TinyLlama 与 Mistral 当前共用 `config.py`、`engine.py`、`datasets.py`、`tasks.py`、`records.py`、`suite.py`、`analysis.py` 和 `artifacts.py`。模型差异只放在适配器与配置，三个核心实验组仍是六个执行任务。

完整说明见 [配置、后台、查询、恢复和较大规模](../docs/reproduction/SHARED_FRAMEWORK.md)、[审查表](../docs/reproduction/SHARED_FRAMEWORK_AUDIT.md)和[历史索引](../docs/reproduction/HISTORY.md)。

```bash
python3 -m experiments list
python3 -m experiments run mistral --dry-run
bash experiments/run.sh --model mistral --config experiments/configs/small.json --dry-run
python3 -m experiments.background start --name mistral-small -- --model mistral --config experiments/configs/small.json
python3 -m experiments status mistral
python3 -m experiments.background status --name mistral-small
```

注册表仍支持独立的非模型工具接入，按 argv 列表转发，路径由 Workspace.discover() 解析。新增类似模型实验应扩展适配器/配置，避免另写正式测量流程。`experiments/mistral` 和 `mistral_repro` 保留薄兼容入口及历史后处理；不再有专用正式算法。

新批次为 `results/<model>/<scale>/<run-id>`。默认小规模；较大规模必须显式选择对应配置。重复 run ID 拒绝覆盖，`--resume` 严格核对全部指纹。后台任务使用服务器 tmux，同 GPU 正式测量使用共用排他锁。
