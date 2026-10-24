# Mistral 兼容入口

当前正式算法统一使用 [公共 v2 框架](../../docs/reproduction/SHARED_FRAMEWORK.md)，覆盖体积质量、TTFT 和四个自适应配置任务，另含真实 TriviaQA。

```bash
bash experiments/mistral/run.sh --config experiments/mistral/config.json
bash mistral_repro/run.sh --config experiments/configs/small.json
python3 -m experiments.background start --name mistral-new -- --model mistral --config experiments/configs/small.json
python3 experiments/mistral/status.py
```

config.json 继承公共 small.json。模型仍固定为 mistral-community/Mistral-7B-v0.2 的 80cf49b18de6354fd8a6d1e81c177dd830e05eea，原 FastChat groupwise 8bit 权重、FP16 KV。新结果保存到 results/mistral/<scale>/<run-id>，不使用旧批次路径。

experiment.py 与 measure.sh 是兼容包装。原独立实现保存在历史批次 source/ 中；build_document.py 是固定 20261024 设计的历史中文报告器，旧 plot_results.py/finalize.py 主要服务旧批次，新批次由公共 analysis/artifacts 自动完成。

历史记录和已知限制见 [历史索引](../../docs/reproduction/HISTORY.md)。旧 NLL、完整输出 QA、缺失自适应的记录不与 v2 混合。
