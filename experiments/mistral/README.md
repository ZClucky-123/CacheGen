# Mistral 兼容入口

Mistral 使用 [公共 v3 框架](../../docs/reproduction/SHARED_FRAMEWORK.md)，与 TinyLlama 共用质量、三模式 TTFT、历史吞吐自适应和轻量消融。此目录保留薄兼容入口和历史报告工具。

```bash
# 默认 small，新批次进入 results/mistral/small/<run-id>
bash experiments/mistral/run.sh --config experiments/mistral/config.json
python3 -m experiments.background start --name mistral-v3-new -- \
  --model mistral --config experiments/configs/minimal.json --run-id v3-new
python3 -m experiments status mistral
```

`config.json` 继承公共 `small.json`。固定模型为 `mistral-community/Mistral-7B-v0.2`，revision `80cf49b18de6354fd8a6d1e81c177dd830e05eea`，保留 FastChat groupwise 8bit 权重和 FP16 KV。较大配置为 `experiments/configs/larger-mistral.json`。

`experiment.py`、`run.sh`、`measure.sh` 和 `mistral_repro` 保持兼容；续跑使用 `--resume <原批次>`。原独立测量器保存在历史 `source/`；`build_document.py` 是固定 2026-10-24 设计的报告器，新协议由公共 `analysis`/`artifacts` 自动完成。

历史评分和旧结果见 [历史索引](../../docs/reproduction/HISTORY.md)，不与 v3 合并；论文差距见 [论文对照](../../docs/reproduction/PAPER_ALIGNMENT.md)。
