# Mistral-7B CacheGen 小规模实验

运行采用云端 tmux 会话 `cachegen`，终端断开后仍继续。云主机/容器需要保持运行。

## 查看运行

在仓库根目录执行：

```bash
tmux attach -t cachegen
# Ctrl+B，松开后按 D：退出查看并保留后台运行。

tail -f results/mistral/latest/run.log
cat results/mistral/latest/manifest.json
cat results/mistral/latest/exit_code
python3 -m experiments status mistral
```

`latest` 指向最近批次；`exit_code` 在进程结束后出现，0 表示进程成功。正式成功还需 manifest 为 completed，并通过预定数量与方法的完整性检查。

需要停止时，先读新批次 manifest 的 `runner_pid`，使用 `kill -TERM <runner_pid>`；入口会转发信号、等待子进程并保留中断状态。历史批次仅有 `pid`，属于当时记录。不要用关闭查看窗口的方式停止实验。中断或失败的批次保留现场。

## 固定设计

配置见 config.json。采用缓存的 `mistral-community/Mistral-7B-v0.2` revision `80cf49b18de6354fd8a6d1e81c177dd830e05eea`，通过仓库原始 FastChat 8bit 权重加载函数加载固定快照目录，KV 为 FP16。所有方法共享同一模型权重。

- 体积/续写：前 10 条 LongChat，各取 1024/2048/4096 token，共 30 条件；32-token 后续文本评分第 2–32 个 token。
- QA：按原顺序取前 10 条不超过 4096 token 的完整 TriviaQA，不截断原问题和证据。预填充除末 token 外的完整输入，重放末 token 后确定性生成最多 32 token。报告完整输出的标准化 EM/F1，取答案别名的最高分。
- 方法：原始 FP16、实际打包 uniform4、uniform8、作者 CacheGen 1/2/3。字节数包含序列化元数据。
- TTFT：前 3 条 LongChat、3 长度、0.1/1/10 Gbps、文本重算/uniform8/CacheGen 2/3，每条件预热 2 次、测量 5 次。共 540 次正式计时、216 次预热。固定种子打乱方法顺序。
- 每条上下文检查原始缓存与完整重算的首 token 一致性。最小验收还比较 logits 与贪心生成结果。

## 测量边界

作者仓库编码器没有论文 anchor/delta 阶段。本轮 TTFT 在本机限速 socket 上传送真实字节，完整接收后恢复并生成首 token，没有传输—解码重叠；离线编码耗时单独记录。

不以某一固定压缩率或加速倍数判断成功。结果报告每个长度的配对 NLL 差值与完整短 TriviaQA 的 EM/F1；描述性门槛为平均 NLL 增量≤0.02，保留≥90% 原始 KV 的 EM 正确题。QA 是混合长度的单独样本，不构成逐长度 QA 等价性证明，也不构成统计非劣证明。

## 文件

每个批次保存 manifest.json、metadata.json、smoke.json、qa_selection.json、quality.json、ttft.json、summary.json、CSV、report.md、tests.log、run.log 和 exit_code。metadata 记录模型、数据、源码和扩展指纹。原始仓库和历史实验数据保持在原位置。

新批次通过 `python3 -m experiments run mistral` 启动，或沿用 `bash mistral_repro/run.sh`。使用 `--config FILE` 传入独立配置；配置会先复制到新批次，源码也会保存快照。共享 GPU 锁避免同一工作区的测量并发。预演使用 `python3 -m experiments run mistral --dry-run`。

## 目录迁移

实验代码位于 `experiments/mistral`，原始结果位于 `results/mistral`，中文报告位于 `reports/mistral`，缓存与字体位于 `.local/mistral`。旧 `mistral_repro`、`runs`、`deliverables` 和 `latest` 使用仓库内相对链接兼容。历史批次元数据及源码快照保持原样；迁移记录位于本机 `.local/migration/20261024-workspace/`。路径覆盖、新增实验与后处理说明见 [工作区指南](../../docs/reproduction/WORKSPACE_GUIDE.md)。
