# 小规模实验批次

正式实验入口、固定参数和测量边界见 [实验协议](../../mini_repro/EXPERIMENT_PROTOCOL.md)。默认正式配置是 [formal4070.json](../../mini_repro/profiles/formal4070.json)，执行 `bash mini_repro/run_local.sh reproduce` 会创建新批次。

## 2026-10-13 RTX 4070 流程验收

批次：[20261013-4070-protocol-validation/report.md](20261013-4070-protocol-validation/report.md)。使用 `validation4070.json`，只用于检查实验入口，**不能替代正式实验结果**。

| 验收任务 | 实际记录 | 进程耗时 | 峰值 GPU 分配 |
|---|---|---:|---:|
| 质量/体积 | 1 个文本片段、3 个问答条件；各 6 方法 | 10.284 秒 | 2.174 GiB |
| TTFT | 1 个上下文、4 方法、各 2 次测量；预热另计 | 4.442 秒 | 2.188 GiB |
| 自适应，倍率 1 | 1 个事实、1 轨迹、6 策略、各 2 次测量 | 6.707 秒 | 2.186 GiB |
| 自适应，倍率 0.01 | 同上 | 7.693 秒 | 2.186 GiB |

41 项测试通过，包括 GPU 编解码、本机 socket、剩余预算回归和报告统计。4 个任务全部完成；逐项样本数与重复次数校验通过；记录的源码/数据指纹与实际文件一致。模型使用固定 revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`，权重 SHA-256 与环境验收值一致。

配置、执行命令及任务状态在 [manifest.json](20261013-4070-protocol-validation/manifest.json)，环境、模型、源码、数据和 CUDA 扩展二进制指纹在 [metadata.json](20261013-4070-protocol-validation/metadata.json)。日志保留在本机批次目录，`.log` 按仓库规则忽略。

该批次确认本机环境能运行整套流程，不能证明方法在正式数据上满足质量门槛或带来加速。正式配置包含 60 个文本片段、90 个问答条件、1080 次 TTFT 与 1800 次自适应计时，本次尚未执行。
