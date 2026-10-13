# TinyLlama 小规模正式实验：RTX 4070 Laptop（2026-10-13）

本目录保存批次 `20261013T060232923276Z` 的完整配置、环境、原始结果与报告。用途为 `formal-small-scale`，6/6 个任务均为 `completed`、退出码均为 0。运行时间为北京时间 **2026-10-13 14:02:32–14:21:35，约 19 分 2 秒**；UTC 时间与逐任务耗时见 [manifest.json](manifest.json)。

这是当前仓库编解码器在 TinyLlama 上的小规模实验，不是论文完整复现。本文描述该批次实际记录，不代表其他机器的性能或所需时间。

## 运行环境与版本

| 项目 | 本批次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，约 8 GiB 显存，计算能力 8.9 |
| 系统 | WSL2，Linux `6.6.87.2-microsoft-standard-WSL2`，glibc 2.31 |
| Python | 3.10.14 |
| PyTorch / CUDA runtime | 2.3.1+cu121 / 12.1 |
| Transformers / NumPy | 4.42.3 / 1.26.4 |
| 编解码依赖 | torchac 0.9.3、torchac_cuda 0.0.0、lmcache 0.1 |
| 模型 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 固定模型 revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| 对应源码提交 | `c0dc0530321faf2e972784fbab8aa4dd236b57bc`（`c0dc053`） |
| 随机种子 | 0 |

[metadata.json](metadata.json) 保存模型文件、源码、数据与 CUDA 扩展二进制的 SHA-256，以及运行时的 Git 状态。模型权重 SHA-256 为 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`。原始元数据保留当时的本机路径；目录移动不改变该批次的记录。

## 实验输入与规模

自然文本取自仓库 `test_data/longchat.jsonl`，带宽轨迹取自 `test_data/bw.jsonl`；问答使用脚本生成并插入上下文的简单事实。完整参数已写入 manifest，不依赖后续版本的默认参数。

| 实验 | 本批次条件 | 测量规模 |
| --- | --- | --- |
| 质量与体积 | 20 条 prompt × 512/1024/1536 token；FP16、uniform4/8、CacheGen 1/2/3 | 60 个续写上下文；10 个事实 × 3 个长度 × 3 个位置 = 90 个问答条件，每种方法均评估 |
| 首 token 延迟（TTFT） | 3 条 prompt × 3 个长度 × 0.1/1/10 Gbps；文本、uniform8、CacheGen 2/3 | 每个条件预热 2 次、计时 10 次，共 1080 次正式计时 |
| 带宽自适应 | 5 个事实、1024-token 上下文、256-token 分块；5 条轨迹；6 种策略 | 带宽倍率 1/0.01 × SLO 0.2/0.5 秒，共 4 个任务；预热 1 次、计时 3 次，共 1800 次正式计时 |

续写 NLL 使用 32-token 后续文本中的第 2–32 个 token。90 个问答条件来自 10 个事实，不是 90 个独立事实；重复计时和同题的不同轨迹也不增加独立问答样本量。

## 本批次结果概览

- **质量与体积：**CacheGen 3 档在 512/1024/1536 token 下，相对原始 KV 的体积压缩比分别约为 3.56/3.93/4.09；对应平均 NLL 变化为 −0.00535/+0.00285/+0.00254，90 个问答条件全部答对。它在三个长度上均通过预设的描述性质量门槛（平均 NLL 增加不超过 0.02，原始 KV 答对题目的保留率至少 90%），这些代理指标不构成普遍质量保证。
- **TTFT：**在 9 个“长度 × 带宽”组合中，CacheGen 3 档的中位 TTFT 均低于 uniform8，但仅有 3 个组合低于文本重算；优势取决于本机计算成本、上下文长度与有效吞吐。设置的带宽不等于实测吞吐，详见 CSV 的 `mean_effective_bandwidth_gbps`。
- **自适应：**本批次未显示策略相对静态策略和文本重算的普遍优势。例如带宽倍率 0.01、SLO 0.5 秒时，自适应有 3/75 次超限，静态策略与文本重算均为 0/75；这 75 次是重复计时观测，不是 75 个独立问题。

更完整的分组结果和解释见 [report.md](report.md)。现有编码器没有论文的 anchor/delta 阶段；TTFT 通过本机限速 socket 收完数据再解码，不含离线编码时间，也没有传输与解码重叠。自适应的网络时间来自轨迹估算，并非真实服务器间的传输测量。

## 文件索引

| 文件 | 内容 |
| --- | --- |
| [manifest.json](manifest.json) / [metadata.json](metadata.json) | 固定实验配置、命令、完成状态、耗时、环境与文件指纹 |
| [report.md](report.md) | 自动生成的配对质量、TTFT、自适应报告 |
| [quality.json](quality.json) / [quality.csv](quality.csv) | 逐例质量与体积记录 / 按长度与方法汇总 |
| [ttft.json](ttft.json) / [ttft.csv](ttft.csv) | 逐次 TTFT 记录 / 中位数、P95、吞吐与速度对比 |
| [adaptive-scale-1-slo-0.2.json](adaptive-scale-1-slo-0.2.json) | 原始带宽倍率，SLO 0.2 秒 |
| [adaptive-scale-1-slo-0.5.json](adaptive-scale-1-slo-0.5.json) | 原始带宽倍率，SLO 0.5 秒 |
| [adaptive-scale-0.01-slo-0.2.json](adaptive-scale-0.01-slo-0.2.json) | 带宽倍率 0.01，SLO 0.2 秒 |
| [adaptive-scale-0.01-slo-0.5.json](adaptive-scale-0.01-slo-0.5.json) | 带宽倍率 0.01，SLO 0.5 秒 |
| [adaptive.csv](adaptive.csv) | 四个自适应任务的策略汇总 |

本机另存各任务同名的 `.log` 文件，保存运行输出；日志不随仓库归档。

## 该机器上的便利入口

本机的 `mini_repro/run_local.sh` 使用仓库内 `.venv/bin/python`，依赖 `.venv/cuda-12.1` 编译工具链和 `.venv/huggingface` 模型缓存；脚本设置离线模式、`TORCH_CUDA_ARCH_LIST=8.9`、`MAX_JOBS=2`，扩展缓存位于 `.venv/torch_extensions`。

保留该环境时，在仓库根目录运行：

```bash
bash mini_repro/run_local.sh reproduce
```

该命令按当前默认配置创建新的 `results/reproduction/<UTC时间>/` 批次，不覆盖本目录。跨机器安装、执行及参数配置见 [mini_repro/README.md](../../../mini_repro/README.md)；与本批次比较时，应核对 manifest 中保存的参数和 metadata 中的版本。
