# CacheGen 本机小规模实验结果（2026-10-05）

本目录保存 TinyLlama-1.1B 在本机 D 盘 Ubuntu WSL2 中运行 CacheGen 的原始 JSON 结果。实验目标是检查作者方法的压缩、延迟和自适应规律；模型、数据规模、网络与论文评估不同。环境配置和脚本说明见 [mini_repro/README.md](../../mini_repro/README.md)，完整复现记录见 [REPRODUCTION_REPORT.md](../../mini_repro/REPRODUCTION_REPORT.md)。

## 文件索引

| 文件 | 用途 | 规模 |
| --- | --- | --- |
| [quality_size.json](quality_size.json) | 正式质量与压缩率实验 | 60 个 LongChat 续写上下文、90 个合成事实问答；6 种缓存方法 |
| [ttft.json](ttft.json) | 正式首 token 延迟实验 | 3 条 LongChat × 3 个长度 × 3 档带宽；文本、统一 8 位、CacheGen 2 档 |
| [adaptive.json](adaptive.json) | 正式自适应实验 | 5 个合成问答 × 5 条带宽轨迹；自适应、固定 1/2/3 档、文本回退 |
| [adaptive_low_bandwidth.json](adaptive_low_bandwidth.json) | 低带宽分支诊断 | 1 个合成问答 × 5 条轨迹，原带宽乘以 0.01 |
| [metrics.json](metrics.json) | 最初的基础验证 | 3 条 LongChat 的 512/1024-token 缓存、5 个合成问答 |
| [quality_size_smoke.json](quality_size_smoke.json) | 质量脚本冒烟测试 | 1 条 512-token 续写、3 个事实位置 |
| [ttft_smoke.json](ttft_smoke.json) | TTFT 脚本冒烟测试 | 1 条 512-token 上下文、1 Gbps |
| [adaptive_smoke.json](adaptive_smoke.json) | 自适应脚本冒烟测试 | 1 个 512-token 问答、2 条轨迹 |

前四个文件是本次分析的依据；其余文件只用于早期环境或脚本检查。重新运行脚本会覆盖对应的默认结果文件，JSON 内保留了运行参数和逐例记录。

## 主要结果

### 质量与压缩率

下表是 60 个续写上下文的平均完整传输字节数。NLL 是未见后续文本的平均负对数似然代理指标，越低越好；问答是本地构造的简单事实题。

| 方法 | 平均大小（十进制 MB） | 平均 NLL | 合成问答答对 |
| --- | ---: | ---: | ---: |
| 原始 FP16 KV | 23.07 | 0.8024 | 90/90 |
| 统一 4 位 | 5.86 | 0.8273 | 90/90 |
| 统一 8 位 | 11.63 | 0.8027 | 90/90 |
| CacheGen 1 档 | 3.60 | 0.8582 | 89/90 |
| CacheGen 2 档 | 4.66 | 0.8286 | 90/90 |
| CacheGen 3 档 | 5.86 | 0.8023 | 90/90 |

相近的测量质量下，CacheGen 3 档相对统一 8 位缩小约 1.98 倍；CacheGen 2 档相对统一 4 位缩小约 1.26 倍。1 档更小，但 NLL 增大且少答对 1 题。该结果支持本机的大小—质量权衡，不能外推到论文的 7B–70B 模型。

### 首 token 延迟

下面是 1024-token 上下文、每组 3 条样本的中位墙钟时间。压缩编码在查询前离线完成，其耗时单列在 JSON 内，未计入 TTFT。传输通过单机限速 socket 模拟。

| 带宽 | 文本重算 | 统一 8 位 | CacheGen 2 档 |
| --- | ---: | ---: | ---: |
| 0.1 Gbps | 156 ms | 973 ms | 434 ms |
| 1 Gbps | 153 ms | 137 ms | 82 ms |
| 10 Gbps | 157 ms | 66 ms | 58 ms |

CacheGen 2 档在全部 9 个“长度 × 带宽”组合中快于统一 8 位，在 1 和 10 Gbps 的 6 个组合中快于文本重算；在 0.1 Gbps 的 3 个组合中，文本重算更快。2 档与统一 8 位的续写 NLL 不同，所以这些延迟结果不能解释为严格的同质量加速。

### 自适应

默认轨迹下，自适应的 25 次决策全部使用 3 档，25/25 合成问答答对，0/25 超过 0.5 秒 SLO。因此默认运行只验证了最高质量档路径，没有给出降档收益。

低带宽诊断把作者仓库 `test_data/bw.jsonl` 的带宽数值乘以 0.01；自适应在 5 条轨迹中选择了较低档或文本回退，0/5 超时。固定 1/2/3 档各为 5/5 超时；纯文本回退为 0/5 超时，而且本次平均估计 TTFT 更低。这个人为缩放场景说明策略分支可运行，不能视作真实网络测量或论文 SLO 收益的复现。

## 如何重跑

在仓库根目录用现有的 D 盘 WSL 环境运行，脚本设置离线模式，不会下载模型或依赖：

```bash
bash mini_repro/run_wsl.sh quality
bash mini_repro/run_wsl.sh ttft
bash mini_repro/run_wsl.sh adaptive
bash mini_repro/run_wsl.sh adaptive --cases 1 --traces 5 --bandwidth-scale 0.01 --results results/tinyllama/adaptive_low_bandwidth.json
```

本次运行用时分别约 4 分 18 秒、42 秒、1 分 33 秒和 28 秒。原始 LongChat prompt 与带宽轨迹来自作者仓库 `test_data/`；事实问答由 `mini_repro/experiment_common.py` 生成。实验结论是方法在本机出现了部分预期规律，尚未验证论文报告的跨模型、真实服务网络下的定量收益。
