# CacheGen 本机小规模验证

本实验在 D 盘的 Ubuntu WSL2 中，用 TinyLlama-1.1B 的真实 KV 缓存调用作者的 `CacheGenSerializer` / `CacheGenDeserializer`。这是**方法验证**，模型和分层量化参数均与论文实验不同，不能将结果当作论文的 7B/70B 基准复现。

## 本次配置

- 机器：RTX 4060 Laptop 8 GiB；Ubuntu 22.04 WSL2 虚拟磁盘 `D:\WSL\Ubuntu\ext4.vhdx`。
- Python 3.10；PyTorch 2.3.1+cu121；Transformers 4.42.3；CUDA `nvcc` 12.1.105；作者的 `LMCache/third_party/torchac_cuda` 源码本机编译。
- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，FP16；权重 2,200,119,864 字节，SHA-256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`。
- Python 环境：`/home/lucky_1123/.venvs/cachegen-mini`；模型缓存：`/home/lucky_1123/.cache/huggingface`。这两条路径都位于 D 盘 WSL 虚拟磁盘内。
- 软件及本地运行费用：0；没有使用付费 API。

配置的 22 层边界为 7/14/22，key bins 为 32/16/16，value bins 为前 2 层 32、其余 16。这是把作者的 32 层 7B 配置按层数缩放的探索性配置，**未经论文校准**。

## 数据和指标

- 压缩：作者仓库 `test_data/longchat.jsonl` 的前 3 个 prompt，每个截取 512 和 1024 token，共 6 个 KV 缓存。截断片段只测真实模型 KV 的编码、解码、体积和平均绝对误差，不用原数据集标签计算准确率。
- 问答：5 个本地构造的短事实，各置于 512-token 上下文末尾。对相同问题比较原始 KV、作者 CacheGen KV 和统一 8 位量化 KV 的贪心输出。准确率只表示答案短语是否出现在输出首行中，不能反映 LongChat 全集或复杂长上下文表现。
- 字节数：原始 KV 和统一 8 位量化均使用 `torch.save` 容器计数，包含元数据；CacheGen 使用作者 `to_bytes()` 返回的完整 pickle 字节数。统一 8 位量化按作者 `src/utils.py` 的每 token、每组 256 通道最大值公式实现，并计入比例因子。
- 时间：单次 GPU 同步后实测的编码、解码时间。首次调用包含一些预热开销；这些时间不能作为稳定吞吐基准。

## 已测结果

| 长度 | 样本数 | 原始 KV | 统一 8 位 | CacheGen | 原始 / CacheGen | 8 位 / CacheGen |
|---:|---:|---:|---:|---:|---:|---:|
| 512 token | 3 | 11.54 MB | 5.81 MB | 平均 2.65 MB | 平均 4.35× | 平均 2.19× |
| 1024 token | 3 | 23.07 MB | 11.63 MB | 平均 4.68 MB | 平均 4.93× | 平均 2.49× |

5 个合成问答中，原始、统一 8 位和 CacheGen 的答案短语均为 **5/5**。CacheGen 的输出有时在答案后继续生成其他文字；完整原文和逐例指标保存在 [`results/tinyllama/metrics.json`](../results/tinyllama/metrics.json)。CacheGen KV 平均绝对误差约为 0.21（统一 8 位约 0.013），说明本配置以更大误差换取更小体积；这个小测试不能证明一般场景的回答质量保持不变。

峰值 PyTorch GPU 分配约 2.44 GB（不包括驱动及其他进程占用）。本机环境约占 D 盘：venv 5.1 GiB、模型缓存 2.1 GiB、pip 下载缓存 2.8 GiB；CUDA 编译器和 WSL 系统文件另计。`ext4.vhdx` 当前约 18.0 GB，迁移前备份约 7.15 GB，均在 D 盘。需要清理空间时，先核对路径后再处理备份或缓存。

## 重新运行

已完成本机依赖安装，脚本设置离线模式，运行时不下载模型或包。在 Ubuntu WSL 中：

```bash
bash /mnt/d/a_study/大三上/大模型网络/code/CacheGen-mini/mini_repro/run_wsl.sh
```

可用 `--compression-limit 1 --qa-limit 1` 做快速检查。输出覆盖 `results/tinyllama/metrics.json`。如从零搭建，请在 D 盘 WSL 内创建虚拟环境，安装 `requirements-mini.txt`，配置 CUDA 12.1 `nvcc`，并从作者 `LMCache/third_party/torchac_cuda` 编译安装扩展；本实验优先使用清华 pip 镜像和 `hf-mirror.com` 获取依赖与模型。构建和下载前应重新核对目标路径、可用空间及下载规模。

## 三组后续实验

这三组脚本复用以上离线环境和仓库内的数据。运行前在 WSL 中进入本工作树，或把下面的 `mini_repro/run_wsl.sh` 换成它的绝对路径。所有输出写到本工作树的 `results/tinyllama/`，不会覆盖上面的 `metrics.json`。

```bash
bash mini_repro/run_wsl.sh quality --samples 20 --lengths 512 1024 1536 --continuation 32 --qa-per-position 10
bash mini_repro/run_wsl.sh ttft --samples 3 --lengths 512 1024 1536 --bandwidth-gbps 0.1 1 10
bash mini_repro/run_wsl.sh adaptive --cases 5 --length 1024 --chunk-size 256 --traces 5 --slo 0.5
```

- `quality` 输出 `quality_size.json`。对同一 LongChat 上下文，比较 FP16、真正打包的统一 4 位、统一 8 位、CacheGen 1/2/3 级的完整传输字节数及未见后续文本的平均负对数似然。KV 不保存最后一个上下文 token 的 logits，因此困惑度代理指标从后续文本**第二个** token 开始计分。另测事实在上下文开头、中间、末尾时的逐题答案，并单列原始 KV 本身答对的题数。三档 TinyLlama 参数由 7B 设置缩放而来，未经论文校准。
- `ttft` 输出 `ttft.json`。比较文本重新 prefill、统一 8 位 KV、CacheGen 2 级 KV。相同的上下文先离线编码，经单机 socket 限速传送后，分别测传输、恢复和首 token 推理时间；离线编码耗时单列，并按方法、带宽、上下文长度汇总中位数与 P95。结果是**本机受控链路测量**，不能代表两台服务器之间的真实网络性能；高带宽档还会受到 Python 线程及系统调度开销影响。文本会往返分词校验，确保三条路径使用相同上下文。
- `adaptive` 输出 `adaptive.json`。把完整上下文 KV 分块，预编码三档位流，读取 `test_data/bw.jsonl` 的 Gbps 轨迹，比较自适应、固定 1/2/3 级和文本回退。每条记录列出选择的等级、按字节/带宽**估算**的传输时间、本机实测的恢复与首 token 时间、SLO 超限和答案。`--bandwidth-scale 0.01` 可人为放慢轨迹，检查是否触发降级及回退；缩放后的轨迹不能当作论文原带宽。

完整运行的样本数可通过参数控制。先用 `quality --samples 1 --lengths 512 --continuation 8 --qa-per-position 1`、`ttft --samples 1 --lengths 512 --bandwidth-gbps 1` 和 `adaptive --cases 1 --length 512 --traces 1` 检查环境。质量实验应比较**同等质量下**的体积，不能仅以压缩比宣称方法有效；自适应脚本的网络时间是轨迹估计值，不是线上 TTFT。

## 本机实验结果索引

完整复现记录（配置、依赖、实验与论文差异）见 [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)；结果文件索引与核心数值见 [results/tinyllama/README.md](../results/tinyllama/README.md)。
