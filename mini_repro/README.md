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
