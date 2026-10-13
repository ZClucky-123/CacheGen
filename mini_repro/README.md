# CacheGen 小规模复现实验

本目录使用 TinyLlama 的真实 KV cache，执行压缩体积与质量、传输与首 token 延迟、带宽自适应三组实验。本文说明如何准备环境、执行实验和读取结果；实验条件及评价口径见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，已归档批次的机器配置和实测结果见 [结果索引](../results/reproduction/README.md)。

当前实验使用仓库编解码器和探索性的 TinyLlama 量化配置，尚未实现论文的差分编码、跨服务器传输及传输与解码重叠，不能作为论文完整复现。自适应实验的网络时间按带宽轨迹估算。

## 1. 准备环境

以下命令均在**仓库根目录**执行。需要 Linux 或 WSL2、Python 3.10、与 PyTorch 2.3.1 / CUDA 12.1 兼容的 NVIDIA GPU 和驱动，以及 CUDA 12.1 Toolkit（含 `nvcc`）和 C++ 编译器。CUDA 扩展需要在目标机器的 Python / PyTorch 环境中编译。

创建并激活环境，安装固定版本依赖：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r mini_repro/requirements-mini.txt
```

PyTorch 软件源来自[官方 2.3.1 安装说明](https://pytorch.org/get-started/previous-versions/#v231)。其余依赖版本见 [requirements-mini.txt](requirements-mini.txt)。

指定已安装的 CUDA Toolkit，安装仓库内的 LMCache 和 CUDA 编解码扩展。下面的 `CUDA_HOME` 是路径示例，应改为目标机器的实际位置；PyTorch wheel 自带的 CUDA 运行时不包含编译所需的 `nvcc`。

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
python -m pip install --no-build-isolation --no-deps ./LMCache
python -m pip install --no-build-isolation --no-deps ./LMCache/third_party/torchac_cuda
python -m pip check
```

设置源码及缓存位置。之后每次打开新终端，都需要激活同一 Python 环境并设置这些变量及 `CUDA_HOME`：

```bash
export PYTHONPATH="$PWD:$PWD/LMCache${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.venv/huggingface"
export TORCH_EXTENSIONS_DIR="$PWD/.venv/torch_extensions"
```

## 2. 准备固定版本模型

首次下载需要联网；执行前不要启用后面的离线开关。模型固定为 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`，revision 与实验脚本一致。下载到同一个 `HF_HOME`，不要用 `local_dir` 改变缓存布局：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    allow_patterns=["*.json", "tokenizer.model", "model.safetensors"],
)
PY
```

`snapshot_download` 的 revision 和文件筛选用法见 [Hugging Face 下载文档](https://huggingface.co/docs/huggingface_hub/v0.23.4/guides/download)。准备完成后启用离线运行：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

实验数据已包含在仓库的 `test_data/longchat.jsonl` 和 `test_data/bw.jsonl` 中。合成事实问答由脚本生成。

## 3. 检查与正式执行

已完成上述环境准备后，依次执行：

```bash
# 检查编解码器、计时、策略、结果校验和报告逻辑
python -m unittest discover -s tests -v

# 预览正式配置和命令；不使用 GPU、不创建结果目录
python -m mini_repro.reproduce --dry-run

# 小样本流程验收；其结果不能用于正式实验结论
python -m mini_repro.reproduce --profile mini_repro/profiles/validation4070.json

# 完整执行正式小规模实验
python -m mini_repro.reproduce --profile mini_repro/profiles/formal4070.json
```

配置文件名中的 `4070` 是历史标识，文件内容规定的是样本、长度、带宽、重复次数和质量阈值，可在兼容的 GPU 上执行。正式配置包含：

| 实验 | 规模 |
| --- | --- |
| 压缩体积与质量 | 60 个续写上下文、90 个问答条件；每个条件比较 6 种方法 |
| 传输与首 token 延迟 | 3 条文本 × 3 种长度 × 3 档带宽 × 4 种方法 × 10 次测量，共 1080 次；预热另计 |
| 带宽自适应 | 5 个事实 × 5 条轨迹 × 6 种策略 × 3 次测量 × 4 组带宽倍率/时限，共 1800 次 |

默认每次创建 `results/reproduction/<UTC时间>/`，已有同名批次会报错。可用 `--output-root` 和 `--run-id` 指定输出位置；修改实验设计时，复制配置文件并用 `--profile` 指向副本。上下文、问题及生成内容必须在 TinyLlama 的 2048-token 窗口内。

## 4. 单独执行实验

调试或补充测量时，可以单独运行以下模块。显式指定新输出路径，避免覆盖历史结果：

```bash
mkdir -p results/manual
python -m mini_repro.quality_size --samples 20 --lengths 512 1024 1536 --continuation 32 --qa-per-position 10 --results results/manual/quality.json
python -m mini_repro.ttft --samples 3 --lengths 512 1024 1536 --bandwidth-gbps 0.1 1 10 --cachegen-levels 2 3 --warmup 2 --repeats 10 --results results/manual/ttft.json
python -m mini_repro.adaptive --cases 5 --length 1024 --chunk-size 256 --traces 5 --slo 0.5 --warmup 1 --repeats 3 --results results/manual/adaptive.json
```

上述命令再次运行会覆盖各自的同名文件。单项输出不包含完整批次的 manifest、元数据及自动汇总；需要可比较的整套记录时使用 `mini_repro.reproduce`。模块的完整参数可通过 `python -m mini_repro.<模块名> --help` 查看。

## 5. 读取与归档结果

完整批次包含以下文件：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 实验配置、命令、任务状态及耗时 |
| `metadata.json` | 机器与软件版本、Git 提交、模型/源码/数据指纹 |
| `quality.json`、`ttft.json`、`adaptive-*.json` | 逐例质量和逐次计时记录 |
| `quality.csv`、`ttft.csv`、`adaptive.csv` | 按实验条件汇总的指标 |
| `report.md` | 质量门槛、压缩率、首 token 延迟及自适应结果 |
| `*.log` | 任务日志，默认仅保留在本机 |

先确认 `manifest.json` 的状态为 `completed` 且所有任务完成，再阅读 `report.md`。脚本会核对预定样本数、方法数和重复次数；中断或失败的批次保留现场，但不能作为完整实验结论。

重建已有完整批次的报告时，执行 `python -m mini_repro.report results/reproduction/<批次目录>`。这会更新该批次的汇总文件；原始逐次记录保留。

仓库仅归档经过核对的完整正式批次。新运行目录默认被 `.gitignore` 忽略；归档时应确认用途为 `formal-small-scale`、所有任务完成，再添加目录白名单和结果索引。机器配置、实际耗时与结果解释应写在该批次目录内。验收、冒烟、中断及零散调试结果保留在本机。

`run_local.sh` 和 `run_wsl.sh` 是已有机器的环境快捷脚本，包含特定路径约定。跨机器执行使用本文的 Python 模块入口；当前归档批次的本机快捷命令见[结果索引](../results/reproduction/README.md)。
