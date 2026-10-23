# Mistral-7B：如何进行复现

先完成[环境验证](MISTRAL_ENVIRONMENT.md)。仓库包含作者原始脚本和我们补充的 `mini_repro` 实验框架。本文分别说明它们的功能、命令和论文实验对应关系，避免把原始脚本的限制当成整个仓库的限制。

## 1. 命令与论文实验的对应关系

图号依据 [CacheGen 论文 arXiv v6](https://arxiv.org/html/2310.07240v6)（2024-07-19）。仓库的[作者实验说明](../../sigcomm_ae.md)将主要脚本归为图 8/9 的实验入口；下面进一步按实际输出区分。

| 本文步骤 / 命令 | 论文位置与实验 | 当前命令提供的内容 |
| --- | --- | --- |
| `main.py` | 图 8/9 共用的 KV 准备步骤 | 原始 KV 文件，不直接产生图表指标 |
| `run_cachegen.py` | §7.2、图 9：KV 体积与生成质量 | CacheGen 编码体积；开启评分后得到质量指标 |
| `run_quantization_baseline.py --bins 256` | §7.2、图 9：默认量化对照 | 256 档均匀量化的 KV 文件体积和质量；对应 8bit 量化基线配置 |
| `run_vanilla.py` | §7.2、图 8/9 的文本重算 / 质量参照 | 文本重算生成；开启评分后得到原模型质量参照，不输出 KV 压缩体积或 TTFT |
| `mini_repro.ttft` | §7.2、图 8：TTFT 与生成质量 | 已实现单机限速 socket、恢复和首 token 计时；模型配置固定为 TinyLlama，接入 Mistral 时需适配 |
| `run_adaptation.py --encode` | §7.4、图 13 的准备步骤 | 每个分块的多个编码等级 |
| `run_adaptation.py` | §7.4、图 13：SLO 违约率与质量 | 当前自适应策略的估算违约率和质量；静态策略、量化基线需另作对照 |
| `mini_repro.adaptive` | §7.4、图 13 的自适应对比框架 | 已有自适应、静态预算、固定等级和文本重算对照，以及质量和 SLO 汇总；模型配置固定为 TinyLlama |

图 8 是 TTFT，图 9 是 KV 体积，图 13 是带宽自适应。图 11 的带宽敏感性和图 14 的开销微基准需要各自的测量设置，不能从下列单次运行日志直接推得。[论文 §7.2–7.5](https://arxiv.org/html/2310.07240v6#S7.SS2)

### 已有的实验框架如何复用

`mini_repro` 已经提供三条实验路径及归档流程：

| 模块 | 已有功能 |
| --- | --- |
| `mini_repro.quality_size` | 原始 KV、uniform4/8、CacheGen 1/2/3 的体积与质量对照 |
| `mini_repro.ttft` | 通过线程和 `socketpair` 传送真实字节、限速、校验接收内容，测量恢复和首 token 时间；包含预热、重复、顺序打乱和实际吞吐统计 |
| `mini_repro.adaptive` | 剩余预算策略、静态预算策略、固定 1/2/3 级和文本重算六种策略；真实执行解码或文本重算，记录质量与 SLO 结果 |
| `mini_repro.reproduce` / `report` | 批次执行、完整性检查、元数据、JSON、CSV 和报告 |

这些模块有[已归档的 TinyLlama 运行记录](../../results/reproduction/README.md)。接入 Mistral 应复用上述功能，重点调整 `mini_repro/run.py` 的模型 ID/revision 和编解码元数据、`quality_size.py` 的模型加载方式、`reproduce.py` 的 2048-token 校验与模型缓存定位，以及对应配置和报告说明。仅设置本节后面的 `MODEL` 环境变量不会改变 `mini_repro` 内的模型常量。

TTFT 路径在收完数据后再解码，尚不包含传输—解码重叠；自适应路径使用轨迹估算传输时间，并实测本地恢复和首 token 时间。这些是已有实现的测量口径，不能与“没有传输实验或没有自适应脚本”混为一谈。下面第 3–5、7 节展开的是作者原始脚本入口；`mini_repro` 的运行方式另见[实验说明](../../mini_repro/README.md)。

## 2. 设置运行条件

以下命令在 Bash 中按顺序执行，变量在同一终端中保留：

```bash
cd /home/aaa/CacheGen
source .venv/cachegen-env.sh
set -euo pipefail

export MODEL=mistral-community/Mistral-7B-v0.2
export DATASET=longchat
export END=1
export MAX_GPU_MEMORY=23
export CALCULATE_METRIC=0

mkdir -p results/mistral
RUN_DIR=$(mktemp -d "$PWD/results/mistral/run-XXXXXXXX")
export RUN_DIR
mkdir -p "$RUN_DIR/raw" "$RUN_DIR/encoded" "$RUN_DIR/logs" "$RUN_DIR/cachegen-results"
printf '结果目录：%s\n' "$RUN_DIR"
```

- `END=1` 表示处理编号 0 的一条样本，范围是 `[0, END)`。先核对单样本输出，再将 `END` 改为 50，在新的 `RUN_DIR` 中重新运行。当前 LongChat 文件包含 50 条。
- `MAX_GPU_MEMORY=23` 是针对 24GB 4090D 的加载参数示例。它不是显存占用的硬上限，也不保证长上下文能运行；T4 需按自身显存调整。不要直接沿用原脚本默认的 48。
- `CALCULATE_METRIC=0` 跳过外部评分，适合离线检查生成和 KV 体积。要得到图 9 的“体积—质量”数据，使用 `CALCULATE_METRIC=1`，并保持各方法的样本和设置一致。
- LongChat 的评分调用 `src/utils.py` 中的外部 OpenAI 接口，模型名硬编码为 `gpt-3.5-turbo`。评分需要有效的 `OPENAI_API_KEY`、网络和可用的评测模型。请求错误不能当作有效质量分数；模型推理本身仍使用本地缓存。

先检查实际输入长度，避免把长文本 OOM 误判成环境问题：

```bash
python - <<'PY'
import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL'], local_files_only=True)
with open('test_data/longchat.jsonl') as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
end = int(os.environ['END'])
assert 0 < end <= len(rows)
for index, row in enumerate(rows[:end]):
    print(index, len(tokenizer(row['prompt']).input_ids), 'tokens')
PY
```

作者的 `scripts/7b.sh` 和 `scripts/7b_baseline.sh` 是 50 条样本的快捷入口。下面展开其中的调用，便于控制显存、范围、日志和目录。

## 3. 准备 KV：图 8/9 的公共输入

```bash
python main.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --save_dir "$RUN_DIR/raw" \
  2>&1 | tee "$RUN_DIR/logs/01-prepare-kv.log"
```

该步骤用原模型生成并保存 KV。输出为 `raw/raw_kv_<编号>.pt`，编号 0 还会保存 `raw/raw_kv_0.pkl`。它是后续 CacheGen 和量化基线的共同输入，不是一项独立的论文性能实验。

使用 `--start 0`：`main.py` 只为编号 0 写 pickle，而下游会读取 `raw_kv_<start>.pkl`；`run_cachegen.py` 还直接用文档编号索引列表。原脚本不能直接把非零 `--start` 当作断点续跑。

## 4. 图 9：CacheGen 的 KV 体积与质量

```bash
python run_cachegen.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --save_dir "$RUN_DIR/raw" \
  --encoded_dir "$RUN_DIR/encoded" \
  --results_dir "$RUN_DIR/cachegen-results" \
  --results_str cachegen \
  --calculate_metric "$CALCULATE_METRIC" \
  2>&1 | tee "$RUN_DIR/logs/02-cachegen.log"
```

输出与含义：

- `encoded/<编号>.pkl`：保存编码后的对象。
- `Average size of KV cache: ... MB`：`len(bytes)/1e6` 的平均值，是本脚本报告的编码体积。
- `Average cachegen accuracy is: ...`：仅在 LongChat 开启评分时输出，是质量轴上的指标。

`run_cachegen.py` 内部固定 `QUANT_LEVEL=2`，在终端单独修改同名环境变量会被覆盖。这条命令给出固定设置下的数据点；要得到多个编码设置的曲线，需要明确记录相应代码或配置变更。

这里的 `--results_dir` 只创建目录，不自动生成完整 CSV 或报告。体积和质量汇总主要在上述日志中。

## 5. 图 9：默认量化基线；图 8/9 的质量参照

### 5.1 默认量化基线

```bash
python run_quantization_baseline.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --save_dir "$RUN_DIR/raw" \
  --bins 256 \
  --calculate_metric "$CALCULATE_METRIC" \
  2>&1 | tee "$RUN_DIR/logs/03-quantization.log"
```

`--bins 256` 与作者快捷脚本一致，表示 KV 的 256 档量化。输出 `quant_kv_<编号>.pt` 和 `max_tensors_<编号>.pt`，`Average size is:` 是两者磁盘文件大小之和的平均值；开启评分后输出 `Average quantization accuracy is:`。

与 CacheGen 比较时要标明统计口径：CacheGen 打印编码对象的长度，量化基线打印两个 Torch 文件的大小，不要把磁盘上 pickle 包装后的大小与另一方法的纯载荷大小混用。

### 5.2 文本重算参照

```bash
python run_vanilla.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --calculate_metric "$CALCULATE_METRIC" \
  2>&1 | tee "$RUN_DIR/logs/04-text-context.log"
```

它直接从文本生成，不注入已恢复的 KV。开启评分后，`Average vanilla accuracy is:` 用作质量参照；它不输出 KV 体积和 TTFT。

`src/utils.py` 对这三种方法都使用 FastChat `load_8bit=True`。这是模型权重的加载方式，和 `--bins 256`、CacheGen 的 KV 编码等级不同。对照时保持模型加载方式一致。

## 6. 图 8：如何理解 TTFT 对应关系

图 8 对比首 token 延迟与质量，TTFT 包含网络和计算部分；论文中的文本基线使用 vLLM。[论文 §7.1–7.2](https://arxiv.org/html/2310.07240v6#S7.SS1)

上述 `run_cachegen.py`、`run_quantization_baseline.py` 和 `run_vanilla.py` 提供相应方法的功能与质量入口，但它们没有直接输出这一定义下的 TTFT：

- `main.py` 虽定义了开始时间，打印 TTFT 的代码被注释。
- `run_cachegen.py` 从本地文件读取、解码并生成 20 个 token，没有网络传输测量。
- `run_vanilla.py` 使用仓库的 Hugging Face / FastChat 加载路径，并非论文的 vLLM 计时基线。

因此，`time python ...` 或整段日志的耗时不能作为图 8 数据：它会包含模型加载、离线准备、磁盘读写和多 token 生成。

测量图 8 可复用 `mini_repro.ttft` 已有的单机模拟和计时功能：模型预先加载，计时覆盖限速传输、恢复/重算到首个输出 token，离线编码单独记录，预热不计入正式测量，并输出实际吞吐。无需因为使用单机模拟而另加一台服务器。

将模型加载、编解码配置和输入长度适配到 Mistral 后，再以相同模型、样本和设置运行质量与 TTFT 对照。现有 TinyLlama 输出应保留其模型标识；单机模拟方式与模型适配是两件不同的事。

## 7. 图 13：带宽自适应与 SLO

`mini_repro.adaptive` 已实现较完整的策略对比框架，接入 Mistral 时可以直接复用其策略、真实 KV 恢复、文本重算和汇总逻辑。它的网络时间来自轨迹估算，并未复用 TTFT 模块的 socket 传输。下面展开作者原始 `run_adaptation.py` 的命令，两套实现的输出口径需要分别记录。

先为同一分块大小生成三个编码等级，再运行策略：

```bash
mkdir -p "$RUN_DIR/adaptive"
export CHUNK_SIZE=1500

python run_adaptation.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --save_dir "$RUN_DIR/adaptive" \
  --chunk_size "$CHUNK_SIZE" \
  --slo 0.5 --encode \
  2>&1 | tee "$RUN_DIR/logs/05-adaptive-encode.log"

# LongChat 此处需要有效的外部评分接口。
python run_adaptation.py \
  --model_id "$MODEL" \
  --dataset_name "$DATASET" \
  --num_gpus 1 --max_gpu_memory "$MAX_GPU_MEMORY" \
  --start 0 --end "$END" \
  --save_dir "$RUN_DIR/adaptive" \
  --chunk_size "$CHUNK_SIZE" \
  --slo 0.5 --total_traces 5 --calculate_metric 1 \
  2>&1 | tee "$RUN_DIR/logs/06-adaptive-evaluate.log"
```

编码阶段保存 `raw_kv_<编号>.pkl` 和 `<编号>_<块号>_<等级>.pkl`。策略阶段读取 `test_data/bw.jsonl`，输出 `Average violation rate:` 和 `Average accuracy:`。前者统计估算的 `ttft > slo` 的样本比例，后者统计生成质量。

这组命令对应图 13 的“自适应策略”数据点。更改 `--slo` 并固定其他条件可比较不同预算；图中静态策略和量化基线需使用相同条件另外测量。论文图 13 使用 20 条带宽轨迹，示例命令沿用作者脚本的 5 条设置，报告时应标明这一差别。[论文 §7.4](https://arxiv.org/html/2310.07240v6#S7.SS4)

需要按源码理解这些输出：

- `src/utils.py::config_selection` 用编码字节数和轨迹带宽估算传输时间，回退分支累加固定 `chunk_delay=0.2`；不是实测跨服务器 TTFT。
- `merge` 的文本回退从预存原始 KV 中切片；不能把它计为真实文本传输和重算。
- 编码、策略阶段的 `--chunk_size` 必须一致。原 `scripts/adapt.sh` 的编码阶段省略该参数，使用默认 1000，而评测阶段设为 1500；这里显式统一为 1500。
- 最后不足一个完整块的处理也应核对：`merge` 会跳过尾部不完整块，分析结果时需记录长度与分块关系。
- 原脚本在打印准确率时直接除以 `len(average_acc)`，所以此处不使用 `--calculate_metric 0`；该设置会使质量列表为空。LongChat 评测因此需要外部接口可用。

## 8. 保存与检查结果

每批使用独立目录，保存第 2 节变量、完整命令、数据和模型指纹、GPU/驱动、源码提交与改动、实际 token 长度、输出与错误日志。上述脚本会写固定文件名，重复运行同一目录可能覆盖文件。

图 9 汇总每种方法的体积和相应质量；图 8 使用统一边界的逐次 TTFT；图 13 汇总同一 SLO、数据与轨迹条件下各策略的违约率和质量。单条样本用于检查流程，扩大样本后再讨论平均表现。

保留原有模型加载方式和比较条件；如因显存调整数据、上下文长度或批次规模，连同结果一起记录。作者 7B 运行说明以 A40 48GB 为条件，24GB GPU 的实验规模应根据实际占用确定。
