# CacheGen首轮实验与结果分析指导

## 1 实验目标与适用版本

适用代码版本为 `ce56636dc041b81f2a05adeebeac3eaebbf4a823`，分支为 `feat/cachegen-joint-optimization`，协议为 `cachegen-joint-v2`。原接口与算法对照基线为 `d6b6626014c30b64011e4348c02510fdfbdb7204`。后续版本记录实际提交号，并使用独立批次保存结果。

首轮实验应形成四类证据：三档三编码的设备数值正确性；真实生成、封装、传输和恢复成本；质量与期限预测的独立验收；相同条件下与原codec及系统路线的配对结果。目标是在质量门槛和TTFT期限内评估最高可行质量及完整请求耗时，三种表示的优势区间由实测确定。

代码交付边界、环境依赖及错误回传格式见[交接说明](handoff.md)；完整协议与运行约束见[整体方案v2.2](CacheGen整体优化与代码修改方案.md)。

## 2 执行前固定条件

| 项目 | 固定内容 |
| --- | --- |
| 版本 | 源码提交号、joint/base配置、数据身份；整个批次使用同一执行版本 |
| 设备软件 | 单张RTX4090D/计算能力8.9，Linux、Python 3.10.12、PyTorch 2.3.1+cu121、CUDA 12.1、Transformers 4.42.3 |
| 模型 | 固定Mistral-7B-v0.2 revision、FastChat8权重、FP16 KV和既定稳定执行设置 |
| 请求几何 | 32层、8个KV heads、128维head；256-token应用块；T=1024/2048/4096 |
| 质量 | q1/q2/q3；4K自动质量按静态准入使用q2/q3；同q比较编码收益 |
| 网络与期限 | 默认目标带宽0.1/1/10 Gbps，SLO为0.2/0.5/1/2秒；记录真实吞吐与实际TTFT |
| 缓存与调度 | 每上下文256 MiB CPU包缓存；单GPU默认流、单链路、接收容量1；分别测serial/pipeline |

固定[来源划分](../configs/source-split.json)：正式测试0～4、成本标定5～9、残差拟合10～14、独立验收15～19。四组来源互斥，各阶段只承担对应的数据用途。

代码、模型、执行参数或数据变化后核对fingerprint及档案失效范围。复现组的新版本先完成差异评估，再安排另一批次的适配与对照。

## 3 推荐执行顺序

### 3.1 预检、构建与设备正确性

在服务器仓库根目录设置已有环境路径：

```bash
export CUDA_VISIBLE_DEVICES=0
export CACHEGEN_PYTHON="$PWD/.venv/bin/python"
export CUDA_HOME="$PWD/.cuda-12.1"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME="$PWD/.venv/huggingface"
JOINT_RUN_ID=joint-v2-round01

bash optimization/scripts/server.sh plan "$JOINT_RUN_ID"

JOINT_RUN_DIR="optimization/output/$JOINT_RUN_ID"
mkdir -p "$JOINT_RUN_DIR"
git rev-parse HEAD > "$JOINT_RUN_DIR/code-commit.txt"
git status --short > "$JOINT_RUN_DIR/worktree-status.txt"
nvidia-smi > "$JOINT_RUN_DIR/gpu-environment.txt"

bash optimization/scripts/server.sh build "$JOINT_RUN_ID"
bash optimization/scripts/server.sh gpu-check "$JOINT_RUN_ID"
```

环境路径按服务器实际安装位置设置。`plan`只展开任务规模；`gpu-check`要求设备用例实际执行、全部通过且零跳过。重点核对原CDF、量化符号、最终FP16 KV与原codec首token对照，以及尾块、全B回退、源/恢复缓冲隔离。设备或数值检查失败时，保存最小复现条件与日志，修复后再进入性能阶段。

默认配置预检计数为4760个成本状态、95200个实测样本及23800个预热样本；正式矩阵为294000个稳定带宽请求和28800个变化带宽请求。准备工作、先前缓存请求、残差与质量验收另计。实际规模以本批次 `plan` 输出为准，GPU时间与费用根据短测估算后安排。

### 3.2 成本诊断、完整标定与独立验收

首轮先观察少量成本状态的真实耗时、搬运和内存，再安排完整标定。CLI的 `calibrate --state-limit N` 可生成诊断成本档案，标记为 `diagnostic_costs`；后续流程需要完整的适用成本状态。命令参数与环境、共享GPU锁的使用见[实验指南](implementation-and-experiments.md)和[服务器脚本](../scripts/server.sh)。

各阶段分别执行：

```bash
bash optimization/scripts/server.sh calibrate "$JOINT_RUN_ID"
bash optimization/scripts/server.sh fit-residual "$JOINT_RUN_ID"
bash optimization/scripts/server.sh verify "$JOINT_RUN_ID"
bash optimization/scripts/server.sh freeze "$JOINT_RUN_ID"
```

成本按q、表示、N、prefix、前一GPU任务、缓冲首次/复用/扩容、执行模式和实际CDF tag区分。残差只使用拟合来源；独立验收检查固定版本的成本模型与余量在新来源上的质量和覆盖率。

通过条件为NLL增量≤0.02、EM保留率≥0.90、预测经验覆盖率≥0.90；各有效分组或回退分组至少5来源、20请求样本，正常请求意外失败使验收未通过。EM原始正确样本为零时，该组需要核查输入、评分和基准表现并补齐证据。`verification.json` 的ready状态决定后续正式入口。

### 3.3 配对比较与结果汇总

验收并冻结后，优先检查固定q2的原codec、`ac_count`、`ac_cdf`、`fixed_bits`以及两编码/三编码联合组，分析表示本身的收益；随后分析自动质量、文本、uniform4等系统对照。按相同source、T、执行模式、带宽、SLO、缓存条件、轨迹和重复编号配对。

```bash
bash optimization/scripts/server.sh benchmark "$JOINT_RUN_ID"
"$CACHEGEN_PYTHON" optimization/scripts/analyze.py \
  --run-dir "optimization/output/$JOINT_RUN_ID"
```

组别筛选由CLI的可重复 `--group` 参数实现，可先用相同筛选条件执行 `--task inspect --plan` 核对规模。`server.sh`接受阶段和run_id，筛选参数通过CLI在相同环境与共享GPU锁下执行。固定q诊断另加 `--fixed-quality` 和 `--diagnostic-only`，保存于 `diagnostic-benchmark/`；正式记录位于 `benchmark/`，分析诊断结果时也加 `--diagnostic-only`。

缓存条件按实际生成历史区分cold、warm、mixed、quality_change和representation_miss。关注 `seed_request` 的先前工作、实际驻留和 `seed_and_target_ttft_s`，并核对各对照组的计时边界。进程重启后重新取得原始KV和实际包，manifest中的配方用于描述来源与候选。

## 4 优先关注的信息

| 信息 | 主要记录或派生方式 | 分析重点 |
| --- | --- | --- |
| 端到端结果 | `ttft_s`、`slo_met`、`source_prefill_s`；summary.csv、paired.csv | 首次与复用的TTFT中位数/P90、实际SLO满足率；源端prefill单列 |
| 质量 | `verification.json`、`verify/qa.jsonl`、benchmark的 `nll_delta` | 各T、路线、缓存状态和轨迹的NLL/EM；自动质量同时报告q分布 |
| 字节与实际吞吐 | `network[].wire_bytes`、`bandwidth_bps`及网络起止纳秒 | 完整wire字节、实测bit/s、目标带宽与实际吞吐的差异 |
| 处理与等待 | `generation[].g_encode_ns/c_seal_ns`、`events`、`decisions[].select_ns` | 生成/D2H、封装、恢复/H2D、选择和资源等待；识别端到端关键路径 |
| 按需行为 | `generation`、`codec_calls`、决策与发送事件 | 未选分支调用数、缓存复用、额外替代生成、回退、失败及未发送工作 |
| 选择与预测 | `decisions`中的q/m、route、F50/F_base/F_safe及最终首token时刻 | 路线胜出比例、质量切换、大小偏差、正向预测误差、独立覆盖率 |
| 内存与生命周期 | `source_bytes`、`cache_peak_bytes`、`temporary_peak_bytes`、`cuda_peak_bytes` | 原始源、CPU包/临时wire及PyTorch显存峰值；定位容量或生命周期异常 |

阶段耗时可能重叠，TTFT使用完整请求墙钟。网络服务从NET_START到DATA_AVAILABLE计算，生成等待、接收槽等待和GPU等待按事件另看。显存峰值包含进程中的模型及相关驻留；pinned容量、内核耗时、驱动显存和设备利用率可在异常场景补充定向profiling。

`analyze.py`提供基础分组指标、配对差值和满足条件时的图表。阶段等待、选择分布、预测误差和资源细分从原始记录进一步汇总。`paired.csv`包含配对后的聚合，特定带宽或执行模式的结论需要回到对应条件分层。默认摘要按请求统计；稳健性分析应保留来源级配对结果，重复请求数与独立来源数分别报告。

## 5 如何依据结果选择优化方向

| 观察到的现象 | 优先核查的证据 | 后续方向 |
| --- | --- | --- |
| 字节减少，首次TTFT仍偏高 | G_encode、C_seal、D2H及恢复等待；同q配对结果 | 排查编号处理、封装和搬运成本，以及编码对恢复的阻塞 |
| 位打包在高带宽下优势较小 | 实际吞吐、R_restore、H2D、选择开销与内存分配状态 | 定向检查融合恢复、缓冲复用和主机控制开销 |
| warm明显快于cold | 实际命中与逐出、先前生成成本、替代包次数 | 优先改进首次生成或缓存策略，并分别报告首次/复用收益 |
| 分阶段服务较快，流水线仍慢 | DATA_AVAILABLE到RESTORE_START等待、链路空闲、CPU选择及封装时间 | 检查共享GPU竞争、前瞻窗口和控制线程调度 |
| 实际首token晚于F_safe | 决策时原始预测、大小估计、已完成反馈与后续重选轨迹 | 排查模型遗漏与成本状态覆盖，再在独立来源上重拟合和验收 |
| NLL或EM未达门槛 | 同q位模式、token/前缀、模型设置及QA原始预测 | 优先定位数值与执行路径；质量证据通过后再评价性能 |
| 某表示或联合组持续较慢 | q、完整wire、G/C/R、缓存状态和同条件系统对照 | 如实报告优势区间或选择开销，依据完整成本调整下一版 |

固定质量的编码比较与自动质量的系统比较分别解释。`ac_cdf`的优势区间可能为空；未观察到胜出区间本身也是结果。期限模型的经验覆盖率与实际SLO满足率分别统计。

## 6 一轮结束后的交付与迭代

每轮保留提交号、配置/数据/环境身份、阶段日志、成本与余量档案、验收报告、原始观测及检查点、分组和配对摘要。所有产物位于 `optimization/output/<run_id>`。恢复时使用相同版本和条件，完整记录作为完成依据；部分尾部、冲突重复键及损坏记录的处理见[交接说明](handoff.md)。

实验小结至少给出：通过与待解决项目；各质量和场景的TTFT/质量/字节结果；主要瓶颈及证据；资源峰值和失败情况；下一轮拟修改范围、预计收益来源及所需重测。实测数值、协议精确字节和条件推算分别标注。

后续优化按独立提交与批次比较。记录触发现象、对应请求和事件、瓶颈假设、拟改模块、失效档案、重测范围及新增GPU时间/费用估计。成本与残差继续使用各自来源集，正式测试来源用于比较和报告。影响数值、协议、缓存键、调度或成本的更改，按实际影响重新完成设备正确性、标定与独立验收后再冻结。
