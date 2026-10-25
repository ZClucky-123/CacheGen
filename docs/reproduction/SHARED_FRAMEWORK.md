# TinyLlama / Mistral 公共实验框架 v3

当前协议为 `cachegen-shared-v3`，四个实验组、七个任务：质量、TTFT、四组倍率×SLO 自适应、轻量机制消融。评分和质量门槛继承 v2，传输与机制实验有新定义，必须开新批次。旧结果保持原样，不能与 v3 合并；旧运行口径见 [v2 指南](SHARED_FRAMEWORK_V2.md)，原始差异见 [审查表](SHARED_FRAMEWORK_AUDIT.md)。

模块与路径见 [工作区指南](WORKSPACE_GUIDE.md)，实测验收见 [报告索引](../../reports/README.md)，与指定 PDF 的实现边界见 [论文对照](PAPER_ALIGNMENT.md)。当前版本不补做 anchor/delta 或完整论文调度器。

## 配置与入口

| 配置 | 质量正式观测 | TTFT 正式观测 | 自适应正式观测 | 消融正式观测 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `experiments/configs/minimal.json` | 54 | 168 | 28 | 24 | 274 |
| `experiments/configs/small.json` | 120 | 432 | 336 | 24 | 912 |
| `experiments/configs/larger-mistral.json` | 1020 | 9450 | 2100 | 24 | 12594 |
| `experiments/configs/larger-tinyllama.json` | 942 | 9450 | 2100 | 24 | 12516 |

正式观测不含预热：minimal 另有 196 次 TTFT/自适应预热，small 312 次，larger 2590 次；消融每行内部另保存三次编解码计时和一次预热。计时重复、长度/位置/轨迹条件均不是新增独立样本。最终精确矩阵见每批 `selection.json` / `expected.json`。

最小配置质量长度为 128/257，TTFT 应用块 128，覆盖一 token 尾块。小配置为 256/512，应用块 256。Mistral 较大配置自然文本 1024/2048/4096，TinyLlama 为 512/1024/1536。较大批次的启动前修复、实测验证和运行信息见 [大规模运行记录](LARGER_READINESS_20261025.md)。v2 的较大配置预检与耗时估算不代表 v3 的更大矩阵。

两个适配器共用全部实验代码。TinyLlama 使用 FP16 权重，Mistral 保持 FastChat groupwise 8bit 权重；KV 均为 FP16。模型权重量化与 KV 量化分开记录。

```bash
# 只生成计划，不加载模型、不写实验结果
bash experiments/run.sh --model tinyllama --config experiments/configs/minimal.json --dry-run
# 检查本地数据、窗口、资源、文本修补与绘图，不加载权重
bash experiments/run.sh --model mistral --config experiments/configs/small.json --preflight-only
```

完整适配器字典使用 `--config custom.json`，不要同时用整体选择默认适配器的 `--model`。结果进入 `results/<model>/<scale>/<unique-run-id>/`，不会覆盖已有目录。

## 脱离终端运行、查询与恢复

```bash
python3 -m experiments.background start --name tiny-v3-new -- \
  --model tinyllama --config experiments/configs/minimal.json --run-id v3-new
python3 -m experiments.background status --name tiny-v3-new
python3 -m experiments status tinyllama
tail -f .local/launches/tiny-v3-new.log
# Ctrl-b 然后 d 可脱离；关闭 SSH/IDE 不影响服务器上的任务
tmux -L cachegen attach -t tiny-v3-new

# 中断后必须用新的会话名；原批次配置自动读取
python3 -m experiments.background start --name tiny-v3-resume -- \
  --resume results/tinyllama/minimal/v3-new
python3 -m experiments.suite --verify results/tinyllama/minimal/v3-new
```

tmux 不能抵御服务器重启，但持久化目录支持恢复。所有同 GPU 入口应共用 `CACHEGEN_LOCAL_DIR`，GPU 锁竞争返回 75。停止某批时先验证当前 manifest 的 runner PID，发送 TERM 后逐行检查点保留、退出码为 143。

SQLite WAL/FULL 按观测提交。恢复验证配置、代码、数据、选择、模型/tokenizer、CUDA 二进制、依赖和执行环境；变化则拒绝继续旧批次。TTFT 编码结果暂存在该批 `pending-prepared/ttft/`：每个方法/分块只编码一次，串行与流水线复用。中断后读取相同字节并检查 SHA256，全部对应观测提交后才移除暂存。摘要保留包头、字节、哈希和离线成本。不要手工删除未完成批次的暂存。

## ② TTFT：整包参考、分块串行、分块流水线

`ttft.modes` 显式包含 `whole`、`chunk_serial`、`chunk_pipeline`；`ttft.chunk_sizes` 可设置多个应用块大小。`transport.queue_capacity` 是接收队列/接收中消息的容量，默认 1；正在主线程恢复的块不占该队列容量。`transport.pacing_quantum_bytes` 为 null 时按目标带宽自动选择 64KiB 至 1MiB 的发送预算小段，也可固定为正整数。

从完整原上下文的预计算 KV 沿 token 维切块，每块独立编码、恢复和存储 CDF/尺度/形状。共同外层帧保存 magic、元数据长度、payload 长度、块编号、token 起止、方法、形状、CUDA 内部分段。全部头和元数据计入字节。CUDA 内核上限仍为 256 token；新批次为保护每通道 256 字节输出缓冲，采用最多 96 token 的内部算术小段。应用块大小、内部实际小段及内核上限分别记录，尾块保持真实长度。

两种分块模式使用同一连接、相同字节、限速与 KV 拼接代码。串行在当前块恢复并完成 CUDA 同步后才发放下一块发送许可；流水线在主线程取走当前块时释放队列容量，使 CPU 接收线程能准备并接收后续块。发送许可在接收容量及缓冲准备好后才发放，防止 socket 缓冲无限预取。发送/接收线程只做 CPU 工作，GPU 解码、文本 prefill、KV 拼接和推理均由主线程执行。

限速预算在发送前等待，最后不足一段时按实际字节预算等待。校准在模型加载前完成，保留接收端的数据可用时刻、目标及实测吞吐。网络区间为接收缓冲 ready 至完整消息 bytes 可用，包含限速等待、socket 和 bytes 物化，不含接收容量等待及上一块 GPU 等待；连接建立和线程开销仍计入端到端 TTFT。没有修改服务器网卡。

TTFT 从建立连接前开始，到首个生成 token 的 CPU 整数 ID 可用为止，GPU 阶段均确认完成。离线 prefill/编码与后续完整回答生成排除在 TTFT 之外。文本按同一 token 区间发送 `utf8_token_patch_v1`，接收端实际重分词并使用收到的修补信息，再在已恢复前缀上逐块 prefill；修补字节也计入体积。整包参考独立报告，整包文本继续使用原单次“上下文＋问题”优化 forward；两种分块文本均采用逐块 prefill 后仅处理问题。流水线主比较为同块同字节的串行模式。

每次记录 receive/send/data_available/restore 起止、容量/发送许可/主线程队列等待、队列停留、发送阻塞、限速等待及重叠区间。重叠表示有效接收区间（含限速等待）与主线程恢复区间相交，不代表真实 NIC 与 GPU 的硬件并发证据。

正确性覆盖所有方法与真实条件：传输字节、顺序、位置、尾块、最终 KV 长度；同字节串行/流水线 KV 完全一致且多 token 输出一致；原始 KV 分块重组完全一致；文本逐块 prefill 与整段 prefill 数值及生成检查。见 `ttft_checks.json`。串行/流水线逐对哈希或首 token 不同，汇总拒绝通过。

## ③ 串行自适应：当前带宽与历史吞吐

保留 `adaptive`（本块设定带宽）、`adaptive_static`、`fixed1/2/3`、`text`，新增 `adaptive_history`（历史吞吐版本）。两种动态策略调用完全相同的候选档位、剩余预算、全剩余块成本预测、首 token 预留和文本回退规则，只替换带宽输入。本地启发式仍有别于论文调度器。

`adaptive.throughput_estimator=previous_completed_message`，首块固定使用 `adaptive.initial_bandwidth_gbps`（默认 1），不得根据本块轨迹初始化。随后用“上一块完整消息字节数×8÷有效传输区间”。历史策略函数只接受带块编号、字节和有效区间的已完成记录，无法读取本块/未来轨迹。轨迹仅在决策完成后用于执行传输，实际设定带宽与吞吐在决策中事后记录。

默认 `socket_serial`：每块单独连接、实际传输、主线程完成恢复后再决定下一块，不与流水线混合。保留块级轨迹含义，每块期间设定带宽固定。v3 不接受 `trace_estimate` 伪装为历史实测反馈；旧估算流程留在 v1 历史入口。

每块保存估计值及来源块、设定带宽、实测吞吐、决策时刻及耗时、剩余预算、各候选剩余/当前成本、档位、回退、传输与恢复区间。请求保存 TTFT、SLO 违约、完整答案/ID/EM/F1、决策序列及完整回答延迟。SLO 只作用于 TTFT。缓存块来自完整原上下文，后层 KV 保留原始前缀依赖；文本回退实际在已恢复前缀上计算。

## ④ 轻量机制消融

明确任务类型为 `ablation`。默认一个自然源文档与一个独立事实的三个位置，单一声明长度；与原质量任务使用相同案例，保留原 raw/uniform4/uniform8/CacheGen1/2/3 基线不改名、不替换。

`ablation.precisions` 默认 `uniform16`、`uniform32`、`layered2`（可另加 `layered1`）；名字中的 16/32 表示 bins，实际符号位宽分别为 4/5 bit。`ablation.entropy` 固定包含 `packed` 与 `arithmetic`。

每个精度配置只调用一次现有按 token/层的最大绝对值尺度量化，产生共同的符号、尺度、形状及每层 bins。两条路径分别执行实际 4/5-bit 位打包及原 CUDA 分组算术编码（CDF、stream lengths 完整计入），使用相同 torch 容器规则和共同反量化。每次重复逐符号比对全部层/通道/token，恢复 KV 要与共同参考完全一致，再计算完整续写 NLL 与 QA；两条熵路径的 NLL/生成输出也必须相同。

统一/分层仅改变每层 bins 分配，尺度规则、编码、概率分组和容器保持一致；使用现有 12/16/32 符号范围，不改 CUDA 内核。统一 16/32 两端和分层 2 的不同平均位宽均明示，不能仅根据更高精度带来的质量提高断言分层有效。

输出纯符号流字节、CDF tensor 字节、完整容器字节、共同量化耗时、排除共同量化的编码耗时及解码耗时。每行保存全部计时重复、符号/尺度哈希、恢复一致性、NLL 或 QA，便于区分熵收益与量化质量。

## 汇总、归档与质量边界

`ttft.csv` 按方法、模式、应用块大小、队列容量、长度和带宽分组；`ttft_pairs.csv` 保存逐源/重复串行÷流水线速度比，`pipeline_comparison.csv` 汇总，数值小于 1 表示流水线更慢。`ttft_timeline.csv` / `ttft_overlap.csv` 保存全量时间线及相交区间。示例图只展示一对实际条件，全量证据在 CSV/JSONL。

`adaptive.csv` 按策略、估计器、块大小和任务分组。`ablation.csv` 分开精度/熵编码，`entropy_pairs.csv` 保存同符号对照。PNG/SVG/PDF 覆盖原三类图及流水线配对、接收/恢复时间线、消融体积—质量图。

配置校验、观测 ID、指纹、矩阵和恢复均包含新增维度，未知任务种类直接报错。每行必要计时、NLL 目标、答案及块连续性均需校验；只有行数齐全仍不能标记完成。

QA 继续采用 `lstrip('\n').split('\n')[0]`、标准规范化 EM/F1、仅原数据别名、EOS/生成上限停止。完整答案与生成 IDs 保留。NLL 评分全部续写 token，最后上下文 token 在恢复 KV[0:L-1] 上重放。v1 漏首目标分数不混用，负 ΔNLL 不能视为质量改善。原质量门槛继续为同长度 ΔNLL≤0.02 且 raw-correct 受控题保留率≥0.9，无 raw-correct 则不可评估。

离线成本是每上下文/精度准备一次的注记，不能按重复计时累加；自适应准备 wall time 包含 raw QA、预热与校验，不等同最小生产建缓存成本。按源文档/事实整体重采样保留关联，少量源的区间和 P95 仅作描述。

归档包含配置、数据、源码/依赖快照、权重/tokenizer 与 CUDA 指纹、选择、原始输出、事件、检查点、图表和 SHA256 清单。新增加的 `chunks.py`、`ttft.py`、`feedback.py`、`mechanisms.py`、`ablation.py`、`analysis_v3.py` 均自动进入源码快照和指纹。失败/中断保留原目录，协议或代码变化需要新 run ID。

## 较大配置与启动前验证

```bash
bash experiments/run.sh --config experiments/configs/larger-mistral.json --preflight-only
python3 -m experiments.background start --name mistral-v3-larger-new -- \
  --config experiments/configs/larger-mistral.json --run-id v3-larger-new
```

较大 TTFT 三模式使矩阵明显扩展，不能继续使用 v2 的 6–8 小时估计。4096-token Mistral raw KV 约 512MiB，在 0.1Gbps 仅字节预算就约 43 秒/次。磁盘还要容纳一个未完成案例的整包/分块暂存字节，预检会估算并拒绝不足空间。增加长度/块大小/队列前需重新检查窗口、主存、显存和峰值。

最小规模两模型功能验收见 [中文报告](../../reports/shared-v3-20261025/验收报告.md)。早期较大配置仅做资源预检：[Mistral](larger-mistral-preflight-20261025-v3.json)、[TinyLlama](larger-tinyllama-preflight-20261025-v3.json)，均记录 `model_loaded=false` 与 `measurements_executed=false`；这些文件不代表后续源码的完整运行验证。实际大上下文验证和新批次见 [大规模运行记录](LARGER_READINESS_20261025.md)。历史批次角色见 [索引](batch-index-20261025-v3.json)。

### 大上下文 FP16 数值敏感性审计

2026-10-25 较大启动检查发现 `longchat-1-2048` 的自然续写第8个 token 分歧：完整生成选择2859、原始KV复用选择264，参考候选差0.0078125，恰为该值的一次FP16间隔；此前7个token相同。关闭半精度归约或换用math注意力仍会在近并列时产生差异。首个尝试在写入任何正式观测前自动停止，其目录保留。

后续批次对完整生成、原始复用及文本分块执行逐步参考前缀 logits 审计，保留两条实际生成序列及首个分歧。数值容差仍为既有 atol=0.1 / rtol=0.02；不同候选只能在最多两次FP16表示间隔内分类为近并列敏感性，明确差异或无法解释的首个分歧仍失败。该分类不会把非相同序列标为相同，不修改实际答案或QA/NLL分数，不改变质量阈值。同字节串行/流水线的KV及输出仍需完全一致，原始KV分块拼接也仍需逐元素相同。源码变更后使用新批次，旧观测不续拼。详见 `experiments/numerics.py`、`tests/test_numerics.py` 和每批 `cache_checks.json`。

进一步检查发现默认 FP16 投影矩阵的归约随输入行数变化，会在文本逐块 prefill 中累积误差。较大配置现在显式指定 `execution`：Transformer 投影固定 256 行计算、显式因果掩码、efficient SDPA、关闭 FP16 reduced-precision reduction。保留原 HF 注意力类、原权重、FP16 KV 和最后位置 LM head；无新增 CUDA 内核。整段和分块共同使用该执行设置。全部 15 个正式 TTFT 上下文的独立检查已达到逐元素相同，六个全续写 NLL 检查通过；完整矩阵验证另行记录。

KV 和 logits 仍按原逐元素 atol=0.1/rtol=0.02 检查，RMS 仅作诊断。该执行设置改变绝对耗时，完整记录在配置和运行指纹中，不能与旧默认执行批次合并计时。minimal/small 默认配置继续使用原执行设置；新消融和完整方法均共用当前批次的设置。
