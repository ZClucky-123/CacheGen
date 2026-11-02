# 联合实现与实验指南

接收版本、职责、阶段通过条件和问题回传入口见[代码交接说明](handoff.md)。

## 方法与接口

注册只持有原始 FP16 KV、token IDs、源修订和配方。三档均准入时，每块最多 9 个质量/表示元数据候选；4096 tokens 排除 q1 后为 6 个。未选候选不读取张量、不量化、不编码、不搬运、不分配包正文。同键生成任务合并，缓存保存实际生成的不可变 CPU wire；每上下文容量为 256 MiB。

`ac_count` 由同档符号生成原整数 CDF 和 colex 组合编号，再调用原算术编码；tag 为 `0x12`，固定零失败在同任务内转全 B 项编号 `0x11`。`ac_cdf` 携带 tag `0x10` 与原 16 位 CDF，跳过编号。`fixed_bits` 直接打包选定符号，跳过 CDF 和算术编码。三者在同档保持符号、最大值和 FP16 恢复结果逐位一致。升降档重新读取原始 KV。

CDF 归一化、组合编号和位流定义见[低带宽方案第 5～6 节](低带宽CDF组合编码优化方案.md)与[高带宽方案第 5～6 节](高带宽GPU位打包与融合恢复优化方案.md)。M45F v2 为 64B 头，schedule 为 1/2/3；v1 保留 q2 兼容。位打包完整字节由几何精确计算：

\[
S=64+4LN+\frac{ND}{8}\sum_{p=0}^{2L-1} b_p.
\]

算术包缺失时使用独立成本来源拟合的大小 P50/P90，不在线生成全部候选以获取精确大小。已有包使用实际 `len(wire)`，生成与封装成本为 0，真实缓存访问仍计费。

`JointCodec.encode_gpu` 返回只含 CPU 字段的暂存结果；`seal` 执行 CPU 封装并释放源侧 pinned 槽。`Runtime` 统一持有源、LRU、默认流 GPU 执行器与动态 socket；`register_context`、`run`、`close_context` 管理上下文。`run` 返回实际时钟、决策、生成、网络、事件与共同最终 KV。最终 KV 在请求计时内分配，各恢复任务直接写对应 token 区间；TEXT 从真实已提交前缀执行 `Engine.prefill`。

所有绝对时刻使用 `perf_counter_ns()`。期限为 `t0_ns+ceil(slo_s×10^9)`；吞吐使用 bit/s，网络服务为 `ceil(8S×10^9/beta_bps)` ns。GPU 编码与恢复共用默认流并竞争；空闲点优先恢复，进行中的编码不抢占。单链路、接收容量 1、生成前瞻 1 的事件模型分别预测 F50 与 F_base：

\[
F_{safe}=F_{base}+\lceil\delta_g10^9\rceil.
\]

最高可行质量内按 F50 排序。余量按请求内最大残差、跨请求最近秩 P90 拟合；每个有效主键或回退键至少 5 个来源、20 个请求。阶段 P90 组成的 F_base 与整请求覆盖率分别记录。

## 数据与阶段

固定[来源划分](../configs/source-split.json)保留 LongChat 原计时来源 0～4 作为正式测试；成本为 5～9，残差为 10～14，独立验收为 15～19。四组互斥。清单读取现有 50 个来源，分词后检查每个来源能覆盖 T 与后续 32 tokens。

成本标定按 q、表示、N、prefix、前一 GPU 任务、缓冲状态、串行/流水线和实际 tag 分组。每状态预热 5 次，再每来源 4 次，合计 20 次实测；P50 取中位数，P90 取第 18 个样本。G 包含生成及所需 D2H；C 包含 CPU 封装与缓存插入；R 包含解析、H2D、恢复及最终写入；另外采集真实访问、选择和 H(T,suffix) 成本。TEXT 的 prefill 每次实际分配，缓冲字段记录 `allocates_each_restore`。

`fit-residual` 仅从残差来源收集已提交选择的预测和真实结束时间。`verify` 使用冻结的成本、大小与余量，覆盖两种执行模式、T=1024/2048/4096、分块/whole_text、冷/热/混合与稳定/变化带宽。单包 whole_text 的缓存状态为冷或热。实际联合轨迹分别进行自然续写 NLL 和原框架 controlled-fact EM 检验；EM 按对应执行条件汇总并要求保留率 ≥0.9，NLL 增量 ≤0.02。缺组、样本不足、覆盖率 <0.9 或正常请求意外失败均使验收未通过。

完整状态网格和正式对照矩阵规模由 `inspect --plan` 给出。该网格覆盖所有声明的资源状态，实际 GPU 时间需用短测服务时间估算；`--state-limit N` 只产生 `diagnostic_costs`，不能进入正式验收。正式测试不用于拟合或反复调整余量。

正式矩阵另设 `benchmark_cache_conditions`。`quality_change` 先实际执行另一已准入固定 q 的单路线请求，再从原始 KV 执行目标 q；`representation_miss` 先执行同 q、目标表示集合之外的一种编码，再执行目标路线。两者的实际先前生成、发送、恢复与驻留在 `seed_request` 中保留，`seed_and_target_ttft_s` 记录两次 TTFT 之和。三表示全部可选的联合组没有集合外表示，故其表示未缓存条件不适用；没有固定质量目标的自动组、单包文本及仅有旧格式的原 codec 同样跳过不适用条件。原 codec 固定 q 的质量变化对照实际生成和发送另一 q，目标准备成本仍从当前计时起点计入。预检计数与这些适用条件一致。

组合数只读表在启用 `ac_count` 的进程启动阶段显式预加载，其纳秒耗时和设备驻留单列为 `startup.jsonl`。成本采集在每个服务区间前从实际 workspace 元数据核对 first/reuse/expand；不可构造的状态明确拒绝，避免把首次分配标成扩容。

## 目标服务器依赖

- Linux、Python 3.10.12，暴露一张 RTX4090D/计算能力 8.9 GPU；GPU 锁由公共环境脚本提供。
- PyTorch 2.3.1+cu121、CUDA Toolkit 12.1、支持 CUDA 12.1 的编译工具链、Ninja、NumPy、Transformers 4.42.3。
- 现有 `lmcache`、`torchac`、`nvtx`、`torchac_cuda` 与 FastChat 8bit 权重加载依赖；保留原 codec 构建。build 阶段先检查 `import torchac_cuda`，缺失时使用原 setup 在本批次 build 目录构建，再构建优化扩展；可用依赖跳过重复构建。
- 已缓存的 Mistral-7B-v0.2 revision `80cf49b18de6354fd8a6d1e81c177dd830e05eea`，离线加载。环境与权重准备见[已有环境说明](../../docs/reproduction/MISTRAL_ENVIRONMENT.md)和[安装工具](../../tools/server_setup/README.md)。
- `test_data/longchat.jsonl` 的至少 20 个独立来源，以及固定原始配置与数据。`HF_HOME` 指向已有权重缓存。

## 执行命令

在已同步 `feat/cachegen-joint-optimization` 的服务器仓库根目录执行。批次名示例为 `joint-v2-r01`，每个新版本使用独立批次。以下为分阶段命令，正式实验独立触发。

```bash
export CUDA_VISIBLE_DEVICES=0
export CACHEGEN_PYTHON="$PWD/.venv/bin/python"
export CUDA_HOME="$PWD/.cuda-12.1"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME="$PWD/.venv/huggingface"

bash optimization/scripts/server.sh plan joint-v2-r01
bash optimization/scripts/server.sh build joint-v2-r01
bash optimization/scripts/server.sh gpu-check joint-v2-r01
bash optimization/scripts/server.sh calibrate joint-v2-r01
bash optimization/scripts/server.sh fit-residual joint-v2-r01
bash optimization/scripts/server.sh verify joint-v2-r01
bash optimization/scripts/server.sh freeze joint-v2-r01

# 验收并冻结后的独立正式阶段
bash optimization/scripts/server.sh benchmark joint-v2-r01
"$CACHEGEN_PYTHON" optimization/scripts/analyze.py --run-dir optimization/output/joint-v2-r01
```

CLI 等价入口为：

```bash
python -m experiments.joint --base-config experiments/configs/larger-mistral.json \
  --config optimization/configs/joint-v2.json --task inspect \
  --run-dir optimization/output/joint-v2-r01
```

固定质量诊断可在已标定批次使用 `--task benchmark --fixed-quality 2 --diagnostic-only --group joint-three-q2`。正常自动 benchmark 要求实际验收 ready 和 frozen。代码、配置、模型、执行设置或设备软件 fingerprint 改变后，建立新批次并重测受影响的成本、余量与准入。

诊断记录和检查点写入 `diagnostic-benchmark/`，正式记录写入 `benchmark/`。恢复时从完整观测回收完成键，保留并清理中断产生的残缺尾部；内容冲突的重复键或损坏完整行使恢复停止。诊断结果的分析命令另加 `--diagnostic-only`，完成键和统计均按scope隔离。

## 产物与解释

| 产物 | 内容 |
| --- | --- |
| `manifest.json`、`environment.json` | 阶段状态、版本、设备软件、配置 fingerprint |
| `source-split.json`、`*-context_manifest.json` | 互斥来源与配方；不代表 wire 已存在 |
| `cost-observations.jsonl`、`cost_rows.json` | 逐状态实际样本和 P50/P90 |
| `first_rows.json`、`selector-costs.json` | H(T,suffix) 与真实选择开销 |
| `runtime_profile.json`、`residual-profile.json` | 预加载成本、质量与端到端余量 |
| `fit-residual/`、`verify/` | 独立请求、失败与实际 QA 轨迹 |
| `verification.json`、`frozen.json` | 质量、覆盖率、缺组、验收与冻结证据 |
| `benchmark/records.jsonl`、`checkpoint.json` | 正式独立配对观测和完成键 |
| `diagnostic-benchmark/` | 固定q诊断的独立观测、检查点和分析产物 |
| `benchmark/summary.csv`、`report.md` | 由实际观测派生的指标与技术报告 |
| `logs/`、`build/`、`tmp/` | 本批次日志、独立扩展和临时构建产物 |

内存包默认不落盘，GPU master 保留至显式释放或关闭上下文。原始源显式释放后，仅匹配实际缓存包与已准入文本路线可用。进程间重新运行不继承活内存缓存；热缓存由该组此前实际生成的发送路线形成。

目标 GPU 短测暴露性能代码修改需求时，先记录短测证据、拟改范围、失效档案、重测范围及预计新增 GPU 时间/费用。实测、协议精确字节和条件推算分别表述。
