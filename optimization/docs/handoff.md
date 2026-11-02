# CacheGen联合优化代码交接

## 版本与职责

开发分支为 `feat/cachegen-joint-optimization`，算法与原接口对照基线为 `d6b6626014c30b64011e4348c02510fdfbdb7204`，协议为 `cachegen-joint-v2`。接收版本使用以下命令记录，执行档案另保存源码、配置、数据和设备软件fingerprint：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

| 责任 | 交付内容 |
| --- | --- |
| 代码维护 | 联合实现、CPU参考与故障路径测试、CUDA源码/绑定、配置、运行脚本、缺陷修复及版本变更说明 |
| 实验执行 | 目标环境准备、构建、设备正确性、原入口回归、成本标定、残差拟合、独立验收、冻结档案与正式实验 |

代码交接版本与实验档案的 `frozen` 状态分别管理。`freeze`在独立验收ready后执行。

## 源码入口与实现范围

| 位置 | 职责 |
| --- | --- |
| `experiments/joint.py`、`joint_config.py` | CLI、固定参数、来源选择和实验阶段 |
| `experiments/joint_codec.py`、`joint_cuda.py` | 三档三编码、选中后生成、分离GPU工作与CPU封装 |
| `LMCache/third_party/cachegen_optim_cuda/` | 精确组合编号、CDF恢复、位打包和融合最终KV写入 |
| `experiments/joint_runtime.py`、`joint_types.py` | 统一纳秒时钟、任务身份、发送冻结、共享GPU执行器、取消及最终KV |
| `experiments/joint_source.py`、`joint_cache.py` | 原始KV源、租约、CPU LRU和同键任务合并 |
| `experiments/joint_policy.py`、`joint_scheduler.py` | 质量/路线选择、元数据事件模拟、GPU竞争和状态刷新 |
| `experiments/joint_profiles.py`、`joint_calibration.py` | 成本与大小、完成历史反馈、残差、覆盖率、质量及冻结门禁 |
| `experiments/joint_records.py` | 完整观测与检查点恢复、诊断/正式记录隔离、重复键校验 |
| `experiments/joint_wire.py`、`mini_repro/transport.py` | 动态长度分帧、单链路、接收容量和I/O空闲超时 |
| `tests/test_joint*.py` | 协议、控制、故障清理、CPU集成及目标GPU检查 |

每块最多登记9个质量/表示元数据候选；选择完成后仅生成所选包。三种编码为 `ac_count`、`ac_cdf`、`fixed_bits`，同档符号和恢复语义保持一致；升降档读取原始KV。TEXT从本次已提交前缀进行真实模型prefill。原codec、原复现入口与历史结果保留独立路径。

完整约束见[整体方案v2.2](CacheGen整体优化与代码修改方案.md)，方法与产物说明见[实现和实验指南](implementation-and-experiments.md)。

## 检查范围与目标环境待确认项

CPU检查覆盖参考协议与数值、选择器和穷举参考对照、模拟资源竞争、小张量与socket集成，以及取消、失败改选、旧结果和断连清理。张量集成依赖CPU PyTorch。GPU与真实模型用例在具备目标条件时执行。

目标环境固定为Linux、Python 3.10.12、PyTorch 2.3.1+cu121、CUDA Toolkit 12.1、Transformers 4.42.3、单张计算能力8.9 GPU，以及原FastChat8权重和torchac依赖。模型revision、环境准备链接及依赖列表见[实验指南](implementation-and-experiments.md)。

目标设备阶段需要取得以下证据：

- 三档、全部尾块长度、三种表示及 `0x10/0x11/0x12` 的CDF/符号/FP16位模式对照。
- 原codec与联合路径的真实模型首token对照，源与最终KV隔离、默认流及pinned槽生命周期。
- 真实TEXT/whole_text轨迹、质量变化、缓存冷热和表示未缓存的行为与质量。
- 实际服务时间、搬运量、GPU竞争、CPU控制开销、大小估计误差和内存占用。
- 独立来源上的NLL/EM、F_safe覆盖率及正式配对性能。

## 分阶段接收与执行

在服务器仓库根目录准备已有权重和环境，设置参数：

```bash
export CUDA_VISIBLE_DEVICES=0
export CACHEGEN_PYTHON="$PWD/.venv/bin/python"
export CUDA_HOME="$PWD/.cuda-12.1"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME="$PWD/.venv/huggingface"
```

各阶段分别触发，统一批次示例为 `joint-v2-r01`：

| 阶段命令 | 核验与输出 |
| --- | --- |
| `bash optimization/scripts/server.sh plan joint-v2-r01` | 无模型/GPU工作；核对来源划分、状态数和请求数 |
| `bash optimization/scripts/server.sh build joint-v2-r01` | 固定环境检查，原torchac按需构建，优化扩展构建；保留build日志 |
| `bash optimization/scripts/server.sh gpu-check joint-v2-r01` | 设备用例实际执行且零跳过，生成匹配fingerprint的正确性证据 |
| `bash optimization/scripts/server.sh calibrate joint-v2-r01` | 成本/大小和H/选择成本；每状态5来源、20实测样本，预热另计 |
| `bash optimization/scripts/server.sh fit-residual joint-v2-r01` | 独立残差来源，按请求内最大误差及最近秩P90拟合余量 |
| `bash optimization/scripts/server.sh verify joint-v2-r01` | 独立验收来源，质量、覆盖率和完整条件矩阵 |
| `bash optimization/scripts/server.sh freeze joint-v2-r01` | ready后冻结实际档案摘要和数据身份 |
| `bash optimization/scripts/server.sh benchmark joint-v2-r01` | 读取匹配的已冻结档案，正式测试来源上执行配对矩阵 |

正式全矩阵的GPU时间与费用应由短测估算后安排。CLI的 `calibrate --state-limit N` 产生诊断成本档案，其状态为 `diagnostic_costs`；完整标定完成后才进入残差与验收阶段。完整参数用 `python -m experiments.joint --help` 查看。

质量门槛为NLL增量≤0.02、EM保留率≥0.90；预测经验覆盖率门槛为0.90，各有效主键或回退键至少5来源、20请求样本。正式4K自动质量按静态准入使用q2/q3。成本、残差、验收和正式测试来源分别由[来源划分](../configs/source-split.json)隔离。

阶段通过条件未满足时保存对应批次日志与失败档案，定位该阶段后再继续。数值、设备或质量问题应附最小复现条件反馈给代码维护方。

## 输出、恢复与错误定位

正式产物位于基于仓库根目录解析的 `optimization/output/<run_id>`。manifest描述配方和阶段；活进程的GPU源和CPU包缓存由真实任务创建。进程重启后重新取得源与包，热缓存由该组实际先前请求形成。

正式benchmark记录位于 `benchmark/`，固定q诊断的记录和检查点位于独立的 `diagnostic-benchmark/`。各scope的 `records.jsonl` 保存完整观测，`checkpoint.json` 保存完成键，诊断完成标记独立于正式完成标记。

观测记录刷盘后保存检查点。相同版本、数据、配置和档案下重新执行同一命令，从完整记录恢复完成键并跳过完成条件。中断留下的无换行尾部先保留为 `records.jsonl.partial-*`，再清除未提交尾部；完整行损坏、同键内容冲突或检查点引用缺失观测时报告数据完整性错误。相同内容的重复记录在分析中只计一次。功能或执行环境改变时创建新批次。失败请求的实际事件、生成和网络工作记录在 `joint_failure` 中。

| 信息 | 处理入口 |
| --- | --- |
| GPU正确性证据缺失/过期 | 核对本批次build与gpu-check日志及fingerprint |
| `profile_fingerprint_mismatch`、冻结摘要不匹配 | 核对提交、数据、配置与软件设备版本，创建匹配的新批次 |
| `cost_model_missing`、`residual_profile_missing` | 核对标定完整性、样本量和分组回退，补相应阶段 |
| `runtime_profile_not_ready`、质量/覆盖率未通过 | 查看 `verification.json` 与verify阶段失败日志 |
| `source_unavailable` | 核对master生命周期、实际缓存命中和TEXT准入 |
| `transport_timeout` | 检查NET_START后的实际字节进展；I/O空闲超时独立于TTFT期限 |
| `device_error`、格式错误或断连 | 终止当前请求，保留日志；设备状态恢复后再开启新请求 |

分析入口为：

```bash
"$CACHEGEN_PYTHON" optimization/scripts/analyze.py --run-dir optimization/output/joint-v2-r01

# 固定q诊断结果单独分析
"$CACHEGEN_PYTHON" optimization/scripts/analyze.py --run-dir optimization/output/joint-v2-r01 --diagnostic-only
```

## 问题回传与后续版本

回传最小信息：

```text
代码提交号 / 分支：
run_id / 阶段 / 完整命令：
GPU型号 / 驱动 / Python / torch / CUDA / Transformers：
输入source_id / T / 块区间 / q / 编码 / 缓存状态：
预期行为 / 实际行为 / 首个错误：
复现步骤 / 相关失败用例：
manifest、environment、失败档案和日志的位置：
```

复现组的新统计结果可更新结果引用；量化、CDF、模型接口、执行配置或计时口径变化时，先核对与固定基线的差异，再决定同步范围。重要修改形成增量提交，并说明受影响模块、成本/残差/质量档案和重测范围。性能代码若由服务器短测触发，实施前记录短测证据、拟改范围、预计新增GPU时间与费用。

旧版本结果保留原批次，新版本使用独立记录。代码交接资料随源码保存，本机专用启动脚本、环境与运行记录由Git忽略规则隔离。
