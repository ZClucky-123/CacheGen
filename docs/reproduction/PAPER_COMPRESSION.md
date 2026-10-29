# 核心方法复现：压缩、传输与验收

范围是以一张 24GB 4090D、Mistral-7B FP16 验证核心方法与结论，不追求论文完整模型、数据或并发规模。顺序为 §5.1 三个 insight → 压缩曲线与消融 → TTFT → 因果自适应。**2026-10-29 已在服务器完成新CUDA扩展编译和30项单测；没有在本地运行测试，也没有加载Mistral或运行正式实验。**

旧 `experiments/`、`mini_repro/`、LMCache 实现及历史结果保留。新网络路径通过适配器复用旧 Engine、分块、socket 和 TTFT replay。不能把旧 FastChat8 权重结果拼入此次 FP16 对照。

## 方法与实现约定

- [论文 §5.1](https://arxiv.org/html/2310.07240v6#S5.SS1)：token 局部性、层损失敏感性、按层/通道分组的分布差异，共三个 insight。
- [§5.2](https://arxiv.org/html/2310.07240v6#S5.SS2)：每 10 token 共享首 token 锚点；锚点 8bit；分层量化差分；按 K/V、层、通道分别离线建模锚点和差分，并复用 CDF。
- [附录 C.2](https://arxiv.org/html/2310.07240v6#A3.SS2)：三层组、bin 参数 0.5/1/1.5；尺度单位未充分定义，不能把项目选择的解释标为作者确认参数。
- [附录 C.1](https://arxiv.org/html/2310.07240v6#A3.SS1)：重算优先和剩余时间预算。本项目额外披露首块前探测、恢复成本、安全余量，属于工程约定。
- [torchac 整数 CDF](https://github.com/fab-jul/torchac#normalized-vs-unnormalized--floating-point-vs-integer-cdfs)：16bit 概率，末端 65536 以 int16 零表示，未见符号也保留正概率。

锚点按每 K/V 层、每 token 的通道最大绝对值量化至 −127..127。差分按同粒度归一化至 `vector_range`，再按 bin 宽度取整。新层边界 10/21/32；旧 CacheGen 基线沿用 10/20/32 和原有 K/V 档位。

`original` 参考原锚点；`reconstructed` 参考按传输 scales 恢复的 FP32 锚点，解码使用同一预测值，最终转 FP16。差分正常范围 −127..127，超出者使用符号 255 与 int32 旁路，全部计字节。range=127、bin=0.5 可产生 ±254，不能遗漏旁路开销。

7.5 只是候选，不等于已确认的作者参数。逐 token 归一化也可能抵消差分幅度下降，不能仅凭原始差分方差小推断压缩成功。新配置仍记录 `interpretation_pending_verification`。

## 数据与候选

正式配置为 [`core-study.json`](../../paper_compression/configs/core-study.json)，协议 `paper-core-study-v2`。

| 用途 | LongChat ID | 操作 |
| --- | --- | --- |
| 校准 | 30–49 | 拟合各候选自己的 CDF，不参与选参 |
| 验证 | 20–29 | 三个 insight、扫描、冻结配置 |
| 测试 | 0–19 | 冻结后报告，不重新选参 |

校准使用 4K；验证/测试使用 1K、2K、4K，每个长度再取随后 32 个真实 token 计算 NLL。原始 FP16 KV 是质量参照。ID、token、数据、权重、执行设置、源码、profile 均留指纹；长度不足报错，不补齐或替换来源。

该划分约束本次选参流程；0–19 已在历史旧路径实验中使用过，不能将其描述为此前从未查看的全新盲测集。

扫描50个候选，每个独立校准。range 固定为 `6,8,10,12,15,20,25,31,50,127`，采用用户给出的10点网格（不是严格等比数列）：

| 类别 | 候选 | 数量 |
| --- | --- | --- |
| 核心方法 | 10个range × original/reconstructed；分层宽度 0.5/1/1.5 | 20 |
| 无差分 | 10个range，非锚点直接量化原始 KV，保留锚点位置与8bit编码；独立拟合 CDF | 10 |
| 无分层 | 所有层宽度1，10个range × original/reconstructed；独立拟合 CDF | 20 |

统一宽度固定1，相对精度由 range 扫描，避免重复等效缩放。验证保留全部曲线，报告匹配 range 的对照及各类折中。

每类选一个最终配置：**每个长度平均 ΔNLL≤0.02**，可行项中取三个长度合计估算字节最小者，平手按名称排序。无分层限定使用与所选核心方法相同的锚点参考。0.02 是项目筛选门槛，不是论文标准。无可行项就停止、保留记录，不看测试后放宽。`selected.json` 记录完整候选、所选参数、profile/源码指纹，绑定验证记录 SHA256；后续重新计算选择结果，拒绝手改或串用。

## 三个 insight

局部性与层敏感性只评估验证来源；概率分组严格分开校准与验证：

1. 局部性：报告原始 KV、间距1/9的两 token 差分、10-token组锚点差分的逐 K/V 层方差、平均绝对值、抽样分位数。按层比较，不用跨层汇总掩盖反例。
2. 层敏感性：每次只舍入浅/中/深一个层组的 K/V，使用同一归一化扰动，记录 ΔNLL、归一化 MSE、原始/扰动后的首话题问答 EM/F1；不预设结果。
3. 概率分组：固定统一量化器（range=7.5、各层宽度1），只在ID30–49拟合不分组/按K/V层/按层和通道/按token位置的四种CDF，在ID20–29计算交叉熵，不用验证集重新拟合。四种方式比较相同符号，使用同样的16bit CDF精度和平滑；记录每种CDF的存储字节。`grouping-cdfs.pt` 保存概率表、校准来源/token/模型/数据指纹，`grouping.json` 报告独立验证交叉熵。它是理论符号码长，不冒充完整算术编码体积。

## 扫描与最终压缩

扫描在 GPU 上用校准产生的实际整数 CDF 算 `Σ−log₂p`。完整估算包括符号码长/8、FP32 scales、int32 escape、每通道/内部段的int32长度、每流2字节终止/对齐预算、固定65536字节容器元数据预算。元数据预算在测试前固定，不宣称等于真实 `torch.save` 开销。

扫描不调用算术编码。uniform8 分母使用真实打包字节。profile 不随包传输，另列存储及按样本数分摊后的比值。

`final` 只评估三个冻结配置和0–19，重新测 uniform8、uniform4、旧CacheGen1/2/3；模型、输入和执行设置相同。全部方法报告逐来源NLL；4K再报告LongChat首话题EM/F1、原始生成及token。这是截短上下文的缩小规模任务，不等于论文完整上下文原结果。

事先指定 **ID0/7/16 ×三长度×三配置=27次**真实算术编码。要求符号、scales、重建KV、逐token NLL与仅量化一致；报告完整实际字节、流字节、估算误差字节/有符号相对误差/平均绝对和最大绝对误差、packet SHA256。全测试集估算与真实编码子集分开标注。

后端有torchac CPU参考；CUDA独立JIT名字/命名空间支持257个CDF边界、block≤32、内部段≤96，不覆盖旧扩展。性能阶段只接受CUDA，不能拿CPU计时代替。

## TTFT 与自适应

`network.CodecEngine` 将新codec接入已有分块、socket和TTFT replay。真实编解码和传输计时；预计算单独记录，profile预置两端。

- TTFT：ID0/7/16、4K；新核心配置/uniform8/uniform4/文本；0.5/1/3/10 Gbps；whole、512-token串行、512-token流水线；每组合一次预热、三次计时，432条正式记录。先完成全部预热，固定种子随机顺序；同一bytes的两分块模式恢复必须一致。报告实际吞吐，目标10Gbps不代表本机一定能达到。
- 自适应：同样三来源、4K、512-token块；两条轨迹 `[0.1,0.5,1,3,10,3,1,0.1]` 和 `[10,3,1,0.1,0.5,1,3,10]` Gbps。预注册SLO为 **0.5秒和1秒**，分别报告，不看结果再调。历史/固定验证所选档/固定最高质量档/文本四策略，三次重复，共144条。这是单机socket逐块控制，不是WAN或按墙钟变化的带宽回放。
- 固定最高质量档定义为所选锚点参考下最大的range=127（最高名义精度），与固定验证所选档分开命名。历史策略的10档质量排序仍来自验证NLL，可能不严格随range单调；不通过测试答案选择最高档。成本文件保存两个固定档的具体名称。
- 首块前1MiB探测，全部策略均计耗时/流量；首块用探测吞吐，之后只用已完成块。策略函数不接收当前/未来轨迹；决策后传输层才读取当前轨迹值。
- 预算扣除已花时间、实测首token保留和 `max(50ms, 剩余时间×15%)`；预测包含所有剩余块传输和恢复。重算可行优先文本，否则按验证质量排序选可行KV；都不可行则选预测最快者并标记。报告实际SLO违约，不保证紧SLO可达；汇总以策略、轨迹和SLO分别分组。
- 首token/恢复成本离线预测三次取最大值，是预置成本假设。该策略含额外保守余量，不声称逐行复刻附录伪代码。

问答生成和NLL核对均在TTFT计时结束后或预备阶段进行。TTFT质量按来源/方法/块尺寸计算一次后用于计时重复；自适应按实际混合KV评估。汇总区分独立来源数与重复次数，不把重复计时当作新增问答样本。

## 服务器命令

从服务器仓库根目录执行，先同步代码，沿用已有离线模型/依赖/CUDA。本地不执行下列命令。

在服务器上同步分支（先确认工作区没有需要保留的未提交修改；不使用reset/clean覆盖云端文件）：

```bash
git status --short
git fetch origin mini-repro-tinyllama
git switch mini-repro-tinyllama
git merge --ff-only origin/mini-repro-tinyllama
git log -1 --oneline
```

阶段一到最终压缩与绘图为止，保存日志、失败即停止：

```bash
bash tools/run_paper_core.sh stage1 .local/paper-core-run01
```

阶段二独立启动，复用同一批冻结配置和profile，不重跑阶段一：

```bash
bash tools/run_paper_core.sh stage2 .local/paper-core-run01
```

stage1根目录须不存在；成功后写 `stage1-complete.json`（v2），绑定全部第一阶段产物和影响其结果的源码。`paper_compression.provenance` 显式区分两阶段：stage1保留量化/CDF/编解码、insight、共用模型/评分/数据辅助、执行设置、研究入口及配置；排除 `network.py`、`policy.py`、网络传输模块、测试和启动脚本。仅修复这些网络文件不会改变校准index、冻结选择或stage1封存指纹；共享编解码、模型、研究入口或配置变化仍需重新验证stage1。

stage2先检查完整状态、图、stage1源码及产物哈希，拒绝缺失、失败或已更改的stage1；每个网络阶段另写 `stage2_source_hashes.json`，记录其网络代码与共享依赖，并写 `stage1_source_hashes.json` 标识所复用的核心版本。运行期间修改stage2代码会让该网络阶段失败，但不会改写或作废已封存的stage1。测试文件自身不作为数值结果依赖。旧v1封存不自动改写为v2。

stage2目录也不能已存在，二者不自动连续执行。需要单独执行各子步骤时：

```bash
source scripts/experiment-env.sh
(
  cachegen_lock_gpu
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_paper_provenance.py -v
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_paper_compression.py -v
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_paper_study.py -v
)
bash tools/run_paper_study.sh insights --output .local/paper-core/insights
bash tools/run_paper_study.sh calibrate --output .local/paper-core/profiles
bash tools/run_paper_study.sh scan --profiles .local/paper-core/profiles --output .local/paper-core/validation
bash tools/run_paper_study.sh plot --input .local/paper-core/validation --output .local/paper-core/validation-plots
bash tools/run_paper_study.sh final --profiles .local/paper-core/profiles \
  --selection .local/paper-core/validation/selected.json --output .local/paper-core/test
bash tools/run_paper_study.sh plot --input .local/paper-core/test --output .local/paper-core/test-plots
bash tools/run_paper_study.sh ttft --profiles .local/paper-core/profiles \
  --selection .local/paper-core/validation/selected.json --output .local/paper-core/ttft
bash tools/run_paper_study.sh adaptive --profiles .local/paper-core/profiles \
  --selection .local/paper-core/validation/selected.json --output .local/paper-core/adaptive
```

`bash tools/run_paper_study.sh plan` 在服务器展示固定矩阵。第一版 `run_paper_compression.sh fit/evaluate` 与两个C2配置仅留开发参考，未执行本次验证选参协议，不用于正式结论。

## 未实测耗时估算

前提为单张4090D 24GB、已有权重/依赖、无其他GPU任务。FP16路径和新CUDA后端尚未验证；不包括排错、失败重跑。

| 阶段 | 工作量 | 预计 |
| --- | --- | --- |
| 单测/首次编译 | CPU与CUDA，不加载Mistral | 5–15分钟 |
| 三个insight | 20校准＋10验证来源，40次NLL/最多40次问答 | 10–25分钟 |
| 校准 | 50候选×20来源，模型加载一次 | 60–180分钟 |
| 验证扫描 | 30来源/长度组合×50候选，无算术编码 | 30–90分钟 |
| 最终压缩 | 60组合×3配置与5基线、27编码、4K问答 | 15–45分钟 |
| TTFT | 432正式＋144预热 | 10–35分钟 |
| 自适应 | 144真实socket实验、10档成本预测 | 15–45分钟 |

阶段一按 **约2–6小时**、阶段二 **约0.5–1.5小时**预留（总计约2.5–7.5小时）。这不是实测承诺。首次运行后用校准 `timings.jsonl`、index的`seconds`、评估的`method_seconds`、各阶段`status.json.wall_seconds`修正估计。校准逐候选重新prefill，不落盘约10GB原始KV，不同时保留50份GPU计数；预留约6–8GB额外磁盘给profile、图、日志和扩展缓存。

输出包括三个insight记录、profile/index、验证/测试records与summary、冻结selected、PNG/PDF曲线及真实/估算字节散点图、网络时序与决策/SLO违约。源码/数据/profile/验证记录变更阻止串用；中断写failed、不自动认定完成。**当前已验证编译及合成输入单测，尚无论文模型实验结论的实测证据。**

## 2026-10-29 服务器编译与回归

4090D、PyTorch2.3.1+cu121、CUDA Toolkit12.1、架构8.9。在服务器临时源码副本运行，不修改已有诊断文件；编译缓存直接存入项目 `.local/torch_extensions/paper_ac256_a9bde53b6f099ec6/`，该目录应随镜像保留。全部检查约64秒：

- `test_paper_provenance.py`：3/3；网络文件变化不改变stage1，量化变化会改变，产物篡改仍被拒绝。
- `test_paper_compression.py`：13/13；含完整字母表、低概率符号、4096/257尾段、CPU/CUDA量化符号与恢复KV一致性。
- `test_paper_study.py`：14/14；含GPU/CPU校准一致性、交叉熵、固定策略、预算以及新codec的串行/流水线适配器检查。

服务器日志为 `.local/cuda-build-check-20261029/tests.log`，退出码0。没有执行stage1、stage2、Mistral推理或正式TTFT计时；这30项测试不能替代模型级质量和性能验收。后续从镜像恢复仍需验证驱动兼容和扩展加载。
