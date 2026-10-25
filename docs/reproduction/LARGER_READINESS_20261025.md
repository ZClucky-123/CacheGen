# Mistral 大规模启动前修复与运行记录

本记录对应 2026-10-25 用户要求的“先修改好大规模复现可能的问题，再启动”。旧 v2/v3 验收数据保持原样；失败、验证和正式批次分别保存，不拼接观测。

## 启动前修复

1. **大上下文数值一致性**：默认 FP16 投影归约随行数变化，2048-token 自然续写出现一次近并列 token 分歧，文本分块在 1K/2K/4K 也会超出原逐元素门槛。新增配置控制的固定 256 行 Transformer 投影、显式因果掩码、efficient SDPA 和完整 FP16 归约；使用原 FastChat 8bit 权重、FP16 激活/KV、HF 注意力类，最后位置 LM head 保持原路径。整段和分块共用该执行配置。独立检查全部 15 个正式 TTFT 输入，文本 KV 15/15 逐元素相同；每个长度取两个源，共六个全续写 NLL 对照通过。原逐元素容差和质量阈值保留，RMS 只记录诊断；同字节串行/流水线仍要求完全一致。
2. **算术编码缓冲安全**：内核允许 256 token，但每通道输出仅 256 字节；极低概率符号可能超过 8 bit/token。保留内核，统一采用最多 96 token 的内部段，包含 CDF/终止开销的空间余量。应用层块仍为 256 token，内部实际段、内核上限和全部元数据分别记录。极端 16-bit CDF 概率及 4096-token 局部稀有符号 GPU 测试逐符号通过。该变化影响完整字节数，因此使用新批次。
3. **持久化与续跑**：JSON 检查点写入后 fsync 文件与父目录；SQLite 按行 WAL/FULL 提交。未完成 TTFT 案例保留原编码字节，续跑核验 SHA256。显存峰值跨尝试取最大值，保留进程主存峰值。
4. **长时间运行保护**：定期检查源码/配置/数据变化，任务边界全量核验；变化时停止，拒绝继续旧指纹。扩大大上下文检查临时张量和稳定执行的显存余量，检查主存、暂存磁盘与模型窗口。
5. **计时和汇总**：接收/恢复重叠计算移到首 token 计时之后；修正接收容量边界检查与配对的队列维度；状态命令识别新分层结果目录并报告 runner 是否存活。位打包在转换 uint8 之前拒绝越界值。

新增及修改源码、配置、测试自动纳入每批源码快照、指纹与 SHA256 清单。稳定执行会影响绝对耗时，旧默认执行批次不与本次混合。参考代码：`experiments/stable.py`、`numerics.py`、`artifacts.py`、`records.py`、`ttft.py`，以及 CacheGen Python 编码器。

## 验证批次

- 完整回归：矩阵测量源码对应 111 项通过，包含真实 GPU 测试；旧 Mistral 兼容测试另 3 项通过。随后修复实际注册表漏查 larger 状态的问题，新增回归后 **112 项＋3 项通过**；只改注册表和状态测试，测量逻辑未变。
- 最终矩阵验证：`results/mistral/small/v3-larger-validation-03`，tmux 初始会话 `mistral-v3-readiness-03`。质量 102、TTFT 315、自适应 28、消融 24，合计 **469 条正式验证观测**，另有 343 次网络实验预热。
- TTFT 使用全部 5 个正式源文档和 1024/2048/4096 三种长度、7 方法、3 模式；验证时带宽为 10Gbps。低带宽由自适应倍率 0.01 的条件和接收端预算测试覆盖。正式低带宽长时间矩阵在验证后运行。
- 已在首个 4K 案例提交 3 条 TTFT 观测后实际发送 TERM，以 143 退出；共保留 315 条观测和 14 份已编码表示，核验暂存 payload SHA256 后以同一指纹恢复。初始会话于北京时间 12:08 启动，续跑会话 `mistral-v3-readiness-03-resume` 于 12:25 启动。控制器记录 `.local/larger-readiness-recovery-before.json`，最终归档会保存验证结果。配置预热之外，中断恢复会重做该案例的预热和校验，事件日志单独保留。
- **最终通过**：469/469 观测，105 组串行/流水线配对，15 个严格且逐元素相同的文本 KV、6 个全续写 NLL 对照、224 次传输正确性检查（含恢复重查）。有 1 组 4K 配对跨两次尝试，编码字节/哈希及首 token 相同。190 个归档文件哈希核验通过。证据为该批 `validation/validation.json`。
- 在第二次尝试运行至 441 条时，容器进程环境重建，tmux/实验进程同时消失，未写退出码；cgroup OOM 计数为零。该次不标记为正常退出。北京时间 13:07 通过新会话 `mistral-v3-readiness-03-resume-2` 恢复，441 条旧行哈希保持不变。第三次于 13:16 完成，尝试退出码分别为 `[143, 未记录, 0]`。
- 实测峰值：GPU 分配 13.45 GiB、分配器保留 21.78 GiB、进程主存 5.73 GiB。配置预检及磁盘空间检查通过。

此前 `results/mistral/larger/v3-larger-20261025`、`v3-larger-20261025-r2` 和 small 下的 `v3-larger-validation-01/02` 均在正确性检查阶段停止，正式观测均为零。保留用于诊断，不能作为有效实验结果。

## 正式规模

Mistral-7B-v0.2 固定 revision，FastChat 8bit 权重，KV FP16，单张 RTX 4090 D。完整配置为 `experiments/configs/larger-mistral.json`。

| 实验 | 规模 | 正式观测 |
| --- | --- | ---: |
| 质量 | 自然文本 20 源×3 长度；受控 10 事实×3 位置×3 长度；TriviaQA 20 题；6 方法 | 1020 |
| TTFT | 5 源×3 长度×3 带宽×7 方法×3 模式×10 次 | 9450 |
| 自适应 | 5 事实×5 轨迹×7 策略×2 带宽倍率×2 SLO×3 次 | 2100 |
| 轻量消融 | 1 自然案例＋3 受控位置，3 精度×2 熵编码 | 24 |
| 合计 | 不含预热 | **12594** |

TTFT 带宽 0.1/1/10Gbps，应用块 256 token；各条件另预热 2 次。自适应长度 1024、块 256 token，各条件预热 1 次；合计另有 **2590 次网络实验预热**。消融每行内部另保存三次编解码计时与一次预热。重复、位置和长度条件不是新增独立源样本。

按本轮实际完整报文和计时更新：TTFT 纯传输预算 **10.30 小时**，含本地开销投影约 11.67 小时；自适应执行约 6.50 小时；质量约 0.51 小时，另计准备、消融、检查与归档，中心估计 **19.3 小时，建议预留 22–24 小时**。原先 14–18 小时的估计低估了低带宽自适应成本，现已被这次实测投影替代。只用一个验证事实/轨迹外推五事实/五轨迹仍有不确定性，实际选档和服务器负载可能改变时间。

用户最后明确范围：**不补做 anchor/delta 和论文调度器，只修复现有框架并交付可运行版本**。固定版本与论文的对应及差距见 [论文对照](PAPER_ALIGNMENT.md)，不能以本轮可执行验收声称论文全部机制或收益已复现。

## 运行与监控

固定服务器源码位于 `/home/aaa/CacheGen/releases/cachegen-shared-v3-stable-20261025`。其 `SOURCE_SHA256SUMS` 固定源码、数据、配置和操作入口；`VERSION.json` 保存指纹与实测来源。该目录为本机副本，不重复上传 Git；仓库提交与归档 `source/` 提供可移植源码。

固定副本直接完成 112 项测试；TinyLlama 真实模型在 512/1024/1536 下的 15 个文本 KV 逐元素相同、6 个 NLL 对照通过。初次打包遗漏的 `mistral_repro` 相对链接已依据失败的兼容测试补齐，重跑全套通过。数据、模型和 CUDA 的真实较大 Mistral 预检也已通过。

固定副本另完成 TinyLlama `minimal/v3-stable-acceptance-20261025`：274/274 条、7/7 任务、退出 0，逐行哈希、矩阵和当前仓库源码一致性核验通过。该批最小验收不代替完整较大 TinyLlama 实验。

所有后台任务使用独立 tmux socket `cachegen`。关闭 SSH、终端或 IDE 后服务器继续运行；服务器重启后需从检查点恢复，使用新的会话名并保留原批次源码环境。

```bash
cd /home/aaa/CacheGen/releases/cachegen-shared-v3-stable-20261025
sha256sum --quiet -c SOURCE_SHA256SUMS
bash run-server.sh --config experiments/configs/larger-mistral.json --dry-run
bash background-server.sh start --name mistral-v3-larger-stable-20261025 -- \
  --config experiments/configs/larger-mistral.json --run-id v3-larger-stable-20261025
bash background-server.sh status --name mistral-v3-larger-stable-20261025
bash status-server.sh mistral --run-dir /home/aaa/CacheGen/results/mistral/larger/v3-larger-stable-20261025
tail -f /home/aaa/CacheGen/.local/launches/mistral-v3-larger-stable-20261025.log
# 仅中断后恢复，换会话名
bash background-server.sh start --name mistral-v3-larger-stable-resume-1 -- \
  --resume /home/aaa/CacheGen/results/mistral/larger/v3-larger-stable-20261025
```

本机副本共用父项目的 `.venv`、模型缓存、CUDA 和 `.local/gpu.lock`，不包含模型权重镜像。其他服务器使用仓库入口，按环境说明重建依赖并下载相同 revision。
