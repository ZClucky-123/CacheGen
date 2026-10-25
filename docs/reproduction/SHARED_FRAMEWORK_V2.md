> 历史 v2 指南快照，用于解释旧批次。当前入口与配置已升级到 v3；核对 v2 实现请使用对应批次的 source/ 快照。

# TinyLlama / Mistral 公共实验框架

当前协议为 `cachegen-shared-v2`，三个核心实验组、六个执行任务。初始差异见 [审查表](SHARED_FRAMEWORK_AUDIT.md)，历史限制见 [历史索引](HISTORY.md)。所有命令从仓库根目录执行，不依赖机器绝对路径。

## 配置与入口

| 配置 | 用途 | 覆盖 |
| --- | --- | --- |
| `experiments/configs/minimal.json` | 功能与实际恢复检查 | 两长度、全部六质量方法、七种 TTFT 方法、四组自适应；一 token 尾块 |
| `experiments/configs/small.json` | 本轮小规模验收 | 2 源文档×2 长度、2 事实×3 位置×2 长度、4 完整 TQA、144 TTFT、288 自适应观测 |
| `experiments/configs/larger-mistral.json` | 增大同一 Mistral 的矩阵 | 20 源文档×3 长度、90 位置 QA、20 TQA、3150 TTFT、1800 自适应观测 |
| `experiments/configs/larger-tinyllama.json` | 增大 TinyLlama 矩阵 | 512/1024/1536 长度、相同方法和策略、7 完整 TQA |

最小/小规模配置通过 `--model tinyllama|mistral` 切换适配器；较大配置各自声明模型。参数控制数据文件、样本数、长度、生成长度、分块、方法、带宽、轨迹、倍率、SLO、预热、重复、种子与 CPU 线程。模型也可给出完整适配器字典，但新增架构必须验证位置和模型输出接口。权重模式固定为 TinyLlama FP16 和原 Mistral FastChat groupwise 8bit；KV 均为 FP16。模型权重量化与 KV 量化分别记录。

完整适配器字典通过 `bash experiments/run.sh --config custom.json` 使用；不要同时传用于整体选择默认适配器的 `--model`。应用级编解码分块由 adaptive.chunk_size 控制，作者 CUDA 算术编码器内部最多 256 token/段的实现边界保持不变。

```bash
# 只看配置与六个任务：不加载 GPU、不写结果
bash experiments/run.sh --model mistral --config experiments/configs/small.json --dry-run

# 前台最小检查；每次使用新的 run ID
bash experiments/run.sh --model tinyllama --config experiments/configs/minimal.json

# 注册入口与兼容入口均执行公共框架
python3 -m experiments run mistral -- --config experiments/configs/small.json
python3 -m experiments run tinyllama -- --config experiments/configs/small.json
bash mistral_repro/run.sh --config experiments/mistral/config.json
```

`python3 -m experiments run ... --dry-run` 预览调度命令；需要查看内部矩阵时直接调用 `experiments/run.sh --dry-run`。结果使用 `results/<模型>/<minimal|small|larger>/<唯一run-id>/`，不覆写已有目录。

## 脱离终端运行、查询与恢复

```bash
python3 -m experiments.background start --name mistral-small-new -- \
  --model mistral --config experiments/configs/small.json --run-id small-new

# 查看启动器状态、tmux 是否仍在；会话结束后 JSON 和日志继续保留
python3 -m experiments.background status --name mistral-small-new
tail -f .local/launches/mistral-small-new.log
python3 -m experiments status mistral

# 如需交互查看；Ctrl-b 然后 d 脱离，不会停止测量
tmux -L cachegen attach -t mistral-small-new

# 中断后用新的 tmux 会话名恢复原批次；模型/配置来自批次本身
python3 -m experiments.background start --name mistral-small-resume -- \
  --resume results/mistral/small/small-new

# 成功批次只核验完整归档
python3 -m experiments.suite --verify results/mistral/small/small-new
```

tmux 由服务器持有，关闭 SSH、IDE 或本地终端不会终止任务。服务器重启或实例销毁仍会停止进程；持久化结果目录后可按相同指纹恢复。对 manifest 的当前 runner PID 发送 TERM 可正常中断并保存 143；不能用旧 PID 停止另一个进程。GPU 锁竞争返回 75，不会启动第二个计时任务。请让同一 GPU 的所有入口共用 `CACHEGEN_LOCAL_DIR`。

每条正式观测使用稳定唯一 ID，SQLite WAL/FULL 提交；中断不丢失已提交的行。恢复前验证配置、全部测量辅助源码、数据、选择、权重/tokenizer、依赖/CUDA/驱动/GPU/线程与资源配额；不一致拒绝续拼并要求新批次。未完成的观测重新执行，已完成的观测跳过；恢复会重新预热，原运行和恢复尝试均可追溯。只有预期 ID 集合完整匹配后才生成成功报告。失败样本及异常在 `failures.jsonl`，运行状态在 `manifest.json`，终止码在 `exit_code`。

完整性检查同时要求离线 prefill、编码、传输、恢复、剩余输入和总计时等必要字段，以及逐 token NLL/生成输出；只有数量齐全仍会失败。离线成本在逐行结果中作为“该上下文准备一次”的注记重复出现，不能按网络计时重复次数累加。自适应还记录全档编码和准备 wall time，后者包含原始 QA、预热与校验，不代表最小生产预计算成本。

## 协议边界

QA 在运行前固定为作者的 `lstrip('\n').split('\n')[0]`，标准规范化 EM/F1、仅数据自带别名，EOS 或生成上限停止；完整原文与 token IDs 一并保存。真实 TQA 与受控位置 QA 分开。每长度门槛使用同长度的受控题；raw EM 正确数为零则保留率为 null、验收为 unassessable。原短语包含规则仅留作诊断，不作为放宽后的主分数。语义人工诊断可另建分析记录，不改标准分或历史主指标。

v2 NLL 评分全部续写 token：恢复完整 KV 后取 `KV[0:L-1]`，重放最后一个上下文 token 并 teacher force 后续 `N-1` 个输入，共得到 N 个目标分数。记录实际目标 token IDs 和逐 token NLL。v1 漏首目标的 NLL 不混入汇总。负 ΔNLL 本身不能解释为质量改善。

文本首 token 路径执行相同 transformer 和权重，只将最后一位置 hidden state 送入同一 lm_head。实测前对多个真实输入和长度，比较未改动完整 forward、优化 forward、raw KV 复用、多 token generate、重复初始化和分块拼接；有损恢复另检查形状/类型/有限值和质量。

TTFT 从建立本地 socket 之前开始，到首个生成 token 的 CPU ID 可用时结束。CUDA 同步一致，传输、恢复/分词、剩余输入推理和完整 wall time 分别记录。离线 prefill/编码单列；收完再恢复与分块串行分别命名，没有传输—解码重叠。带宽校准覆盖 64KiB/1MiB/8MiB（较大配置另含64MiB）、不限速及全部目标带宽；在线实测中记录实际吞吐，调试字节比对不计入传输区间。没有调整 SSH 主网卡。

自适应每块仅观察当前带宽，以当前带宽预测剩余已知块成本，并扣除已耗时和首 token 预留。保留剩余预算、原等分静态预算、固定 1/2/3 和文本六策略。默认实际传输、恢复/重算；`trace_estimate` 明确记录估算网络时间。后面的缓存块来自完整原上下文的预计算，而文本回退在当前已拼接的前缀 KV 上计算；记录块起止、实际缓存长度和回退。SLO 用于 TTFT，完整回答延迟另存。

## 较大规模：已提供配置，不自动启动

```bash
# 先审阅配置；正式运行会重新核对实际环境和数据
bash experiments/run.sh --config experiments/configs/larger-mistral.json --dry-run
bash experiments/run.sh --config experiments/configs/larger-mistral.json --preflight-only

# 由用户决定启动时间；以下命令不是本轮自动执行的任务
python3 -m experiments.background start --name mistral-larger-new -- \
  --config experiments/configs/larger-mistral.json --run-id larger-new

# TinyLlama 的对应入口
python3 -m experiments.background start --name tiny-larger-new -- \
  --config experiments/configs/larger-tinyllama.json --run-id larger-new
```

较大配置针对本机约 24GiB GPU、30GiB 主存配额、固定缓存模型与 CUDA 扩展。Mistral 保留 FastChat 8bit，最大自然文本 4096 token；资源预检估算权重、KV 及恢复临时空间，逐样本处理，不积累所有已解码 KV。建议保留至少 2GiB 额外磁盘给记录/快照和日志，实际预检根据可用空间拒绝不足环境。增加到 8K/16K 前应重新核对预检与峰值，不能由窗口上限直接推出显存够用。低带宽的大 payload 和重复矩阵会显著增加耗时，预计需数小时，具体取决于实际吞吐和策略回退。本配置不支持单张 4090 运行官方 70B 规模。

TinyLlama 只有 7 个 <=2000 token 的完整 TQA，较大规模主要增加自然文本和条件/计时矩阵；若要增加独立真实问题，需另备适配窗口的新数据并建立新批次。事实仍只有 10 个独立事实，不能通过取模重复计数。

两份较大配置已执行实际数据/窗口/资源/绘图预检，未加载权重或执行测量：见 [Mistral 证据](larger-mistral-preflight-20261024.json)与 [TinyLlama 证据](larger-tinyllama-preflight-20261024.json)。预检包含所有文本传输条件的精确 token 恢复；TinyLlama 有一个条件需要修补一个 token。

文本传输使用 `utf8_token_patch_v1`：发送 UTF-8 文本与显式 token 修补信息，接收端实际重分词后按收到的元数据修补。全部 JSON framing 与修补 token 都计入传输体积，逐次记录 patch 数量。这样可保持部分词 token 或拼接 token 段的原输入，不能把它描述成未经修补的纯文本网络基线。

绘图依赖固定在 `mini_repro/requirements-mini.txt`，运行前会验证当前解释器能实际输出图片，避免测量完成后才发现其他 Python 版本的二进制库不兼容。

归档保存配置、协议、模型/权重/tokenizer 指纹、Git HEAD 与补丁、完整公共源码和相关依赖源码、数据/选择、环境、逐例记录/生成/顺序、检查点、图表与 SHA-256 清单。区间按源文档/问题/事实整体重采样，跨长度共享来源保持关联；重复计时不增加独立样本，小样本 P95 仅作描述。
