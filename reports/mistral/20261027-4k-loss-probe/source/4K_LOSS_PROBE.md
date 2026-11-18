# 4K 压缩损失的最小复验

目的：区分粗量化损失与编码解码问题，并判断提高 K 或 V 精度分别有何影响。使用原批次 ID 3、5、16 三个高损失来源和 ID 7 低损失对照；固定原来的 4096-token 上下文和 32 个正确后续 token。它是有目的选择的根因诊断，不替代完整质量实验。

## 本地检查：无需 GPU 或模型

在仓库根目录运行：

```bash
python3 -m tools.diagnose_4k_loss --dry-run
```

此命令核验原归档的 config、selection、metadata、records SHA256，检查评分目标一致，显示计划；不导入 torch、不建输出目录、不加载模型、不访问网络。需要本地已有原始 `records.jsonl`；从 Git 克隆后，先按原批次 README 恢复分卷文件。

```bash
.venv/bin/python -m unittest discover -s tests -p test_4k_loss_probe.py -v
```

CPU 测试覆盖样本/目标配对、归档损坏、K/V 单因素配置、禁止覆盖原结果、单个符号差异和非有限数检测。另有 CUDA 专用测试，真实检查 257/4096-token 合成 KV 的五种配置、符号无损恢复与重建一致性；无 CUDA 时明确跳过。该合成测试不等于真实 Mistral 复验。

## 准备服务器时

先在平台启动已有实验镜像并获得新的 SSH 连接信息。确认原项目、`.venv`、Mistral 权重及原始结果仍在。将本次新增的 `tools/diagnose_4k_loss.py`、`tools/run_4k_loss_probe.sh` 放到同一仓库；若运行测试，也同步 `tests/test_4k_loss_probe.py`。不需要上传本地 Python 环境，也不需要重复下载模型。

脚本会检查原有执行源码、关键依赖版本及依赖源码、全部固定模型文件、CUDA 扩展二进制 SHA256；使用原配置中的 FastChat8 权重、稳定执行设置、线程数和 TF32 设置。检查不通过时保留错误并停止，应恢复原环境或明确分析环境变化，不绕过检查继续声称相同设置。

## 服务器执行

在服务器的仓库根目录运行；不与其他 GPU 实验同时执行：

```bash
bash tools/run_4k_loss_probe.sh --dry-run
bash tools/run_4k_loss_probe.sh
```

入口会使用共用环境和 GPU 锁，创建唯一的 `.local/diagnostics/4k-loss-<UTC时间>/`。开始加载模型前会逐项校验约 14.5GB 固定模型文件，耗时取决于磁盘速度；不要把校验阶段误认为卡住。没有 CUDA 时直接报错，不生成模拟结果。真实运行耗时和显存尚未在 GPU 上验证，不给出未经测量的完成时间保证。

若只想先验证一个高损失来源的完整路径：

```bash
bash tools/run_4k_loss_probe.sh --sources 3
```

每次运行使用新目录，不会续拼或覆盖原实验；可用 `--output .local/diagnostics/自选新目录` 指定不存在的输出目录。关闭终端可能中断前台运行，建议保持当前 SSH 连接直到完成；中断时已完成记录会落盘，但应在新目录重跑，不与新尝试混合。

## 实际对照与输出

每篇文档先比较完整前向 NLL、原始 KV 复用 NLL，以及原批次 raw NLL；平均误差超过原检查的 0.01 容差就停止。随后只 prefill 一份原始 KV，并复用于下列五组量化配置：

| 配置 | K 精度 | V 精度 |
| --- | --- | --- |
| cachegen1 | 原 CacheGen1 | 原 CacheGen1 |
| cachegen2 | 原 CacheGen2 | 原 CacheGen2 |
| cachegen3 | 全 32 bins | 全 32 bins |
| cachegen1_k32 | 全 32 bins | 保持原 CacheGen1 |
| cachegen1_v32 | 保持原 CacheGen1 | 全 32 bins |

每组先用原始量化/反量化函数直接恢复 KV，然后执行真实序列化、CUDA 算术编码、解码。对比全部量化符号、尺度和最终 FP16 KV，要求逐元素一致；再对两条恢复路径分别评分，要求相同 KV 得到相同逐 token NLL。任何不一致都会停止，并在 `status.json` 标为 failed，不能拿部分结果继续归因于量化。

原编码器要求 K、V 的 CDF 最后一维相同；单独提高 K 或 V 精度会使这一维不同。因此仅这两个诊断对照使用共同的 CDF 符号范围，仍保持原量化符号、实际 bins、尺度与反量化不变，复用原 CUDA 编码/解码内核及容器。三档原配置继续使用原序列化器。输出明确记录 `encoder_path` 和 `encode_timing_scope`；这两类编码时间范围不同，不能用于性能加速比较。这个诊断适配仍需在服务器验证，不声称已经通过 CUDA 验证。

默认四篇共有 **44 条方法/路径评分**（每篇 raw 加五组×两路径），另有 **4 次完整前向评分对照**。不会跑 TTFT、自适应、问答生成或完整 12594 条矩阵。

| 文件 | 用途 |
| --- | --- |
| `plan.json`、`runtime.json` | 固定 token、配置、原始指纹、运行硬件和诊断脚本哈希 |
| `status.json` | running/completed/failed、完成评分数量、错误信息 |
| `raw_checks.jsonl` | 完整前向、原始 KV 复用及旧 raw 的 NLL 对照 |
| `codec_checks.jsonl` | 每组符号/尺度/KV 一致性、内部段长和编码字节数 |
| `scores.jsonl` | 每个路径的完整 token NLL、目标 token ID 和相对 raw 的 ΔNLL |
| `summary.csv` | 成功完成后生成的 NLL 比较表 |
| `head_errors.jsonl`、`head_errors.csv` | 每层、每 head、K/V 分开的 MAE/RMSE/最大误差 |

JSONL 按评分/变体刷新落盘，失败时仍可带回本地分析。真实运行后应把整个新目录备份回本地，再关闭服务器；看到 `COMPLETED` 且 `status.json` 为 completed 才代表本轮完成。

## 如何判断原因

- 符号、尺度或重建 KV 不一致：优先排查编码/解码或布局问题，不能直接归因为量化。
- 编码两路径完全一致，而两条路径都比 raw 差：本轮差异发生在量化或后续推理敏感性，编码没有额外改变这些样本的重建结果。
- 仅提高 K 或 V 精度改善：为该分量提供线索；若只有两者同时提高才改善，需要考虑交互，仍不能直接定位某一层。
- 原始 KV 与完整前向或旧 raw 明显不一致：先解决基线复现/环境问题，再解释档位差异。

相关分析：[逐文档、逐 token 损失分布](../reports/mistral/20261026-larger-analysis/loss_distribution/损失分布与原因分析.md)。脚本只增加诊断入口，不修改原量化配置、实验调度、评分门槛和归档数据。
