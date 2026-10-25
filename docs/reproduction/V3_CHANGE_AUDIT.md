# v3 修改与回归对应表

当前操作说明见 [共享框架](SHARED_FRAMEWORK.md)。v2 历史原始结果及归档源码没有修改。v3 批次使用自己的协议、配置、矩阵及检查点。

| 要求 | 代码位置 | 验证证据 |
| --- | --- | --- |
| 应用块独立编码，完整元数据与尾块 | `experiments/chunks.py` | 帧长度/截断测试；真实 257-token 上下文；每块 token_start/end、shape、dtype、quantization_schedule、内部≤256分段 |
| 同一编码结果重复用于串行/流水线 | `experiments/ttft.py` | prepared_payload_sha256；`pending-prepared` 原字节持久化；配对汇总强制校验 |
| 一连接、CPU 收发、主线程 GPU | `mini_repro/transport.py` / `experiments/ttft.py` | 主线程 ID 测试；串行完成后发许可；容量为1的阻塞及恢复异常线程退出 |
| 发送前等待字节预算、真实可用时刻 | `mini_repro/transport.py` | `test_receiver_availability_waits_for_full_budget_including_short_tail` 直接检查 data_available，未只检查总耗时 |
| CUDA 完成后的恢复时间与端到端 TTFT | `experiments/ttft.py` | 回调完成 CUDA 同步，首 token 为 CPU int；全量 receive/restore/queue/send/overlap 时间线 |
| 文本实际重分词、依赖前缀 prefill | `experiments/engine.py` / `experiments/chunks.py` | 显式 token 修补计字节；真实模型文本分块 vs 整段 KV 数值/多步生成检查 |
| 原始 KV 与有损表示一致性 | `experiments/ttft.py` / `tests/test_v3_mechanisms.py` | raw 分块拼接完全相等；同字节两模式恢复完全相等、生成相同 |
| 历史吞吐信息隔离、固定首块初始化 | `experiments/feedback.py` | 不接收当前/未来轨迹；仅 CompletedTransfer 三字段；首块固定值、后续来源必须为 i−1 |
| 同规则的两种动态策略 | `mini_repro/adaptive.py` / `experiments/tasks.py` | 相同 next_decision；同一带宽输入时档位/候选预测/预算逐项相等；只执行串行网络 |
| 算术编码开关 | `experiments/mechanisms.py` / `experiments/ablation.py` | 同次量化对象，全部符号/尺度 SHA；4/5位实际打包；全部符号及反量化完全相等 |
| 统一/分层精度 | `experiments/mechanisms.py` | 同一量化器及 FP32 bins，与原编码器零值/小幅值符号一致；统一16/32及分层2展示预算差异；CUDA内核未扩展 |
| 调度未知类型直接报错 | `experiments/config.py` / `experiments/records.py` / `experiments/suite.py` | expected_keys 的未知 kind 测试；调度枚举检查，消融有独立 kind |
| 模式/块大小/估计器/变体独立编号 | `experiments/records.py` | 稳定 ID 唯一性测试；精确集合校验；新设置变动引起配置指纹变化 |
| 指纹、恢复与归档 | `experiments/artifacts.py` / `experiments/records.py` | 全源码自动快照，SQLite行SHA，配置/代码/数据/模型/环境变化拒绝；实际TERM和tmux恢复记录 |
| 汇总区分新增维度 | `experiments/analysis.py` / `experiments/analysis_v3.py` | TTFT各模式独立，逐对速度比，独立消融表及熵配对表，CSV全量时间线 |

运行测试：

```bash
bash -c 'source scripts/experiment-env.sh; "$CACHEGEN_PYTHON" -m unittest discover -s tests -v'
bash -c 'source scripts/experiment-env.sh; "$CACHEGEN_PYTHON" -m unittest discover -s experiments/mistral -p test_experiment.py -v'
```

实际测量使用新批次和 tmux。轻量功能矩阵不用于证明论文调度器、WAN 性能或统计显著的流水线收益。`uniform16`/`uniform32` 指16/32-bin档位，实际按4/5位符号保存；现有 `uniform4`/`uniform8` 基线保持独立。
