# CacheGen 联合优化

联合优化采用 `cachegen-joint-v2`。每块先比较质量与表示，再生成选中的包；三档对应 `ac_count`、`ac_cdf`、`fixed_bits`，并保留真实 TEXT/whole_text 重算。目标是在质量门槛和 TTFT 期限内选择最高可行质量，再选择完整预计完成时间最短的路线。

复现实验与历史结果入口为[仓库 README](../README.md)和[复现文档](../docs/reproduction/README.md)。原 `experiments` 入口、旧 codec、`results` 与 `reports` 延续复现职责。优化配置、运行脚本、技术文档与产物集中在本目录。

```text
optimization/
├── README.md
├── docs/                 设计、接口、实验与技术说明
├── configs/              联合参数与固定来源划分
├── scripts/              服务器分阶段执行、结果分析
└── output/<run_id>/      每个独立批次的全部新产物
experiments/joint_*.py    类型、源、缓存、codec、运行时、选择、预测与标定
experiments/joint.py      联合 CLI
LMCache/third_party/cachegen_optim_cuda/  独立 CUDA 扩展
tests/test_joint*.py      CPU 参考、状态、集成及目标 GPU 检查
```

技术入口：[整体方案 v2.2](docs/CacheGen整体优化与代码修改方案.md)、[CDF 组合编码](docs/低带宽CDF组合编码优化方案.md)、[GPU 位打包与融合恢复](docs/高带宽GPU位打包与融合恢复优化方案.md)、[实现与实验指南](docs/implementation-and-experiments.md)。联合准备、缓存、计时与调度规则以整体方案为准。

首轮执行顺序、重点指标与结果驱动的优化方向见[首轮实验与结果分析指导](docs/experiment-guide.md)。

代码接收、职责边界、设备待验证项、阶段通过条件及问题回传格式见[代码交接说明](docs/handoff.md)。交接提交与独立验收后的档案冻结分别记录。

在仓库根目录执行功能测试与无设备预检：

```bash
python -m unittest discover -s tests -p 'test_joint*.py' -v
python -m experiments.joint --task inspect --plan --run-id joint-v2-r01
```

服务器依次执行 build → gpu-check → calibrate → fit-residual → verify → freeze；独立验收通过并冻结后另行执行 benchmark。完整依赖和命令见[服务器执行指南](docs/implementation-and-experiments.md)。脚本各阶段均需显式指定，不自动租用设备或启动下一阶段。

输出路径从仓库根目录解析。默认生成 UTC 时间 run_id；也可显式设置 `--run-id` 或 `--run-dir optimization/output/<run_id>`。新日志、环境、来源、成本、残差、准入、检查点、指标和报告均在对应批次内。进程内 GPU 源与 CPU 包缓存由 manifest 描述；新进程重新获取源和实际执行的包，不把清单视为已有缓存。

功能测试包含标准库 CPU 参考、虚拟执行器、小规模 socket、张量集成及目标 GPU 用例，按依赖条件执行。目标服务器固定 PyTorch 2.3.1+cu121、CUDA 12.1、Transformers 4.42.3。CUDA 构建、目标设备逐位正确性、真实混合轨迹质量、成本、预测覆盖率及性能仍待目标 GPU 确认。
