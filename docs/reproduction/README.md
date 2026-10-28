# 复现文档导航

| 顺序 | 文档 | 用途 |
| --- | --- | --- |
| 当前补全 | [核心方法复现](PAPER_COMPRESSION.md) | 三个 insight、固定数据划分、压缩曲线与消融、TTFT/自适应、服务器命令及耗时 |
| 1 | [公共框架指南](SHARED_FRAMEWORK.md) | 当前 v3 配置、协议、执行、监控与恢复 |
| 2 | [工作区指南](WORKSPACE_GUIDE.md) | 模块、路径、环境和归档 |
| 3 | [论文对照](PAPER_ALIGNMENT.md) | 已实现、近似及未实现机制 |
| 4 | [稳定版本与大规模验证](LARGER_READINESS_20261025.md) | 修复、实测证据、固定副本和耗时 |
| 5 | [报告索引](../../reports/README.md) | 验收和原始记录 |
| 6 | [历史索引](HISTORY.md) | 协议/评分及历史批次边界 |

环境见 [Mistral 环境说明](MISTRAL_ENVIRONMENT.md)及 [TinyLlama 环境与历史入口](../../mini_repro/README.md)。`MISTRAL_REPRODUCTION.md` 记录作者原始脚本；`SHARED_FRAMEWORK_V2.md`、旧整理记录和计划供历史核查。原公共框架默认 v3，四组实验默认七个任务；新压缩路径通过独立命令进入，不能将旧批次视为新方法的验证结果。
