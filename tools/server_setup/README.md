# 服务器环境安装脚本归档

这些文件原先散落在 `/home/aaa/`，2026-10-24 归拢到此处。活动脚本改为根据自身位置寻找仓库，并使用共享环境脚本；已去除旧机器绝对路径和固定进程号。重构前版本保存在本机 `.local/migration/20261024-workspace/setup-before/`。

| 文件 | 用途 |
| --- | --- |
| `cachegen-complete-environment.sh` | 补齐 PyTorch、CUDA wheel 与环境依赖 |
| `cachegen-finish-setup.sh` | 完成依赖安装、扩展安装和环境检查 |
| `cachegen-prefetch-cuda.sh` | 下载 CUDA 依赖 wheel |
| `cachegen-download-torch.py` | 下载 PyTorch wheel |
| `cachegen-verify-model.py` | 校验已有 Mistral 权重分片 |
| `cachegen-mistral-smoke.py` | Mistral 生成与真实 KV 编解码冒烟检查 |
| `.cachegen-original-constraints.txt` | 原环境版本约束 |
| `.cachegen-cuda-wheels.txt` | CUDA wheel 清单 |

本机环境已安装，本次整理未重新执行安装流程。环境位置与验收方法见 [MISTRAL_ENVIRONMENT.md](../../docs/reproduction/MISTRAL_ENVIRONMENT.md)，实际实验入口见 [mistral_repro/run.sh](../../mistral_repro/run.sh)。

旧下载工具已归入 `.local/download-tools/`；环境日志在 `.venv/logs/`，早期 `/tmp/cachegen-*.log` 是系统临时记录。
