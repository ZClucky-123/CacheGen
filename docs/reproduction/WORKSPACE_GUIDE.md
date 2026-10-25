# 工作区与公共框架

当前协议为 `cachegen-shared-v3`。配置、实验设计和执行命令以 [公共框架指南](SHARED_FRAMEWORK.md)为准；本页说明模块、目录和扩展。历史整理记录见 [2026-10-24 记录](WORKSPACE_ORGANIZATION_20261024.md)。

## 目录职责

本机仓库根目录为 `/home/aaa/CacheGen`。代码按脚本位置发现根目录，迁移后需准备目标机器的模型和工具链。

- `experiments/` 是两模型正式框架，包含配置/适配器、Engine、数据评分、TTFT/反馈/消融、检查点、分析和归档。
- `mini_repro/` 提供复用工具并保留 TinyLlama v1；其历史轨迹估算口径不代表 v3。
- `LMCache/`、`src/`、根目录 `run_*.py` 保留作者代码和原始入口。
- `test_data/` 保存固定数据；`tests/` 保存 CPU/GPU 和兼容回归。
- `results/<model>/<scale>/<run-id>/` 保存批次，`reports/` 保存中文验收及可分享报告。
- `docs/reproduction/` 保存指南、环境、论文对照和历史索引。
- `mistral_repro -> experiments/mistral` 为相对兼容链接。

本机 `.venv/`、`.cuda-12.1/`、`.local/` 和 `releases/` 不上传。固定服务器副本重用父项目环境、模型和 GPU 锁，见 [稳定版本记录](LARGER_READINESS_20261025.md)；新机器不能依赖这些本地目录。

## 路径和环境覆盖

相对路径覆盖均相对仓库根目录解析。

| 变量 | 默认值或作用 |
| --- | --- |
| `CACHEGEN_REPO_ROOT` | 代码所在仓库，通常无需设置 |
| `CACHEGEN_RESULTS_DIR` | `results`，再按模型/规模/批次分层 |
| `CACHEGEN_REPORTS_DIR` | `reports` |
| `CACHEGEN_LOCAL_DIR` | `.local`；同 GPU 的所有入口必须共用锁目录 |
| `CACHEGEN_PYTHON` | 优先 `.venv/bin/python`，否则当前 Python |
| `CUDA_HOME` | 优先 `.cuda-12.1`、`.venv/cuda-12.1`，最后 `/usr/local/cuda-12.1` |
| `HF_HOME` / `HF_HUB_CACHE` | 默认 `.venv/huggingface` |
| `TORCH_EXTENSIONS_DIR` | 默认 `.local/torch_extensions` |
| `TORCH_CUDA_ARCH_LIST` | 本机 `8.9`，迁移时按目标 GPU 设置 |
| `OMP_NUM_THREADS` | 默认 8，实际线程数记入指纹 |

```bash
CACHEGEN_RESULTS_DIR=results/new-study \
  bash experiments/run.sh --model mistral --config experiments/configs/minimal.json --run-id new-study
```

使用 `extends` 扩展配置；新增模型差异放入适配器，复用公共流程。运行中修改源码、配置或数据会被 InputGuard 检出并停止，恢复也拒绝不同指纹。

## 后台、检查点和状态

`experiments.background` 使用独立 tmux socket `cachegen`，SSH/IDE 关闭后继续运行。GPU 锁冲突返回 75。服务器重启或进程环境重建后，用新会话名从原批次 `--resume`；tmux 本身不能跨重启存活。

SQLite WAL/FULL 按观测提交，JSON 检查点同步写入。TTFT 未完成案例的原编码字节保存在 `pending-prepared/`，恢复核验哈希，完成后删除。不要手工删除未完成批次暂存。

默认状态覆盖该模型全部规模，精确查询使用 `--run-dir`。同时查看 `runner_alive`、后台 `tmux_alive` 和日志：manifest 的 running 是最后记录状态，强制终止后可能残留。对确认属于当前批次的 runner PID 发送 TERM，可保留检查点并返回 143。

## 分析和归档

任务完成后自动生成逐观测 JSONL、分组 CSV、配对/时间线、PNG/SVG/PDF 和 report，再核验精确矩阵并封存哈希。

```bash
source scripts/experiment-env.sh
"$CACHEGEN_PYTHON" -m experiments.suite --verify results/mistral/minimal/v3-final-03
```

历史 `build_document.py`、`plot_results.py`、`finalize.py` 保留旧设计口径。历史报告、原始记录、源码快照及清单保持原字节；后续修复建立新批次，不用当前源码补写旧证据。

新结果默认不提交。确认用途、完成状态、退出码、精确观测集合和哈希后，再为特定目录添加 Git 白名单并更新索引。环境、模型、wheel、编译缓存、运行中批次和 ZIP 不加入提交。
