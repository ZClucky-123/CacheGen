# 工作区、兼容性和后续实验

本机所有项目实体均位于 `/home/aaa/CacheGen`；上传 GitHub 时以该目录为仓库根目录。项目代码根据脚本位置寻找仓库，克隆到其他目录后不需要修改活动代码中的机器绝对路径。

```text
CacheGen/
├── experiments/                 标准库调度器、注册表与各组实验
│   └── mistral/                 Mistral 代码、默认配置和原始三项测试
├── mini_repro/                  原 TinyLlama API 与公共算法
├── mistral_repro -> experiments/mistral
├── results/                     按实验和批次保存原始测量
│   ├── mistral/                 本次成功与首次失败批次
│   └── reproduction/            已归档 TinyLlama 正式批次
├── reports/mistral/             中文报告、图表与数据
├── tools/server_setup/          相对路径的安装与环境校验工具
├── scripts/experiment-env.sh    共用环境变量与 GPU 锁
├── tests/                       全套回归与工作区兼容性测试
├── .local/                      本机依赖、字体、下载工具与迁移记录
├── .venv/                       本机 Python、模型与历史环境日志
└── .cuda-12.1/                  本机 CUDA 工具链
```

## 查看本次结果

- [完整中文报告](../../reports/mistral/20261024-mistral7b/小规模复现实验报告.md)
- [图表 PDF](../../reports/mistral/20261024-mistral7b/图表合集.pdf)
- [原始结果与失败诊断](../../results/mistral/README.md)

## 位置与环境覆盖

以下变量均可选，默认路径相对仓库根目录；相对路径覆盖也按仓库根目录解析。

| 变量 | 默认值或作用 |
| --- | --- |
| `CACHEGEN_REPO_ROOT` | 根据代码位置发现；一般无需设置 |
| `CACHEGEN_RESULTS_DIR` | `results`，每个实验再分子目录 |
| `CACHEGEN_REPORTS_DIR` | `reports` |
| `CACHEGEN_LOCAL_DIR` | `.local`；同一 GPU 的测量请共享此目录以共用锁 |
| `CACHEGEN_PYTHON` | 优先 `.venv/bin/python`，否则当前 Python |
| `CUDA_HOME` | 优先仓库 `.cuda-12.1`、`.venv/cuda-12.1`，再用 `/usr/local/cuda-12.1` |
| `HF_HOME` / `HF_HUB_CACHE` | 模型缓存；默认 `.venv/huggingface` |
| `CACHEGEN_MODEL_CHECKSUMS` | Mistral 权重校验清单；默认 `.venv/mistral-download-verified.json` |
| `TORCH_EXTENSIONS_DIR` | 默认 `.local/torch_extensions` |
| `TORCH_CUDA_ARCH_LIST` | 默认 `8.9`，换 GPU 时按实际架构设置 |
| `CACHEGEN_PLOT_PYTHON` | 后处理解释器，默认运行后处理脚本的 Python |

```bash
# 在另一结果位置复跑；配置文件内容会保存到新批次 source/config.json。
CACHEGEN_RESULTS_DIR=results/new-study \
  python3 -m experiments run mistral -- --config experiments/mistral/config.json
```

Mistral 默认固定模型和 revision。修改样本数、长度、重复次数等测量参数时使用单独配置，保留默认配置以供比较；切换模型架构应建立新的实验条目。TinyLlama 使用已有 `--profile` 接口。启动后结果目录不复用旧批次；GPU 测量冲突返回 75。可对 manifest 的 `runner_pid` 发送 TERM 停止新批次，监督入口会转发信号并保留 interrupted 状态和退出码 143；预检失败也有可见的 manifest。

## 后处理和报告

原始实验自动输出参数化的 `report.md`、JSON 和 CSV。需要图表时，后处理 Python 应已安装 `matplotlib` 与 `numpy`：

```bash
python3 experiments/mistral/finalize.py results/mistral/<批次>
```

已有完整 `artifacts.json` 时，后处理先核验归档并直接返回，保持结果和源码快照不变；校验失败另写 `artifact-verification-error.json`，不覆盖原校验清单。历史源码优先与批次快照核对；新批次额外记录相对仓库的指纹，用于迁移后追踪。

`build_document.py` 是 **2026-10-24 固定实验设计的中文交付报告生成器**，保留当时的统计口径和说明；后续改变实验设计时使用通用 `report.md` / `plot_results.py`，或为新设计单独扩展报告模板，不将旧模板的样本叙述直接用于新设计。本机现有绘图依赖在 `.local/mistral/plotdeps`，字体在 `.local/mistral/report-assets`，未作为模型或环境依赖上传。

## 兼容与归档

`mistral_repro` 是仓库内相对链接，旧 `bash mistral_repro/run.sh` 和 `python3 mistral_repro/status.py` 继续有效；`mini_repro` 导入和原有脚本参数保持兼容。Mistral 的旧 `runs`、`deliverables`、`latest` 链接指向新的结果和报告目录。

历史批次的绝对路径代表执行时现场，原始 JSON、报告包和源码快照保持字节不变。本机已有 `/root/cachegen-repro` 链接改指仓库内入口；不再需要 `/home/aaa` 下的兼容文件。

新结果默认留在本机。归档前核对完成状态、退出码、样本/方法/重复数量及 SHA-256，再为具体批次添加 `.gitignore` 白名单并更新结果索引。已验证的 Mistral 原始记录、诊断失败批次与交付包已列入本次上传；`.local`、Python 环境、模型、wheel、编译缓存及 CUDA 均被忽略。

目录与暂存区原始备份在 `.local/migration/20261024-workspace/`。迁移过程不删除唯一数据副本；详细记录见 [整理说明](WORKSPACE_ORGANIZATION_20261024.md)。

按用户要求，本地报告 ZIP 已删除，所有 `.zip` 默认忽略，上传使用展开后的报告、图表和原始数据。
