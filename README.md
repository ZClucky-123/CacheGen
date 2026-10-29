# CacheGen 公共复现实验框架

TinyLlama 与 Mistral 共用 `cachegen-shared-v3`：质量与体积、TTFT、带宽自适应、轻量机制消融四组实验，默认七个任务。配置、测量、恢复、评分、汇总和归档共用代码，模型差异集中在适配器。

目标是复现论文核心方法与结论。现有公共框架和历史结果作为基线保留；新增独立的 [核心复现路径](docs/reproduction/PAPER_COMPRESSION.md)，覆盖三个 insight、分集校准/扫描、压缩消融、TTFT及因果自适应。2026-10-29已在4090D编译CUDA扩展，30项单测通过；正式模型实验尚未执行，量化单位仍属披露的实现解释。详细边界见 [论文对照](docs/reproduction/PAPER_ALIGNMENT.md)。

## 文档入口

| 目的 | 文档 |
| --- | --- |
| 新压缩路径、参数不确定项及服务器验收命令 | [论文压缩规格](docs/reproduction/PAPER_COMPRESSION.md) |
| 配置、协议、后台启动、监控与恢复 | [公共框架指南](docs/reproduction/SHARED_FRAMEWORK.md) |
| 模块、目录、路径与扩展方式 | [工作区指南](docs/reproduction/WORKSPACE_GUIDE.md) |
| 大上下文修复、固定版本、规模和耗时 | [稳定版本记录](docs/reproduction/LARGER_READINESS_20261025.md) |
| 实现与论文的对应及差距 | [论文对照](docs/reproduction/PAPER_ALIGNMENT.md) |
| 已完成的实测、验收和原始记录 | [报告索引](reports/README.md) |
| v1/v2/v3 批次与评分边界 | [历史索引](docs/reproduction/HISTORY.md) |
| 依赖、CUDA 和固定模型准备 | [环境说明](docs/reproduction/MISTRAL_ENVIRONMENT.md) |

新核心研究分两阶段，服务器同步分支后运行 `bash tools/run_paper_core.sh stage1 .local/paper-core-run01`；完成压缩与绘图后，再独立运行 `bash tools/run_paper_core.sh stage2 .local/paper-core-run01` 做网络实验。stage1只绑定核心源码，stage2另记录网络源码指纹。详见上方核心复现文档。所有编译与测试仅在服务器执行。

## 项目结构

```text
CacheGen/
├── experiments/                 两模型公共框架、配置、调度及分析
│   ├── configs/                 minimal、small、larger 和预验证配置
│   ├── engine.py                模型、编解码、prefill、生成及 NLL
│   ├── ttft.py、chunks.py       独立分块及同字节三模式测量
│   ├── feedback.py              上一块实测吞吐反馈
│   ├── mechanisms.py           共用量化中间结果与熵编码对照
│   └── suite.py、records.py     精确矩阵、逐行检查点及严格恢复
├── mini_repro/                  共享工具和保留的 TinyLlama v1 入口
├── LMCache/、src/                作者编解码器、CUDA 和辅助代码
├── test_data/、tests/            固定数据及 CPU/GPU 回归
├── results/<model>/<scale>/<id>/ 原始观测、源码快照及 SHA256 清单
├── reports/                     中文验收与可分享报告
├── docs/reproduction/           当前指南、论文对照及历史记录
├── scripts/experiment-env.sh    共用环境和 GPU 锁
└── tools/server_setup/          环境安装、模型下载和校验
```

`mistral_repro` 是到 `experiments/mistral` 的相对兼容链接。`.venv/`、`.cuda-12.1/`、`.local/` 和本机 `releases/` 不上传；Git commit 与每批源码指纹标识具体版本。

## 运行

在仓库根目录执行。实际测量需准备固定模型与 CUDA 扩展；计划及状态命令只需 Python 标准库。

```bash
python3 -m experiments list
python3 -m experiments status mistral
bash experiments/run.sh --model mistral --config experiments/configs/minimal.json --dry-run

# 数据、窗口、资源和绘图预检；不加载权重
bash experiments/run.sh --model mistral --config experiments/configs/minimal.json --preflight-only

# 独立 tmux；每个新实验使用唯一会话名和 run ID
python3 -m experiments.background start --name mistral-v3-new -- \
  --model mistral --config experiments/configs/minimal.json --run-id v3-new
python3 -m experiments.background status --name mistral-v3-new
python3 -m experiments status mistral --run-dir results/mistral/minimal/v3-new
tail -f .local/launches/mistral-v3-new.log

# 中断后用新会话名，从原配置和检查点恢复
python3 -m experiments.background start --name mistral-v3-resume -- \
  --resume results/mistral/minimal/v3-new
```

切换模型使用 `--model tinyllama`。较大配置自身包含模型，直接选择 `--config experiments/configs/larger-mistral.json` 或 `larger-tinyllama.json`，不再叠加 `--model`。SSH/IDE 关闭不影响 tmux；服务器重启或进程环境重建后需从检查点恢复。

## 规模与证据

| 配置 | 质量 | TTFT | 自适应 | 消融 | 正式观测合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimal | 54 | 168 | 28 | 24 | 274 |
| small | 120 | 432 | 336 | 24 | 912 |
| larger-mistral | 1020 | 9450 | 2100 | 24 | 12594 |
| larger-tinyllama | 942 | 9450 | 2100 | 24 | 12516 |

预热另计，larger 另有 2590 次网络实验预热。Mistral 较大长度为 1024/2048/4096，TinyLlama 为 512/1024/1536。Mistral 使用 FastChat 8bit 权重，两模型 KV 均为 FP16。

两模型各有 274 条的 [v3 功能验收](reports/shared-v3-20261025/验收报告.md)。后续 Mistral 大上下文修复又完成 469 条完整预验证、15 个严格 KV、6 个 NLL 对照和真实恢复核验；公共回归为 112 项，另有 3 项兼容测试。不同协议、模型和执行配置分别保存，不拼接旧观测。

## 测试与归档

```bash
source scripts/experiment-env.sh
"$CACHEGEN_PYTHON" -m unittest discover -s tests -v
PYTHONPATH="$PWD/experiments/mistral:$PYTHONPATH" \
  "$CACHEGEN_PYTHON" -m unittest discover -s experiments/mistral -p test_experiment.py -v
"$CACHEGEN_PYTHON" -m experiments.suite --verify results/mistral/minimal/v3-final-03
```

`manifest.json` 保存任务和状态；`records.sqlite3`/`records.jsonl` 保存逐观测输出；`source/`、`metadata.json` 和 `artifacts.json` 保存源码、环境及文件哈希。配置、代码、数据、模型或环境变化时拒绝续拼。新结果默认留在本机，仅显式归档已核验批次。

作者原始脚本保留，见 [sigcomm_ae.md](sigcomm_ae.md)；它们与当前公共协议分别解释。[论文](https://doi.org/10.1145/3651890.3672274) · [原始 CacheGen 项目](https://github.com/UChi-JCL/CacheGen) · [LMCache](https://github.com/LMCache/LMCache)。
