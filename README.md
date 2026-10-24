# CacheGen：KV 缓存压缩与小规模复现

本仓库基于 CacheGen（SIGCOMM 2024）的作者代码，保留原始实验脚本，并提供 **Mistral-7B** 和 **TinyLlama** 两组小规模复现实验。可以在这里阅读编解码实现、查看已归档结果，或使用统一入口开展后续实验。

**当前实验工作区与结果位于 [`mini-repro-tinyllama` 分支](https://github.com/ZClucky-123/CacheGen/tree/mini-repro-tinyllama)。** 初次了解项目，先看下面的结果入口、目录结构和 AI 阅读指南。

## 1. 先看已有结果

| 实验 | 设备与规模 | 阅读入口 |
| --- | --- | --- |
| Mistral-7B，2026-10-24 | RTX 4090 D；30 个自然文本条件、10 个完整问答、540 次正式 TTFT 计时 | [中文完整报告](reports/mistral/20261024-mistral7b/小规模复现实验报告.md) · [图表 PDF](reports/mistral/20261024-mistral7b/图表合集.pdf) · [原始批次](results/mistral/README.md) |
| TinyLlama，2026-10-13 | RTX 4070 Laptop；质量/体积、TTFT、带宽自适应，6 个任务完成 | [批次说明](results/reproduction/20261013T060232923276Z/README.md) · [自动报告](results/reproduction/20261013T060232923276Z/report.md) |

Mistral 本轮 CacheGen 3 的体积压缩比约为 **4.41–4.91 倍**，相对 uniform8 的 TTFT 速度比约为 **2.16–2.49**；相对文本重算未见中位 TTFT 优势，问答联合质量门槛未获得支持。具体分组、评分口径和适用范围以报告为准。

这些实验使用单卡和简化传输实现，覆盖作者仓库实现的小规模测量；完整执行与质量门槛是否得到支持，应分别根据记录判断。不同模型、设备和问答设计的结果分别保存。

全部批次见 [结果索引](results/reproduction/README.md)。报告、PDF、图表和数据以展开文件保存，ZIP 已删除并被 Git 忽略。

## 2. 目录结构

```text
CacheGen/
├── experiments/                   统一调度、路径管理与各组实验
│   ├── __main__.py                python -m experiments 的入口
│   ├── registry.json              实验名称、执行命令、结果子目录
│   ├── workspace.py               仓库路径、位置覆盖、状态读取
│   ├── process.py                 子进程监督与中断转发
│   ├── lifecycle.py               批次初始化、失败和中断状态
│   └── mistral/                   Mistral 配置、测量、后处理和测试
├── mini_repro/                    TinyLlama 实验与复用的评估工具
├── LMCache/                      作者 KV 编解码实现及 CUDA 扩展
├── src/                          模型加载、缓存接口等辅助代码
├── test_data/                    LongChat、TriviaQA、带宽轨迹等数据
├── results/                      逐批次原始记录与源码快照
│   ├── mistral/                  Mistral 成功批次及首次失败诊断
│   └── reproduction/             TinyLlama 正式批次
├── reports/mistral/              中文报告、图表、逐题问答及引用数据
├── docs/reproduction/            环境、运行方式、目录与归档说明
├── docs/superpowers/             设计与实施计划记录
├── scripts/experiment-env.sh     共用环境变量与 GPU 测量锁
├── tools/server_setup/           环境安装、下载与校验工具
├── tests/                        算法、计时、工作区与兼容性测试
├── main.py、cachegen.py           原始项目入口
├── run_*.py                      原始 CacheGen、量化、文本等实验脚本
└── mistral_repro -> experiments/mistral
```

`.venv/`、`.cuda-12.1/` 和 `.local/` 保存本机 Python 环境、模型、工具链及缓存，受 Git 忽略规则管理。克隆仓库后按环境说明准备这些依赖。`mistral_repro` 是兼容旧命令的相对链接，修改 Mistral 代码时使用 `experiments/mistral/`。

## 3. 快速浏览与预演

```bash
git clone --branch mini-repro-tinyllama https://github.com/ZClucky-123/CacheGen.git
cd CacheGen

# 以下命令只需 Python 标准库，不加载模型或使用 GPU。
python3 -m experiments list
python3 -m experiments status mistral
python3 -m experiments status tinyllama
python3 -m experiments run mistral --dry-run
python3 -m experiments run tinyllama --dry-run
```

`--dry-run` 显示将要执行的命令和路径，不创建实验批次。`status` 读取已有归档；尚无批次时返回 `not_started`。

实际测量使用 Linux/WSL2、Python 3.10，以及与实验配置匹配的 PyTorch、CUDA 扩展和固定版本模型。准备方式分别见 [Mistral 环境说明](docs/reproduction/MISTRAL_ENVIRONMENT.md)和 [TinyLlama 运行说明](mini_repro/README.md)。历史环境验证与实际实验设备可能不同，具体批次的设备和版本见其 `metadata.json`。

环境准备完成后，在仓库根目录运行：

```bash
# 使用默认 Mistral 配置创建一个新批次。
python3 -m experiments run mistral

# 使用独立配置；参数会交给对应实验入口。
python3 -m experiments run mistral -- --config experiments/mistral/config.json
python3 -m experiments run tinyllama -- --profile mini_repro/profiles/formal4070.json
```

新批次保存到对应 `results/` 子目录。Mistral 会保存配置和源码快照；共享环境脚本通过 GPU 锁避免同一工作区同时进行多轮测量。路径覆盖、中断和后处理见 [工作区指南](docs/reproduction/WORKSPACE_GUIDE.md)。

## 4. 如何用 AI 快速了解这个仓库

将当前分支的仓库交给能读取代码的 AI，或提供下表中的文件。先让它建立目录与调用关系，再围绕一个实验阅读实现，可以减少重复阅读历史快照和大体积数据。

| 顺序 | 让 AI 阅读 | 应当回答的问题 |
| --- | --- | --- |
| 1 | 本 README、[工作区指南](docs/reproduction/WORKSPACE_GUIDE.md) | 项目做什么，代码、结果、报告和本机依赖分别在哪里？ |
| 2 | [registry.json](experiments/registry.json)、[统一入口](experiments/__main__.py) | 有哪些实验，一条命令实际调用什么？ |
| 3 | [Mistral 配置](experiments/mistral/config.json)、[实验实现](experiments/mistral/experiment.py) | 模型、数据、方法、评分和计时如何组织？ |
| 4 | [模型加载](src/utils.py)、[编解码目录](LMCache/lmcache/storage_backend/serde/)、[公共实验工具](mini_repro/experiment_common.py) | 模型、KV 缓存、压缩恢复和评估如何连接？ |
| 5 | 一个批次的 `manifest.json`、`metadata.json` 和报告 | 实际跑了什么，测量是否完整，结论有哪些边界？ |
| 6 | [测试目录](tests/)、[Mistral 测试](experiments/mistral/test_experiment.py) | 哪些行为已有验证，修改后应该检查什么？ |

可以直接复制下面的提示词：

```text
请基于这个仓库的实际文件，帮我建立项目地图。先只读分析。

1. 阅读 README.md、docs/reproduction/WORKSPACE_GUIDE.md、
   experiments/registry.json 和 experiments/__main__.py，说明目录职责和实验入口。
2. 选择 Mistral 实验，阅读其 config.json、run.sh、measure.sh、experiment.py，
   追踪“命令入口 → 模型加载 → KV 编解码 → 质量/TTFT 评估 → 结果保存”。
   每一步标注文件路径和关键函数；需要时再阅读 src/、mini_repro/ 和 LMCache/。
3. 阅读 results/mistral/20261024T064543Z/ 下的 manifest.json、metadata.json、report.md，
   对照 reports/mistral/20261024-mistral7b/小规模复现实验报告.md，
   区分默认配置、真实执行记录、描述性指标和已获支持的结论。
4. 给我一份目录职责表、一张调用关系图，以及接下来最值得阅读的 5 个文件。
   如果我要调整样本数、修改压缩方法或新增实验，分别指出修改入口和相关测试。

不要运行安装、下载、模型加载或 GPU 实验，也不要修改文件。
跳过 .git/、.venv/、.cuda-12.1/、.local/ 和 __pycache__/；
历史 source/ 快照、逐例 JSON 和 CSV 只在核对具体问题时按需阅读。
所有结论给出文件依据，缺失信息明确标为未知。
```

若重点是 TinyLlama，将提示词中的 Mistral 配置、实现和结果入口替换为 `mini_repro/README.md`、`mini_repro/EXPERIMENT_PROTOCOL.md`、对应 `profiles/` 文件及 `results/reproduction/20261013T060232923276Z/`，再从注册表中的 TinyLlama 命令追踪实现。分析当前行为时从活动代码开始，核对历史版本时再按批次归档内容查找源码证据。

## 5. 修改与扩展实验

| 目的 | 修改或阅读入口 |
| --- | --- |
| 调整 Mistral 测量参数 | [experiments/mistral/config.json](experiments/mistral/config.json)；复制为独立配置后传入 `--config` |
| 调整 TinyLlama 实验组合 | [mini_repro/profiles/](mini_repro/profiles/)；通过 `--profile` 选择 |
| 修改编解码或量化方法 | [LMCache 编解码实现](LMCache/lmcache/storage_backend/serde/)和各实验的缓存恢复、评估调用 |
| 新增一组模型或实验 | 在 `experiments/<name>/` 实现入口，再登记到 [registry.json](experiments/registry.json)；步骤见 [注册说明](experiments/README.md) |
| 修改路径、环境或运行管理 | [workspace.py](experiments/workspace.py)、[process.py](experiments/process.py)、[lifecycle.py](experiments/lifecycle.py)、[experiment-env.sh](scripts/experiment-env.sh) |
| 使用原始论文实验脚本 | [sigcomm_ae.md](sigcomm_ae.md)、[Mistral 原始脚本操作说明](docs/reproduction/MISTRAL_REPRODUCTION.md) |

在已准备好的实验环境中运行回归检查：

```bash
source scripts/experiment-env.sh
"$CACHEGEN_PYTHON" -m unittest discover -s tests -v
"$CACHEGEN_PYTHON" -m unittest discover -s experiments/mistral -p 'test_*.py' -v
```

其中包含 GPU 编解码测试，需要可用的实验环境。新增结果默认保留在本机，核对完成状态、样本/方法/重复次数和文件校验后，再按归档规则加入 Git。历史批次的原始数据、源码快照和报告用于追溯，保留执行时的内容。

## 6. 如何核对实验记录

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | 本次参数、阶段、进度和完成状态 |
| `metadata.json` | 实际模型、设备、依赖、源码与数据指纹 |
| `source/` | 该批次保存的代码与配置版本；以实际归档内容为准 |
| `quality.json`、`ttft.json`、CSV | 逐例质量、体积与计时记录；具体文件随实验而定 |
| `report.md`、`reports/` | 汇总结果、图表和解释 |
| `artifacts.json`、`report_manifest.json` | 对应归档输出的文件大小与 SHA-256（若该批次提供） |
| `run.log`、`exit_code` | 执行日志和进程退出结果（若该批次提供） |

以具体批次的记录核对结果。现有小规模实验未覆盖论文完整的 anchor/delta 编码、跨服务器传输及传输与解码重叠；TinyLlama 自适应实验的网络时间来自轨迹估算。指标定义与限制见各实验协议和报告。

## 相关资料与原始项目

- [CacheGen 论文](https://arxiv.org/pdf/2310.07240.pdf)，SIGCOMM 2024。
- [LMCache 项目](https://github.com/LMCache/LMCache)。
- [原始实验说明](sigcomm_ae.md)及 [原始环境定义](env.yaml)。
- 原始项目联系：Yuhan Liu，yuhanl@uchicago.edu。
