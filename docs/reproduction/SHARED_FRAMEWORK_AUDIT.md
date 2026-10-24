# 公共框架审查（2026-10-24）

审查起点：`f714e6c`，实际路径 `/home/aaa/CacheGen`；Git 工作区干净，未发现适用的 AGENTS.md。本机为 RTX 4090 D 24GiB、cgroup 主存 30GiB、10 核 CPU 配额，已安装 tmux。原有路径说明中的 T4 和 WSL 属于历史环境。

## 修改前是否对齐

结论：Mistral 未完整对齐 TinyLlama。已有 `Workspace`、注册表、环境变量、监督进程及 GPU 锁，但正式实验是两套实现。

| 项目 | TinyLlama 原实现 | Mistral 原实现 | 公共实现 |
| --- | --- | --- | --- |
| 六种体积/质量方法 | 有 | 有 | 统一 Engine.encode 和 quality |
| 自然文本 NLL | 第 2..N 个续写 token | 同口径，但另写一份 | v2 重放最后上下文 token，统一评分全部 1..N |
| 受控开头/中间/末尾 × 长度 | 有 | 无 | 同一事实构造和逐长度门槛 |
| 完整 TriviaQA | 无 | 有，但按全部输出评分 | 独立补充任务；作者首行规则；保留完整生成 |
| TTFT | 文本、U8、CG2/3 | 同四种 | 任意已实现方法可配置加入 |
| 文本词表投影 | 全上下文 logits | 全上下文 logits | 只投影最后位置；同权重验证 |
| 自适应 | 六策略，四组倍率×SLO | 无 | 统一策略与实际 socket 执行 |
| 自适应网络时间 | 估算 | 无 | 默认 socket_serial；估算另名 trace_estimate |
| 缓存验证 | 局部 | 单输入、短生成；TTFT 首 token | 多真实输入、多长度、多 token、位置、mask、重复、拼接及 NLL |
| 高带宽 | 4KB Python 循环 | 复用同函数 | memoryview/recv_into、64KiB..1MiB 分段，先校准 |
| 检查点恢复 | 批次失败后不能续测 | 部分结果可见，不能严格续测 | SQLite FULL 逐观测提交、指纹验证和严格矩阵 |
| 源码归档 | 指纹为主 | 漏部分 mini_repro 帮助模块 | 保留目录结构的全部相关代码、数据与依赖源码 |

原协议是三个实验组、六个任务：quality、ttft、以及倍率 1/0.01 与 SLO 0.2/0.5 秒的四个 adaptive。策略来自 `mini_repro/adaptive.py`：adaptive、adaptive_static、fixed1、fixed2、fixed3、text；未自行删减。旧作者 `src/utils.py` 中的固定 chunk_delay 模拟不进入 v2 实际测量。

## 复用、合并与保留

直接复用 `mini_repro` 的 LongChat 选择、受控事实、量化打包、计时排程、带宽轨迹及策略定义。`next_decision` 由旧估算规划和新实际执行共同调用。传输合并到 `mini_repro/transport.py`，旧导入路径继续可用。

Mistral 原 542 行专用实验器已由公共框架和薄兼容入口替代；专用测量 shell 不再另行实现生命周期。重复的编解码、评分、QA/NLL、TTFT、汇总和元数据逻辑不再是正式入口。旧批次 source/ 中的原实现保持不变。`build_document.py` 和旧绘图/归档器保留用于历史报告；新报告使用公共 analysis。

`mini_repro` 的 v1 函数和历史 profile 保留以便核查既有测试与历史口径；当前正式入口统一使用 `experiments.suite`。协议差异与入口迁移见 [操作说明](SHARED_FRAMEWORK.md)。没有新增根目录平行项目，也没有移动或删除原始批次、模型或用户数据。

## 模块职责

- `experiments/config.py`：固定协议、模型 ID/revision、权重模式、层数、KV heads、head dimension、窗口和量化分层。
- `engine.py`：实际模型加载、显式位置/mask、新缓存容器、末位置投影、续写 NLL、QA、统一编解码。
- `datasets.py`：两模型共享选择和独立问题/事实身份；JSONL 使用文件逐行迭代。
- `tasks.py`：quality、ttft、adaptive，记录每次顺序、分项耗时、原始生成和选档。
- `records.py`、`suite.py`：检查点、配置/代码/数据/环境/模型一致性、完整性、失败和退出码。
- `artifacts.py`、`analysis.py`：资源预检、源码/依赖快照、分组统计、来源聚类区间、图表与校验清单。
- `background.py`：独立 tmux 启动及启动器日志/状态；同 GPU 使用共用排他锁。

## 实施中发现

TinyLlama 本地初始只有 tokenizer 和 20MiB 未完成权重。首次最小批次 `results/tinyllama/minimal/v2-check-01` 因缺权重在预检失败，未产生正式测量。公开固定 revision 的权重通过后台下载脚本补齐并校验；失败记录不覆写。

仓库真实 TriviaQA 最短也超过 1024 token；TinyLlama tokenizer 下完整输入不超过 2000 token 的问题仅 7 个，较大配置据此上限设为 7。Mistral 较大配置选择 20 个不超过 4096 token 的完整输入。受控事实仅有 10 个独立事实；扩大位置、长度、轨迹与计时次数不会把它们算成新独立事实。
