# 2026-10-24 工作区整理与兼容性重构

用户要求全部文件实体归入 `/home/aaa/CacheGen`，仅该仓库上传 GitHub，并允许为后续实验大幅调整目录。现在 `/home/aaa` 只有 `CacheGen` 一个条目。

## 原始情况与整理结果

原目录包含一个 Git 仓库 `CacheGen`、一个无独立 Git 历史的 `cachegen-repro` 工作区、6 个安装/验证脚本、2 个版本清单、旧下载依赖和空文件 `dada.md`。前者已有 TinyLlama 历史实验，后者是刚完成的 Mistral 实验。

- Mistral 代码移入 `experiments/mistral`，结果移入 `results/mistral`，中文报告移入 `reports/mistral`。
- 共用标准库调度、路径发现和环境设置；增加注册表，让新实验独立接入。
- 下载工具、字体、绘图依赖、迁移备份和空文件归入 `.local`；Python 环境、模型和 CUDA 留在仓库内部的原路径。
- `mistral_repro` 及 Mistral 原 `runs`、`deliverables`、`latest` 使用相对链接兼容；本机 `/root/cachegen-repro` 改指仓库内入口。
- 安装脚本改用自身相对路径，移除机器固定进程号；原脚本保存在 `.local/migration/20261024-workspace/setup-before/`。

完整结构、后续运行和环境变量见 [工作区指南](WORKSPACE_GUIDE.md)，新增实验见 [注册说明](../../experiments/README.md)。

## 本轮结果

[成功批次](../../results/mistral/20261024T064543Z/)在北京时间 14:45:47–15:20:30 完成，有 30 个自然文本条件、10 个完整问答和 540 次正式 TTFT 计时。manifest 为 completed、退出码为 0，后处理完成。首次编译失败的批次也保留，见 [批次索引](../../results/mistral/README.md)。

[中文报告](../../reports/mistral/20261024-mistral7b/小规模复现实验报告.md)、[图表 PDF](../../reports/mistral/20261024-mistral7b/图表合集.pdf)保持原始字节。CacheGen 3 压缩约 4.41–4.91 倍；相对 uniform8 的 TTFT 速度比约 2.16–2.49，相对文本重算未见中位 TTFT 优势；完整输出 QA 的联合质量门槛未获支持。

## 兼容性修正与核验

新增测试覆盖换目录、含空格路径、路径覆盖、无历史结果、新实验注册、参数原样转发、GPU 排他锁和子进程失败返回。

运行入口监督子进程组，转发 INT/TERM 并等待退出；预检开始前创建 manifest，失败或中断保留状态和非零退出码。已完成的后处理只核验归档；校验失败写单独诊断文件，保留原始 SHA-256 清单。

历史批次的模型、配置、源码、原始结果和图表不因本次重构而重算或改写。活动代码和导航有意调整；迁移前逐文件指纹、原始版本及 Git index 备份保存在 `.local/migration/20261024-workspace/`。首轮整理的记录另在 `.venv/organization-20261024/`，作为更早阶段的历史现场保留。

开始整理时暂存区有 8,751 个 CUDA 工具链文件。为避免把本机环境上传，在备份 index 后取消这部分暂存，实体文件继续留在 `.cuda-12.1`；新忽略规则排除本机依赖，只有显式归档的结果和报告加入仓库。

## 发布边界

沿用 `origin` 仓库和 `mini-repro-tinyllama` 分支，不合并到 main、不改写远端历史。GitHub 只接收仓库中的代码、说明、测试和核验过的原始批次与报告，环境、模型、wheel 和本机下载/编译缓存保留在忽略目录。

用户随后要求不上传 ZIP 并删除本地 ZIP：已删除报告压缩包及临时上传载荷，展开文件继续保留；未上传的本地提交重新整理为不包含 ZIP 的版本。
