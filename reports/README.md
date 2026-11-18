# 可分享的实验报告

- [核心复现与论文差异记录（2026-11-18 更新）](mistral/20261029-paper-core-analysis/论文差异记录.md)：逐条对照论文结论；开头是全长实验的汇总判定，后面保留 4K 版本、原因分析和更正。

- [全长 TTFT（2026-11-14）](mistral/20261114-fulllength-ttft/README.md)：20 篇实测解码和首 token 时间，按带宽算延迟；比 8 bit 快 2.6–3.6 倍，比传文本快多少取决于带宽。

- [全长论文差分方法补充检验（2026-11-14）](mistral/20261114-fulllength-paper-abs/README.md)：bin 按原始单位、补测范围 4；差分仍无收益，浅层精度很关键。

- [全长论文完整压缩方法（2026-11-04）](mistral/20261104-fulllength-paper/README.md)：差分 + 分层 + 算术编码 + 离线共享概率表，并拆开差分和分层的贡献。

- [全长算术编码与更低精度（2026-11-02）](mistral/20261102-fulllength-ac/README.md)：分清算术编码和分层的贡献，找出准确率下降的位置。

- [全长 LongChat 质量验证（2026-11-02）](mistral/20261102-fulllength-qa/README.md)：50 篇不截断，8/4/3 bit 与作者代码三个等级的体积和准确率。

- [4K K/V 定向复验（2026-10-27）](mistral/20261027-4k-loss-probe/README.md)：云端四来源复验，44 条评分、20 组编解码一致性与 K/V 精度单独对照，约 92 秒完成。

- [Mistral 较大规模结果分析（2026-10-26）](mistral/20261026-larger-analysis/实验结果分析.md)：12594 条观测的压缩/质量、双文本基线、流水线、自适应首块及消融分析，附离线复算脚本和四页图表。

- [4K 损失分布与原因分析（2026-10-27）](mistral/20261026-larger-analysis/loss_distribution/损失分布与原因分析.md)：逐来源、逐 token 配对与留一敏感性；区分粗量化证据、内容敏感性和上下文长度混杂。

- [共享 v3 稳定版本：大上下文修复与验收](shared-v3-stable-20261025/验收报告.md)：Mistral 469 条完整预验证、严格 KV/NLL、真实恢复和固定源码证据，附当前运行入口。

- [共享 v3：两模型功能验收](shared-v3-20261025/验收报告.md)：各 7/7 任务、274 条观测，含流水线配对、时间线、历史吞吐及消融。

- [共享 v2：Mistral 完整小规模验收](mistral/v2-small-final-20261024/验收报告.md)：6/6 任务、552 条观测，含逐例 QA、三组 PNG/SVG/PDF 与完整原始记录链接。

- [Mistral-7B / RTX 4090 D，2026-10-24](mistral/README.md)：中文报告、7 组图表和逐题问答。
- [TinyLlama / RTX 4070 Laptop，2026-10-13](../results/reproduction/20261013T060232923276Z/README.md)：历史正式批次与自动报告。

原始记录位于 `results/<model>/<scale>/<run-id>/`。上面两份 v3 报告对应不同源码和执行设置，分别解释；旧验收不重写。报告展示不替代原始状态、参数、记录和校验清单。配置与论文边界见 [框架指南](../docs/reproduction/SHARED_FRAMEWORK.md)及 [论文对照](../docs/reproduction/PAPER_ALIGNMENT.md)。
