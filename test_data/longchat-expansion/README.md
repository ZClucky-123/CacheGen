# LongChat 扩展候选集：200篇提示

2026-11-25整理。**这是同源扩展候选集，不是已找到的CacheGen论文原始200篇。**未用于新实验，不覆盖现有`test_data/longchat.jsonl`。

## 查找与对齐结果

- CacheGen主分支、数据文件历史、`sigcomm_ae`/`cuenc`/`standalone`分支已核查，未找到论文完整200篇清单。当前公开LongChat文件仅50篇，作者AE说明其为小子集。
- LongChat官方`longeval`分支固定提交为`4b5faa52fda961b8ea5125b15ae984ff0d1c0f9d`。
- 官方`longeval/evaluation/topics/testcases/15_topics.jsonl`有50篇。与CacheGen当前50篇逐条核对：话题标签顺序、对话正文全部相同，CacheGen只添加统一的聊天模板。50/50可精确对应。
- 官方话题对话池`conversations.jsonl`只有30段基础对话，官方生成器每篇无放回抽取15个话题并排列。扩大提示数量不等于扩大基础语料多样性。
- 尚未恢复论文原200篇的采样顺序、随机种子和具体清单，不能用“同样200篇”冒充“原论文200篇完全一致”。未联系作者或发送外部消息。

## 已准备的候选文件

`longchat_200_expanded_candidate.jsonl`：保持原ID0–49的内容与标签，新增ID50–199共150个不重复话题排列；统一15话题、相同对话池、相同原始提示及CacheGen包装模板。随机种子为20261125。生成只用标准库做数据整理，没有调用模型、API或测试推理。

`manifest.json`：记录来源提交、各下载文件SHA256、候选文件SHA256、模型固定配置和每篇对应的话题ID与来源类别。

`sources/`：保存固定版本的原始50篇、30段话题对话、官方生成入口、配置及工具源码。源码仅供核对，没有执行其中的模型/API调用。

生成工具：`tools/prepare_longchat_expansion.py`，拒绝覆盖已有候选文件。抽样分布与官方方法一致；使用Python标准库`random.sample`，不声称重建官方原始随机序列。

## 固定模型与后续实验边界

按当前约定保持：`mistral-community/Mistral-7B-v0.2`，revision `80cf49b18de6354fd8a6d1e81c177dd830e05eea`，FP16权重。此次不切换FastChat 8bit加载方式，不改变模型权重或既有运行配置。

本文件尚未分配新的校准/验证/测试集合，也未用固定tokenizer核对200篇实际token长度。后续需在服务器离线分词，保留完整上下文，不能为贴合论文9.4K而擅自截断。

历史ID30–49已用于校准，原50篇也已用于反复观察和配置研究。不能把整个200篇称为从未见过的测试集。后续应单独报告原50篇与新增150篇，并为新配置确定不重叠的校准、验证来源。若需要200篇都作为未见测试样本，应另生成测试与校准集合，而非直接复用这份候选。

扩大到200篇只能增加同任务的排列覆盖与统计量，不能替代另一数据任务或模型的跨域验证。

## 官方来源

### 从克隆仓库准备输入

本次 GitHub 提交仅包含生成脚本与说明，不包含本目录的原始语料、扩展候选 JSONL 和机器生成的 manifest。请在实验服务器上准备输入；已有文件时不要重复生成或覆盖。

从上述固定 LongChat 提交分别获取以下两个文件，并保存到 `test_data/longchat-expansion/sources/`：

| 固定提交内路径 | 保存名称 | SHA256 |
| --- | --- | --- |
| `longeval/evaluation/topics/testcases/15_topics.jsonl` | `15_topics.jsonl` | `4b16257d64628e59c25fb44b8e9b8981dd51005ffecab31a442bf18dd12a799d` |
| `longeval/evaluation/topics/conversations.jsonl` | `conversations.jsonl` | `d4c04b04cd4bc0bf70d29b3d0dd8450f18facf32fc2435d9b9be26fc8463619f` |

检查文件 SHA256 后，在仓库根目录执行：

```bash
source scripts/experiment-env.sh
"$CACHEGEN_PYTHON" -m tools.prepare_longchat_expansion
"$CACHEGEN_PYTHON" -m tools.prepare_delta_diagnostics --output .local/delta-readiness-new
```

第一个步骤仅整理数据；第二步要求服务器已有固定模型快照，仅核对权重与分词、不做模型前向。只提供这两个必要来源也能生成候选；manifest 会记录实际存在的来源文件，可能与历史准备时保存全部辅助源码的 manifest 不同。历史结果须使用其归档指纹核对，不能用新生成 manifest 替换旧记录。

- [CacheGen AE说明](https://github.com/UChi-JCL/CacheGen/blob/6bed34ca9d495289955ed754cf2ad0c43346ee48/sigcomm_ae.md)
- [LongChat官方15话题50篇](https://github.com/DachengLi1/LongChat/blob/4b5faa52fda961b8ea5125b15ae984ff0d1c0f9d/longeval/evaluation/topics/testcases/15_topics.jsonl)
- [官方话题对话池](https://github.com/DachengLi1/LongChat/blob/4b5faa52fda961b8ea5125b15ae984ff0d1c0f9d/longeval/evaluation/topics/conversations.jsonl)
- [官方生成方法](https://github.com/DachengLi1/LongChat/blob/4b5faa52fda961b8ea5125b15ae984ff0d1c0f9d/longeval/utils.py)

上游文件保留原样并记录来源；本地保存和分析不等同于获得任意再分发许可，未对外发布本候选集。
