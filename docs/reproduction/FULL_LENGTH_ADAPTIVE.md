# 全长 LongChat 带宽自适应

2026-11-24已核对：1200条正式条件完整完成，90个下载文件哈希匹配。自适应没有全面优于固定低精度或纯文本。完整结论见 [结果报告](../../reports/mistral/20261123-fulllength-adaptive/README.md)。下文启动与失败记录按历史保留。

目的：补齐之前全长实验未测的动态带宽自适应。复用作者公开实现的三个压缩精度和当前真实socket传输代码，不把11月的“体积÷带宽”TTFT计算当作动态实测。

## 固定协议

- Mistral-7B FP16、4090D；测试ID0–19，不截断，约8.6–9.3K token。发送端只处理上下文KV；问题留给接收端现算。
- 上下文完整prefill一次，再沿token分块（1536 token，保留尾块）。每块独立编码为作者公开实现三种精度及文本；概率表随KV包发送，全部计字节。该路径没有差分。
- 五种策略：已完成历史吞吐自适应、固定低/中/高精度、纯文本。最高精度为`cachegen3`。所有策略复用相同字节包；没有当前/未来带宽信息的策略输入。
- 三条轨迹：恒定3Gbps；`[10,10,3,1,0.5,1,3,10]`；`[0.5,0.5,1,3,10,3,1,0.5]`。按文档的实际块数等距取样，保持两端点，记录完整轨迹。
- SLO固定0.5/1/2/4秒。0.5/1秒沿用之前设置；2/4秒是全长诊断预算，不根据测试结果调整。三个带宽条件、四个SLO、五种策略、20篇、每条件一次，共1200条；单篇试跑60条，不合并进正式结果。
- 沿用保守策略：重算可行优先文本，否则优先可行高精度KV；扣除已花时间、实测首token开销和安全余量。没有可行表示也继续完成请求，报告预测不可行与实际违约，不保证SLO一定能达到。

## 实测与边界

通过单机socketpair真实发送、限速、解码/重算；每块决策后，传输层才读取该块配置带宽。首块前发送1MiB探测，之后只使用已完成传输。探测时间和流量在五种策略中统一计入，比较的是策略，纯文本基线因此也承担探测开销。

TTFT从探测前开始，到原始问题现算并得到首token为止。生成和评分在计时外，报告话题包含率、与原始回答逐字一致率、对原始回答token的条件NLL及其差值。NLL包含完整问题的条件，不能拿上下文后续32token的旧NLL直接合并。

离线成本已知：包大小、原始前缀上的恢复/重算代价（预热后3次取最大值）、首token保留时间。这是实验假设。源KV放CPU，GPU只保留当前处理的块/前缀，避免全长双副本OOM。每次实际混合KV回答均重新评估。调度按块串行；带宽逐块变化，不是墙钟时间回放，不是实际WAN。20篇是独立样本，不把1200个条件当作1200篇独立文档。

代码和实验数据仅在服务器运行。配置：`paper_compression/configs/full-length-adaptive.json`；入口：`tools/full_length_adaptive.py`。所有新输出必须使用不存在的目录，失败写状态并停止，不覆盖旧实验，不自动接续不兼容产物。

## 服务器步骤

```bash
source scripts/experiment-env.sh
(
  cachegen_lock_gpu
  "$CACHEGEN_PYTHON" -m unittest discover -s tests -p test_full_length_adaptive.py -v
)
# 单篇试跑；通过后用另一目录启动正式批次。
(
  cachegen_lock_gpu
  "$CACHEGEN_PYTHON" -u -m tools.full_length_adaptive --ids 0 --output .local/full-length-adaptive-pilot
)
(
  cachegen_lock_gpu
  "$CACHEGEN_PYTHON" -u -m tools.full_length_adaptive --output .local/full-length-adaptive-full
)
```

正式批次使用nohup启动脚本、总日志、PID和exitcode，断开SSH继续运行。逐条记录包含所选表示、带宽估计来源、预算、预测、实际网络时序、质量；各来源另存准备成本和运行顺序。`status.json`持续记录完成条数，结束检查完整无重复矩阵并写`summary.json`。

本地只写代码与读取产物；不运行模型、CUDA编译或测试。尚未产生本轮实验结论；预计耗时以单篇试跑实测后估算。

## 2026-11-23 启动记录

服务器6项单测通过。ID0的完整上下文为8873 token，单篇60条试跑完成，状态completed、exitcode为0、字节核对全部通过，计入实验步骤约210.6秒。试跑保存在 `.local/full-length-adaptive-20261123/pilot/`，不并入正式统计。

正式批次在试跑通过后由后台控制脚本启动：`.local/full-length-adaptive-20261123/full/`，预计1200条。总日志 `full.log`、PID `full.pid`（启动时1222）、结束码 `full.exitcode` 和启动脚本 `launch-full.sh` 均在该日期根目录。预计约1–1.5小时，待实际进度修正。当前仅确认正式批次已启动，不能提前判定质量或SLO结论。新增代码尚未提交或推送。

## 显存修复与分批续跑

首次正式运行完成ID0–8共540条后，在ID9回答生成的`legacy_to_blob`处申请1.14GiB失败。原目录及失败日志保留，原脚本另备份到日期目录的`original-code/`。原因是逐token生成反复堆叠整份KV，同时保留原上下文和首token缓存。

入口改用原生DynamicCache继续贪心解码，保留原上下文供回答NLL评分；不改精度、压缩、传输、问题、生成上限或TTFT计时。服务器7项回归测试通过，其中包括新旧生成逐token一致、EOS终止、原上下文不变及禁止调用全量KV堆叠的检查。本地未执行测试。

续跑控制脚本为`.local/full-length-adaptive-20261123/launch-remaining.sh`：先核对真实Mistral全长ID0新旧回答完全一致，并验证ID9失败条件；只有全部成功才在新目录`remaining/`运行ID9–19，共660条。验证结果写`memoryfix-verification.json`，日志、PID、退出码分别为`remaining.log`、`remaining.pid`、`remaining.exitcode`。原540条与续跑660条需核对完整条件矩阵后合并，并披露两批源码指纹与显存修复；不能将单篇试跑或修复验证混入正式统计。

真实验证已通过：ID0新旧生成token完全一致；ID9完整9288 token的原失败条件（constant3、4秒、纯文本）完成，回答正确、Δ回答NLL为0，验证阶段GPU峰值实际分配约19.73GiB。ID9–19已由PID9624控制脚本启动。`launch-combine.sh`在续跑成功后校验1200个条件完整无重复、字节核对、配置与数据一致，再输出`combined/`及两批指纹清单；失败则停止，不覆盖原结果。
