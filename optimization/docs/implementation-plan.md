# CacheGen联合实现批次

设计版本为联合方案v2.2，基线为`d6b6626014c30b64011e4348c02510fdfbdb7204`。

| 批次 | 模块 | 验证风险 |
| --- | --- | --- |
| 时钟、源、缓存及动态线格式 | joint_types/source/cache/wire、transport动态入口 | 生命周期、错误身份、缓存计费和网络超时 |
| 三档三编码及CUDA | joint_codec/cuda、独立扩展 | 符号、CDF、字节序、尾块及FP16逐位等价 |
| 预测和控制 | joint_scheduler/policy/profiles/runtime | 单GPU竞争、期限、选择最优性、失效快照及取消 |
| 校准与交付 | joint_calibration/config/CLI、配置、脚本和指南 | 数据隔离、验收门槛、冻结档案及输出隔离 |

本机检查使用小张量、整数参考、假时钟、虚拟执行器和socket。目标设备承担CUDA构建、数值等价、真实质量、服务成本、残差和独立验收。正式矩阵读取已冻结且验收通过的档案。
