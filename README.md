# CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming

## 复现结果与实验入口

- **2026-10-24 Mistral-7B / RTX 4090 D：**[中文实验报告](reports/mistral/20261024-mistral7b/小规模复现实验报告.md)、[图表 PDF](reports/mistral/20261024-mistral7b/图表合集.pdf)。已完成 30 个文本条件、10 个问答、540 次 TTFT 计时。
- [全部实验结果](results/reproduction/README.md)：分别列出 Mistral 和历史 TinyLlama 批次。
- [工作区与扩展指南](docs/reproduction/WORKSPACE_GUIDE.md)、[Mistral 实验说明](experiments/mistral/README.md)、[环境说明](docs/reproduction/MISTRAL_ENVIRONMENT.md)。

在仓库根目录使用统一入口；查看状态和预演只需 Python 标准库：

```bash
python3 -m experiments list
python3 -m experiments status mistral
python3 -m experiments run mistral --dry-run
# 启动新的测量；需要已配置的模型、Python 与 CUDA 环境。
python3 -m experiments run mistral
python3 -m experiments run tinyllama -- --profile mini_repro/profiles/formal4070.json
```

代码在 `experiments/`，原始记录在 `results/`，报告在 `reports/`；`.local/`、`.venv/` 和 CUDA 工具链仅保留本机。旧 `mistral_repro` 路径和 `mini_repro` 模块继续兼容。新增实验通过注册表接入，位置覆盖与归档规则见工作区指南。

**For the latest update and integration, please check out the [LMCache](https://github.com/LMCache/LMCache) project!**

This is the code repo for [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf) (SIGCOMM'24). 
The code structure is organized as follows:

- ```LMCache```: The modules for KV cache encoding / decoding with CacheGen's customized codec 
- ```test_data```: The example testing cases for CacheGen. 
- ```src```: Some helper functions used by CacheGen (e.g., transforming tensor to tuple, transforming tuple to tensor etc.)

## Installation

To install the required **python** packages to run CacheGen with conda
```
conda env create -f env.yaml
conda activate cachegen
pip install -e LMCache
cd LMCache/third_party/torchac_cuda 
python setup.py install
```

### Examples 

Please refer to the page [sigcomm_ae.md](sigcomm_ae.md) for running examples for CacheGen. 

### Contact 
Yuhan Liu (yuhanl@uchicago.edu)
