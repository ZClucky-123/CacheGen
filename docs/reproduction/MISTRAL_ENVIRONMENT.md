# Mistral-7B 环境验证

本文说明 Mistral 环境的配置、存储位置和验证方法。验证通过后，按[复现操作说明](MISTRAL_REPRODUCTION.md)运行实验。记录日期：2026-10-23。

## 1. 环境与验证记录

| 项目 | 配置与验证记录 |
| --- | --- |
| 源实例 | Tesla T4，15360 MiB 显存；Ubuntu 22.04.4，x86_64 |
| 环境与模型 | 已安装；Mistral 三个权重分片已通过 SHA-256 校验 |
| 环境检查 | `pip check` 通过；仓库 41 项测试通过，包含 GPU 编解码测试 |
| Mistral 验证 | 离线加载、短文本生成、真实 KV 编解码、解码后缓存续算通过 |
| CUDA 扩展 | 已检查二进制，包含 `sm_75` 和 `sm_89` |

环境验证时的源码提交为 `cc14424f75ed69b695e8e4e3e06e2728a083ce4f`。本次文档补充晚于该验证，不修改历史记录中的提交号。

## 2. 存储位置与镜像迁移

源实例中的路径如下，均位于**系统盘**，不在 `/root/bayes-tmp`：

| 内容 | 路径 |
| --- | --- |
| 项目 | `/home/aaa/CacheGen` |
| Python 环境 | `/home/aaa/CacheGen/.venv` |
| CUDA 12.1 编译器及开发头文件 | `/home/aaa/CacheGen/.cuda-12.1` |
| Hugging Face 模型缓存 | `/home/aaa/CacheGen/.venv/huggingface` |
| Torch 扩展缓存 | `/home/aaa/CacheGen/.venv/torch_extensions` |
| 环境说明 | `/home/aaa/CacheGen/.venv/README.server.md` |
| 安装、验证日志 | `/home/aaa/CacheGen/.venv/logs` |

交接核对时系统盘约 52GB、已使用约 31GB；目标实例还需为实验生成的 KV 和结果预留空间。平台保存范围、容量限制、是否需关机以及镜像可用区域，以控制台说明为准。

## 3. 固定模型与软件版本

模型是 **`mistral-community/Mistral-7B-v0.2`**，不是 Mistral Instruct，也不是 TinyLlama。固定 revision 为：

```text
80cf49b18de6354fd8a6d1e81c177dd830e05eea
```

三个权重分片合计约 14.49GB，逐文件大小与 SHA-256 见[校验记录](mistral-server-20261023/mistral-download-verified.json)。本地缓存的 `refs/main` 已指向上述 revision，供原脚本按模型 ID 离线加载。

| 依赖 | 已验证版本 |
| --- | --- |
| Python | 3.10.12 |
| PyTorch / CUDA runtime | 2.3.1+cu121 / 12.1 |
| CUDA 编译器 | nvcc 12.1.105 |
| Transformers / NumPy | 4.42.3 / 1.26.4 |
| FastChat / Accelerate | fschat 0.2.36 / 0.32.1 |
| bitsandbytes / PEFT | 0.43.3 / 0.11.1 |
| torchac / torchac_cuda / LMCache | 0.9.3 / 0.0.0 / 0.1 |
| packaging / WaveDrom / protobuf | 24.1 / 2.0.3.post3 / 5.27.2 |

完整[依赖版本清单](mistral-server-20261023/requirements-server-versions.txt)用于核对。

`src/utils.py` 的原始 Mistral 加载路径使用 FastChat `load_8bit=True`。这是模型权重的加载方式，与 CacheGen 的 KV 量化等级是两件事。后续比较方法时应保持模型加载方式一致，不应悄悄换成另一种权重量化或另一模型版本。

## 4. 环境验收

以下命令用于已恢复完整镜像且路径保持不变的实例：

```bash
cd /home/aaa/CacheGen
source .venv/cachegen-env.sh

python -c 'import sys; print(sys.executable)'
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -m pip check
python -m unittest discover -s tests -v

# 如需保留源机器的历史记录，先复制它；下一个命令会更新此 JSON。
cp -n .venv/mistral-smoke.json .venv/mistral-smoke-source-t4.json
python .venv/tools/mistral_smoke.py
```

验收应确认：解释器来自项目 `.venv`；GPU 是目标 4090D；CUDA 可用；`pip check` 无冲突；测试通过；模型脚本打印 `MISTRAL_GPU_SMOKE_PASSED`，并在 `.venv/mistral-smoke.json` 中写入 `status: passed` 和目标 GPU 名称。测试数量在后续代码更新后可能改变，应连同提交号记录。

激活脚本已设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和模型缓存位置。正常模型验证不需要再次下载，也不依赖之前的 SSH 代理转发。

如缺少 `.venv/tools` 文件，仓库保留了已验证的[验证工具副本](mistral-server-20261023/tools/mistral_smoke.py)和[激活文件副本](mistral-server-20261023/cachegen-env.sh)。恢复前先核对路径和镜像内容；这些副本不能替代缺失的 Python 环境、CUDA 扩展或模型。

这些命令检查环境、模型和编解码功能，不对应论文中的定量实验。论文图表、实验脚本和测量口径见[复现操作说明](MISTRAL_REPRODUCTION.md)。
