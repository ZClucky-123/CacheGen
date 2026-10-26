# Mistral 较大规模完整实验归档

批次 `v3-larger-stable-20261025` 于北京时间 2026-10-26 05:24 完成，7/7 任务、12594/12594 条正式观测，退出码 0。质量 1020、TTFT 9450、自适应 2100、消融 24。

[实验报告](report.md)、[汇总数据](summary.json)、[质量](quality.csv)、[TTFT](ttft.csv)、[自适应](adaptive.csv)、[流水线配对](pipeline_comparison.csv)和 figures/ 可直接用于本地分析。完成状态不代表已达到论文全部数值结论；方法边界见仓库的论文对照文档。

为满足 GitHub 单文件限制，完整原始批次按 gzip 压缩并分成不超过 40MiB 的文件，存于 archive/；包含原始 JSONL、SQLite、事件、时间线、源码、数据选择、报告和全部 175 项校验材料，不包含 SQLite 临时 sidecar。分卷及完整压缩包 SHA256 见 archive/manifest.json。

在本目录执行以下命令，恢复未压缩的大文件及全部原始文件（会以归档原件覆盖同名文件）：

```bash
cat archive/raw.tar.gz.part* > /tmp/cachegen-mistral-larger-raw.tar.gz
python3 - <<'CHECK'
import hashlib, json
from pathlib import Path
m = json.loads(Path('archive/manifest.json').read_text())
p = Path('/tmp/cachegen-mistral-larger-raw.tar.gz')
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['sha256']
print('Archive SHA256 verified')
CHECK
tar -xzf /tmp/cachegen-mistral-larger-raw.tar.gz -C ..
```

恢复后在仓库根目录运行 `python3 -m experiments.suite --verify results/mistral/larger/v3-larger-stable-20261025`，或按 artifacts.json 对每个原始文件逐项检查 SHA256。模型权重和 Python/CUDA 环境不在结果归档内，本地读取 JSON/CSV 无需 GPU。
