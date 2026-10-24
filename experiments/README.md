# 实验注册与运行

`python3 -m experiments list|run|status` 是不依赖 PyTorch 的统一调度入口。每组实验保留自己的模型、配置验证和测量实现，注册表只决定调用哪个命令、在哪里查看结果。

| 名称 | 代码 | 配置 | 结果 |
| --- | --- | --- | --- |
| `mistral` | [mistral/](mistral/) | [config.json](mistral/config.json) | [results/mistral](../results/mistral/README.md) |
| `tinyllama` | [mini_repro](../mini_repro/README.md) | [profiles](../mini_repro/profiles/) | [results/reproduction](../results/reproduction/README.md) |

```bash
python3 -m experiments run mistral --dry-run
python3 -m experiments run mistral -- --config experiments/mistral/config.json
python3 -m experiments status mistral
python3 -m experiments status mistral --run-dir results/mistral/20261024T064543Z
```

`--dry-run` 打印实际命令、工作目录和路径覆盖，不写结果、不加载 GPU。`--` 之后的参数逐项交给原实验命令，不经过 shell 拼接。若要预演 TinyLlama 内部的六个任务，使用 `bash mini_repro/run_local.sh reproduce --dry-run`。

## 新增实验

1. 在 `experiments/<name>/` 放置代码、配置及测试，复用现有 `mini_repro` 公共算法。
2. 在 [registry.json](registry.json) 增加一项，例如：

```json
{
  "my_experiment": {
    "description": "简短说明",
    "command": ["{python}", "{repo}/experiments/my_experiment/run.py"],
    "results_subdir": "my_experiment"
  }
}
```

3. 命令读取 `CACHEGEN_RESULTS_DIR`、`CACHEGEN_REPORTS_DIR`、`CACHEGEN_LOCAL_DIR`，在 `results/<name>/<唯一批次>/` 保存 manifest、配置快照、日志和原始记录；先提供不加载模型的预演。
4. Shell 入口可以 `source scripts/experiment-env.sh`，实际 GPU 测量前调用 `cachegen_lock_gpu`。Python 入口可通过 `Workspace.discover()` 读取相同路径。使用统一 `.local` 的本机测量共享一把锁。

历史 `mini_repro` 模块仍是公共工具与 TinyLlama API，不通过改名破坏既有导入。模型专用的维度、加载方式和评分约定放在各自实验中。
