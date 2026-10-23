# W&B 原始日志与本地解码

这个仓库保存各项目的 W&B 原始 run，并提供不依赖云端的解码工具。原始日志和工具进入 Git；可重新生成的解码结果放在 `derived/`，不进入 Git。

## 数据流与目录

```text
开发机／集群的训练项目/wandb/
  → 日志增量包 → 人工下载到 Mac
  → ~/src/wandb-logs/MirrorOPD-verl/wandb/
      ├─ 本次收到的 run → 临时副本 → wandb sync → W&B 云端
      └─ 该项目全部日志变化 → Git commit + push
          → 其他机器 git pull → 本地解码 → JSONL / JSON / CSV
```

Mac 和分析机器直接访问本仓库，不在训练源码仓库保留 `wandb` 兼容目录或软链接。训练端仍按训练脚本的 `WANDB_DIR` 写入原始日志，再由训练项目的打包工具传出。

```text
wandb-logs/
├── MirrorOPD-verl/wandb/offline-run-.../   # 原始数据，进入 Git
├── tools/wandb_local.py                  # 本地读取工具
├── tests/                               # 解码边界测试
└── derived/MirrorOPD-verl/offline-run-.../# 派生结果，不进入 Git
```

同一 run ID 的不同目录分别导出，不按 run ID 自动合并，避免把重启或重复快照混成一次训练。下面的 `RUN` 可以是 run ID，也可以是完整目录名；CSV 导出必须匹配唯一目录。

## 安装与日常使用

需要 Python 3.10 或更新版本、Git 和 Make。解码依赖固定为 `wandb==0.28.1`，安装在仓库自己的 `.venv`，不改训练环境。

```bash
cd ~/src/wandb-logs
git pull --ff-only
make setup
make decode
make list
```

之后拉取并解码只需：

```bash
make refresh
```

`refresh` 先执行 `git pull --ff-only`，拉取失败即停止。读取器按原始 `.wandb` 的大小、修改时间和读取器版本跳过已导出的文件，不计算文件摘要。需要强制重建时执行：

```bash
.venv/bin/python tools/wandb_local.py decode --force
```

指定项目、run 或导出 CSV：

```bash
make decode PROJECT=MirrorOPD-verl RUN=krkmq098
make csv PROJECT=MirrorOPD-verl RUN=krkmq098
```

CSV 写到对应的 `derived/.../history.csv`。JSONL 保留各 step 实际记录的键；CSV 使用所有键的并集，缺失值留空，不补零。嵌套对象在 CSV 中编码为 JSON 字符串。JSONL 是分析的主格式，CSV 只用于方便查看。

MirrorOPD 的指标横轴、统计及绘图仍由训练源码仓库维护，不放进通用解码器。在 `MirrorOPD-verl` 源码仓库中执行：

```bash
python3 examples/mirroropd/docs/scripts/analyze_run.py export \
  --run wncfht/MirrorOPD-verl/krkmq098 --out outputs/e15-local-新快照
```

该入口调用本仓库的独立 Python 解码，并转换为项目分析所需的快照格式，不读云端或凭据。源码仓库的 `examples/mirroropd/docs/分析复现.md` 是项目分析的使用入口；本 README 是原始日志与解码协议的使用入口。旧云端导出在项目中改名为 `export-cloud`，不作为自动回退。

当前分析机器的本地仓库放在 `/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/wandb-logs`，与训练源码仓库并列。该机器的 Git SSH 操作先在同一 shell 中执行：

```bash
source /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/.ssh/activate.bash
```

该机器还在日志仓库的本地 Git 配置中设置了对应的 SSH config，使 `make refresh` 内部调用 Git 时也能认证。这个机器专用配置不进入 Git，Mac 使用自己的 GitHub 认证。

## 导出的数据

| 文件 | 内容 |
| --- | --- |
| `history.jsonl` | 所有成功解析的 history 记录，保持原顺序、稀疏键和原始 step，不采样 |
| `system.jsonl.gz` | 每条系统监控记录的 `timestamp` 与 `metrics`，不采样 |
| `config.json` | 按记录顺序应用 config 更新和删除后的结果 |
| `summary.json` | 按记录顺序应用 summary 更新和删除后的结果 |
| `run.json` | run 身份、记录计数、step 范围、退出码及解析状态 |
| `metric-definitions.jsonl` | 原始指标定义，包括自定义 step 轴等信息 |
| `console.jsonl` | 日志中包含的控制台输出记录 |
| `artifacts.jsonl` | 日志中包含的 artifact 描述和引用；不下载外部内容 |

`_step` 是 W&B history 的 step，不自动等同于 optimizer step 或训练外层步。分析时应结合 `metric-definitions.jsonl`、`training/global_step`、`step/outer` 等实际存在的字段选择横轴。

原始 `files/` 仍放在 run 目录中，不重复复制到派生结果。图片、表格、模型等文件的引用不会自动变成文件内容，原始文件或外部存储仍需保留。`config.json` 和 `summary.json` 从事务日志重建，不把 sidecar YAML/JSON 当作唯一事实源。

读取器保留 W&B/Python JSON 中的 `NaN` 和 `Infinity`，不默默改成零或空值。严格 JSON 消费端需要自行转换这些非有限数值。普通 JSONL 值中的大整数、布尔值、字符串和嵌套结构都保留。

## 完整性与错误

| 状态 | 含义 |
| --- | --- |
| `finished` | 读到了退出记录；是否成功取决于 `exit_code`，不是仅凭状态判断 |
| `snapshot` | 未读到退出记录，只能确认当前文件中的内容，不能据此区分运行中、崩溃或未完成传输 |
| `partial_snapshot` | 文件在分片记录中结束；已导出此前的完整记录，未完成的最后一条记录保留在原始文件中 |
| `decode_error` | 帧、Protobuf 或值解析失败，保留之前成功解析的记录，错误位置写入 `run.json` |
| `source_changed` | 读取过程中原始文件发生变化，需要等文件稳定后重跑 |

`partial_snapshot` 会显式提示，只分析已成功导出的部分；它不是一次完整训练的证明，也不能据此断言文件中间损坏。遇到 `decode_error` 或 `source_changed`，命令退出码为 1，CSV 命令不继续导出；批量解码仍处理后面的 run。不会静默跳过坏块后宣称完整。`list` 显示的 `stale` 表示原始文件在上次导出后变化，需要重新解码。

工具复用 W&B SDK 的分块读取器及 Protobuf 定义[^wandb-reader]。SDK 原来的扫描入口以 `r+b` 打开文件，因此本工具先只读取得单个文件快照，再让读取器扫描内存；峰值内存至少包含一个原始 `.wandb` 文件。工具不会调用 `wandb.init`、`wandb.sync`、Public API 或读取认证信息。后续 SDK 若扩展协议，需要升级依赖并重新运行测试；未知记录和缺失值会明确报错。

```bash
make test
```

测试覆盖稀疏指标、非有限值、大整数、配置与 summary 的更新和删除、嵌套键、截断日志、错误头部、缓存失效和只读输入。

### 参考文献

[^wandb-reader]: W&B contributors. SDK 0.28.1 的 [DataStore 读取器](https://github.com/wandb/wandb/blob/v0.28.1/wandb/sdk/internal/datastore.py)与 [Protobuf 定义](https://github.com/wandb/wandb/blob/v0.28.1/wandb/proto/wandb_internal.proto)。
