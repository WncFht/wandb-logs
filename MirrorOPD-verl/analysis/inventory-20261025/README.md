# 2026-10-25 实验运行快照

云端清单冻结于 2026-10-25 09:25（北京时间），项目 `wncfht/MirrorOPD-verl`；本地 `.wandb` 来自 `wandb-logs` 提交 `ff8b9f0`，由该仓库的通用读取器解码。原始二进制和全量 history 继续在独立日志仓库维护。

| 文件 | 内容 |
| --- | --- |
| cloud-index.json | 当时的 43 个云端 run 名称、ID、状态；只读 API 查询 |
| inventory.json | 23 个正式训练 run 的实际设置、进度、源目录与原记录 Git 字段；另外列出排除的 20 个工程 run |
| runs.csv | 相同正式运行记录的表格版本；百分比列以 0–100 表示 |
| runs/ID/config.json | 从本地事务日志重建并脱敏的完整实际配置 |
| runs/ID/evaluations.json | 每次验证的 outer、optimizer 坐标及原始 mean accuracy 字段和值；值以 0–1 表示 |

总表按名称识别正式 Math E 系列和 math/code/if-full 训练。未发现的实验配置写在系列 README；目录存在、启动入口存在或短测通过不等于正式实验完成。

E17/E18/E19/E20 的 `update_applied` 记录合计分别为 3200/4096/4096/3024，均未观察到 skipped update。E20 最后完整 outer 为 189，最后验证为 outer 160/update 2560。旧字段无实际更新计数时保留 null，不用 outer 或事件数冒充 optimizer 更新。

复算入口见[复现与维护](../../../../MirrorOPD-verl/examples/mirroropd/docs/experiments/复现与维护.md#数据归属与更新实验运行记录)。生成器从解码后的 history 提取验证坐标，不把 summary 中的旧验证值放在末次更新位置。此目录作为首次盘点的历史快照保留。日常更新直接读取解码结果，汇总写入 `derived/MirrorOPD-verl/inventory/`，不再复制各 run 的 config/evaluations。
