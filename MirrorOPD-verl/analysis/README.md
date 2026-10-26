# MirrorOPD 实验分析证据

这里保存正式实验报告的冻结输入与分析产物。数据从 MirrorOPD 源码文档目录迁入，原始数值、API／本地来源和历史截断边界保留；不把旧快照自动补齐成当前完整 run。

| 目录 | 对应内容 |
| --- | --- |
| e15-krkmq098-20261023 | E15 的完整训练动态输入与统计 |
| five-run-comparison-20261024 | E13–E17 五组比较；E17 为当时截断快照 |
| teacher-convergence-20261024 | 师生参考和有限分布验算 |
| position-convergence-20261024 | E16/E17 批内与后期收敛统计 |
| oneshot-student-6uorrk2k-api-20261020 | 历史学生 val-only API 快照 |
| oneshot-teacher-u87buult-api-20261020 | 历史教师 val-only API 快照 |
| equal-update-budget-20261025 | E18–E20：共同 optimizer 预算、批内 KL、生成行为、验证与成本，含全部指标序列 |
| inventory-20261025 | 首次盘点的云端清单、配置与末次结果；作为历史快照保留，日常更新不再复制整套配置 |
| length-mirror-20261026 | E35 长窗口与 E45/E47 镜像目标：共同预算、位置、内层拟合和成本，保留完整数值序列 |

原始 .wandb 在 `../wandb/`，可重建解码结果在仓库根目录的 `derived/MirrorOPD-verl/`。日常运行汇总也写在 `derived/MirrorOPD-verl/inventory/`；只有具体报告需要固定数据边界时才新增归档。

实验设计、报告正文、展示图和项目专用脚本继续由同级源码仓库维护，见[实验目录](../../../MirrorOPD-verl/examples/mirroropd/docs/experiments/README.md)与[复现说明](../../../MirrorOPD-verl/examples/mirroropd/docs/experiments/复现与维护.md)。通用 .wandb 解码器仍只在本仓库维护。跨机使用时拉取两个仓库；本地相对链接按两个 checkout 同级排列。
