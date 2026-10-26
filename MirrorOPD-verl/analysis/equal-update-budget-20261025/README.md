# E18–E20 等更新预算分析数据

报告与脚本在同级源码仓库：[详细分析](../../../../MirrorOPD-verl/examples/mirroropd/docs/experiments/equal-update-budget/README.md)。原始日志使用本仓 `188690d` 中的三个 run；原始文件的最后更新来自 `ff8b9f0`。不复制原始 `.wandb` 或完整 history，直接读取通用解码器的结果。

| 文件 | 内容 |
| --- | --- |
| analysis.json | 源目录、cache 边界、实际设置、健康度、共同预算、验证、批内拟合、时间与资源统计 |
| configurations.json / config-differences.json | 实际配置的扁平表示与 11 个差异字段 |
| evaluation.csv | 每个实际验证点的 mean/CI、pass/majority、长度、终止与解析缺失 |
| metric-catalog.csv / metric-coverage.csv | 2235 个指标的逐 run 原始轴、观测次数、范围和覆盖；first20/last20 表示观测数，不是统一的更新窗口 |
| comparison.json.gz | 全部 history 数值指标的原始轴序列、真实坐标映射与分析结果；用于统计复算与绘图 |

三组完整记录分别为 64/128/189 outer、4096/4096/3024 updates；共同验证坐标为 512/1024/1536/2048/2560。E20 的 partial_snapshot 不代表远端作业必然失败，缺少最后退出证据。原始日志文件状态、记录数和截断位置保留在 analysis.json 的 source 字段。

KL 在对应更新之前观测；首末差覆盖 epochs−1 次已执行更新。新批起点和同批末次分别统计，换批比值只使用两个端点都在指定窗口内的记录。时间只合并 outer 与 validation；嵌套 actor、checkpoint 与 validation_rollout 不重复加总。

每次验证的 prompt/response/materialized/observed 数量已核对；没有缺失或非有限主评分。缺少逐题记录，未做跨 run 配对检验。位置分桶、ESS、entropy 没有记录，不补零。八组展示图的独立纵轴与 CI 端帽系列计数警告已逐图检查；图形审计无 error。

代码中的 analysis_base_commit 表示开发分析脚本时的源码基点，analysis_revision=1 表示本报告分析版本；脚本随报告一同保存。数值、坐标和窗口边界由真实日志复算及独立小样本测试核对。新原始日志应使用新分析目录；本快照的解释边界保持不变。

分析脚本与报告展示图保存在源码仓库提交 `719f67fa01defbac24c29da9adbdb04381d13ef3`。
