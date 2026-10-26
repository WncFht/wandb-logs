# E35 长窗口与 E45/E47 镜像目标分析证据

原始日志冻结于 merge `080cd7f244182cee6023aff8c7fdac0f225427e2`，包含原本地两个分析归档提交及远端三个日志接收提交。五个 run：E17=cix5i2x2、E20=lwgb5eex、E35=ntzdb4o7、E45=625vw4ob、E47=e7pkdbtg。

本目录保留报告引用的逐项统计、配置、指标目录和无采样数值序列。原始 .wandb 已在日志仓库中，不重复归档完整 history 副本。E45 的原始尾部截断，完整记录止于 outer 244 / update 3904；其他四组均有 exit=0。没有使用云端状态覆盖这些事实。

| 文件 | 内容 |
| --- | --- |
| analysis.json | 运行边界、配置差异、精确共同验证点、窗口、位置、恒等式及成本 |
| configurations.json | 五组完整展开配置，凭据已由项目导出器清理 |
| metric-catalog.csv | 全部数值字段的原始横轴、覆盖与统计 |
| series.json.gz | 全字段无采样序列，保留原始业务横轴 |

报告与可编辑分析/绘图脚本在同级源码仓库的 `examples/mirroropd/docs/experiments/length-mirror-comparison/` 和 `scripts/`。从源码仓库根目录复算：

```bash
# 在原始日志处于上述冻结提交的工作区中，先完成 make decode。
# 使用独立工作区可以保留当前 main 的更新，不要覆盖更新后的日志。
LOGS_REPO=../wandb-logs
CACHE="$LOGS_REPO/derived/MirrorOPD-verl/comparison-20261026"
for rid in cix5i2x2 lwgb5eex ntzdb4o7 625vw4ob e7pkdbtg; do
  python3 examples/mirroropd/docs/scripts/analyze_run.py export \
    --run "wncfht/MirrorOPD-verl/$rid" --logs-repo "$LOGS_REPO" \
    --out "$CACHE/exports/$rid"
done
python3 examples/mirroropd/docs/experiments/scripts/analyze_length_mirror.py \
  --exports "$CACHE/exports" --output "$CACHE/results" \
  --logs-commit 080cd7f244182cee6023aff8c7fdac0f225427e2
```

Export 拒绝覆盖已有目录；复算使用新的 CACHE 目录，或直接复用已核对来源的导出缓存。静态绘图使用已有原生 matplotlib 环境：

```bash
DATA="$LOGS_REPO/MirrorOPD-verl/analysis/length-mirror-20261026"
"$PYTHON_BIN" examples/mirroropd/docs/experiments/scripts/plot_length_mirror.py \
  --data "$DATA" --out examples/mirroropd/docs/experiments/length-mirror-comparison/figures
```

figure-viz 的定位沿用源码仓库 fvlocate；可通过 FIGURE_VIZ_MPL 指定现有 skill 的 assets/mpl。分析不占用 GPU、不请求云端、不修改训练设置。所有 marker 是实际验证点；不插值补点、不把重复 checkpoint 当 seed、不把 raw MDPO objective 当作 teacher KL。
