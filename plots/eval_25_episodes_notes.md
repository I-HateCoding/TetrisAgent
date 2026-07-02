# 25 局评估图说明

- 数据来源：25 个 episode 的 `reward`、`steps` 和 `avg_decision`。
- 图文件：[eval_25_episodes.svg](./eval_25_episodes.svg)
- 原始数据：[eval_25_episodes.csv](./eval_25_episodes.csv)

## 当前统计

- `Average reward = 96881.12`
- `Median reward = 64986`
- `P25 reward = 25730`
- `Worst reward = 2874`
- `Best reward = 417044`
- `Average steps = 16683.72`
- `Average decision time = 0.00572 s`

## 简要结论

- 模型已经能稳定打到几千到上万的 reward。
- 高分局上限已经很高，说明策略具备较强生存能力。
- 仍然存在少量极端低分 episode，模型稳定性还有改进空间。
- 报告中建议同时汇报 `Average reward`、`Median reward`、`P25 reward` 和 `Worst reward`，不要只展示最高分。
