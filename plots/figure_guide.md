# 图表使用建议

## 结果图

- [eval_25_episodes.svg](./eval_25_episodes.svg)
  逐局折线图，适合展示 25 局评估的波动情况。

- [eval_25_boxplot.svg](./eval_25_boxplot.svg)
  同风格 reward 箱线图，适合展示奖励的四分位数、非离群范围和高分离群局。

- [eval_25_summary_bar.svg](./eval_25_summary_bar.svg)
  奖励汇总柱状图，展示 `Average / Median / P25`。

- [eval_25_steps_bar.svg](./eval_25_steps_bar.svg)
  步数汇总柱状图，和奖励图同风格，适合成对展示。

- [eval_25_reward_profile.svg](./eval_25_reward_profile.svg)
  更偏结果分析的稳定性图，强调高均值不等于高稳定性。

- [eval_25_reward_steps_scatter.svg](./eval_25_reward_steps_scatter.svg)
  `reward vs steps` 散点图，适合说明“生存越久，reward 越高”，以及存在高分离群局。

## 方法图

- [tetris_feature_infographic.svg](./tetris_feature_infographic.svg)
  棋盘特征示意图，适合放在“特征增强 DQN”或“状态表示优化”部分。

## 推荐搭配

- 如果报告偏简洁：用 `summary_bar + steps_bar + feature_infographic`
- 如果报告更完整：用 `summary_bar + steps_bar + reward_steps_scatter + reward_profile + boxplot + feature_infographic`

## 当前风格

- 统计图统一为白底、黑色坐标轴、浅灰虚线网格、蓝色主图形。
- 标题和数值标注统一使用较粗字重，适合直接放到课程报告或答辩材料里。
- 方法图 `feature_infographic` 保留信息图表达，但建议后续如果继续扩图，也沿用相同的标题和文本层级。
