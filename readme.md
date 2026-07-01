# TetrisAgent

这是一个基于 `tetris_gymnasium` 的俄罗斯方块智能体项目。当前包含两类智能体：

- `ExpectimaxAgent`：启发式搜索智能体，用棋盘高度、洞、凹凸度等特征打分。
- `DQNAgent`：after-state DQN 智能体，对每个候选落子后的棋盘打分，并选择分数最高的合法动作。

项目使用 `GroupedActionsObservations` 包装环境。包装后，每个 action 表示一种最终落子方案，observation 则是一批候选棋盘：`observation[action]` 就是执行该落子后的棋盘。

## 环境配置

```powershell
conda create -n tetris_env python=3.10
conda activate tetris_env
pip install gymnasium tetris_gymnasium numpy opencv-python torch
```

训练脚本默认使用 `--device auto`：检测到 CUDA 时会自动使用 GPU，否则使用 CPU。可以先检查 PyTorch 是否能看到 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 训练 DQN

DQN 训练脚本在 `tetris/train_dqn.py`。它训练的是 after-state value network：网络输入是一个“落子后的棋盘”，输出一个标量价值。

训练脚本默认使用 Double DQN target、增量 shaped reward 和混合探索。日志里的 `raw` 是环境原始 reward，`shaped` 是实际用于训练的 reward。

增量 shaped reward 只奖励/惩罚棋盘变好或变差，例如洞数增加会扣分、洞数减少会加分。这样比每一步都按绝对高度/洞数扣分更稳，不容易学成“早点死掉少扣分”。混合探索会在 epsilon 探索时优先采样启发式动作，减少纯随机探索带来的短命样本。

日志里的几个值含义不同：

- `raw`：环境原始 reward 的累计值；在当前环境里它就是主要成绩指标。
- `shaped`：DQN 实际训练用的奖励信号，不等于游戏分数。

```powershell
python -u tetris\train_dqn.py --episodes 500 --output .\models\dqn_afterstate.pt
```

如果想同时导出 TorchScript 推理模型：

```powershell
python -u tetris\train_dqn.py --episodes 500 --output .\models\dqn_afterstate.pt --export-torchscript .\models\dqn_afterstate_scripted.pt
```

常用训练参数：

```powershell
python -u tetris\train_dqn.py `
  --episodes 1000 `
  --batch-size 64 `
  --buffer-size 50000 `
  --learning-starts 1000 `
  --target-update 1000 `
  --epsilon-start 1.0 `
  --epsilon-end 0.05 `
  --epsilon-decay-steps 50000 `
  --reward-mode shaped `
  --exploration-policy mixed `
  --heuristic-exploration-prob 0.75 `
  --hole-penalty 0.8 `
  --height-penalty 0.03 `
  --max-height-penalty 0.10 `
  --bumpiness-penalty 0.05 `
  --terminal-penalty 50.0 `
  --output .\models\dqn_afterstate.pt
```

如果希望训练时确认进度，增加实时日志：

```powershell
python -u tetris\train_dqn.py `
  --episodes 500 `
  --log-every 1 `
  --step-log-every 500 `
  --save-every 10 `
  --eval-every 50 `
  --eval-episodes 20 `
  --best-metric eval_p25 `
  --output .\models\dqn_afterstate.pt
```

训练会保存两个文件：`--output` 是最新 checkpoint，默认额外生成同名 `_best.pt`，例如 `dqn_afterstate_best.pt`。`_best.pt` 按固定 validation seeds 的 greedy 评估保存，更适合作为之后评估和继续训练的起点。`eval_p25` 比 `eval_avg` 更保守，能避免某几个高分 seed 把 checkpoint 误判成 best。

如果发现纯 CNN DQN 走偏，比如经常留下明显空洞，可以启用棋盘特征增强版。它会把高度、洞数、凹凸度、井深等归一化特征和棋盘图一起输入网络，比只看棋盘图更不容易学歪：

```powershell
python -u tetris\train_dqn.py `
  --use-board-features `
  --resume .\models\dqn_afterstate_finetuned.pt `
  --episodes 1000 `
  --lr 0.00001 `
  --epsilon-start 0.03 `
  --epsilon-end 0.005 `
  --epsilon-decay-steps 50000 `
  --exploration-policy mixed `
  --heuristic-exploration-prob 0.7 `
  --hole-penalty 1.5 `
  --hole-reduction-bonus 1.0 `
  --terminal-penalty 100 `
  --log-every 1 `
  --step-log-every 500 `
  --save-every 1 `
  --eval-every 50 `
  --eval-episodes 20 `
  --best-metric eval_p25 `
  --output .\models\dqn_afterstate_features.pt
```

这个命令可以从旧模型迁移：能对上的 CNN 权重会加载，新增的棋盘特征输入层会重新初始化。训练结束后优先评估 `dqn_afterstate_features_best.pt`。

如果没有先 `conda activate tetris_env`，可以用 `conda run --no-capture-output` 避免输出被缓存：

```powershell
conda run --no-capture-output -n tetris_env python -u tetris\train_dqn.py --episodes 500 --output .\models\dqn_afterstate.pt
```

如果想和纯环境 reward 对照，可以关闭 reward shaping：

```powershell
python -u tetris\train_dqn.py --reward-mode raw --episodes 500 --output .\models\dqn_raw.pt
```

如果已有模型效果不错，建议用低学习率、低探索率继续微调，而不是重新训练：

```powershell
python -u tetris\train_dqn.py `
  --resume .\models\dqn_afterstate_best.pt `
  --episodes 2000 `
  --lr 0.00001 `
  --epsilon-start 0.03 `
  --epsilon-end 0.005 `
  --epsilon-decay-steps 50000 `
  --exploration-policy mixed `
  --heuristic-exploration-prob 0.7 `
  --hole-penalty 1.5 `
  --hole-reduction-bonus 1.0 `
  --terminal-penalty 100 `
  --log-every 1 `
  --step-log-every 500 `
  --save-every 1 `
  --eval-every 50 `
  --eval-episodes 20 `
  --best-metric eval_p25 `
  --output .\models\dqn_afterstate_finetuned.pt
```

如果想复现实验中的旧版绝对特征惩罚，可以使用：

```powershell
python -u tetris\train_dqn.py --reward-mode absolute --episodes 500 --output .\models\dqn_absolute.pt
```

## 使用 DQN 智能体运行

```powershell
python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate_best.pt --render-mode ansi --delay-ms 0
```

可视化观看一局：

```powershell
python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate_features_best.pt --render-mode human --delay-ms 80
```

也可以使用 TorchScript 文件：

```powershell
python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate_scripted.pt --render-mode ansi --delay-ms 0
```

## 评估 DQN

```powershell
python tetris\evaluate_dqn.py --model .\models\dqn_afterstate_best.pt --episodes 50 --seed 42 --render-mode ansi --delay-ms 0
```

边渲染边评估时可以减少局数：

```powershell
python tetris\evaluate_dqn.py --model .\models\dqn_afterstate_finetuned.pt --episodes 3 --render-mode human --delay-ms 80
```

## 运行 Expectimax 基线

```powershell
python tetris\main.py --agent expectimax --depth 1 --render-mode ansi --delay-ms 0
```

多局评估：

```powershell
python tetris\evaluate_expectimax.py --episodes 20 --depth 1 --render-mode ansi --delay-ms 0
```

## 文件说明

- `tetris/main.py`：创建环境，运行单局游戏，支持 `expectimax` 和 `dqn`。
- `tetris/dqn_model.py`：DQN 使用的小型 CNN 模型。
- `tetris/train_dqn.py`：after-state DQN 训练脚本。
- `tetris/dqn_agent.py`：加载 DQN 模型并执行动作选择。
- `tetris/evaluate_dqn.py`：DQN 多局评估脚本。
- `tetris/expectimax_agent.py`：启发式 Expectimax 智能体。
- `tetris/evaluate_expectimax.py`：Expectimax 多局评估脚本。

## 说明

当前 DQN 是一个基础版本，重点是把完整训练、保存、加载、评估链路打通。后续可以继续补充更强的网络结构、更细的 reward shaping、训练曲线记录、断点续训和更系统的超参数搜索。
