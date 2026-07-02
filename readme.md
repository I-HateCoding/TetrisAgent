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

训练会保存两个文件：`--output` 是最新 checkpoint，默认额外生成同名 `_best.pt`，例如 `dqn_afterstate_best.pt`。`_best.pt` 按固定 validation seeds 的 greedy 评估保存，更适合作为之后评估和继续训练的起点。`eval_p25` 比 `eval_avg` 更保守，能避免少数高分 seed 把 checkpoint 误判成 best。

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

如果训练中终端暂时没有输出，或者 `nvidia-smi` 显示 GPU 利用率很低，不一定代表训练卡住。环境模拟、候选棋盘生成、棋盘特征计算和 validation evaluation 大多在 CPU 上运行；同时日志只会按 `--step-log-every` 或 `--log-every` 间隔输出。只要还能继续出现 `Step ...`、`Episode ...` 或 `Eval episode=...`，训练就在正常进行。

也可以用下面命令检查最新 checkpoint 是否仍在更新：

```powershell
Get-Item .\models\dqn_afterstate_features_stable.pt
```

如果 `LastWriteTime` 最近更新过，说明训练至少已经完成过保存。若长时间没有日志、CPU/GPU 都接近 0%，再考虑是否真的卡住。

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

评估和可视化默认会使用 `auto` 设备；如果 PyTorch 能看到 CUDA，就会把 DQN 模型放到 GPU 上。即使只是运行模型，每一步也会对多个候选落子棋盘做网络前向计算，并且 PyTorch 会保留 CUDA 上下文和显存缓存。因此 `nvidia-smi` 里看到 `python.exe` 占用一部分显存是正常现象。

实际测试中，`human` 可视化模式强制使用 CPU 可能会无响应，因此可视化更建议继续使用默认 GPU。若训练进程同时运行，可视化会让训练变慢；此时可以减少可视化局数，或等训练保存 checkpoint 后再单独观看。

```powershell
python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate_features_best.pt --render-mode human --delay-ms 80
```

也可以使用 TorchScript 文件：

```powershell
python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate_scripted.pt --render-mode ansi --delay-ms 0
```

## 评估 DQN

```powershell
python tetris\evaluate_dqn.py --model .\models\dqn_afterstate_features_best.pt  --episodes 50 --seed 42 --render-mode ansi --delay-ms 0
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
