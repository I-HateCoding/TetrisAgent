环境配置：

```powershell
conda create -n tetris_env python=3.10
conda activate tetris_env
pip install gymnasium tetris_gymnasium numpy opencv-python torch
```

使用 DQN 智能体运行：

```powershell
conda run -n tetris_env python tetris\main.py --agent dqn --dqn-model .\models\dqn_afterstate.pt --render-mode ansi --delay-ms 0
```

将 `.\models\dqn_afterstate.pt` 替换为实际训练好的 DQN 模型路径。
