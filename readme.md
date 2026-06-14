# TetrisAgent

本项目是一个围绕俄罗斯方块（Tetris）环境进行 AI 智能体设计与实验的课程项目。当前仓库仍处于早期阶段，现有代码只完成了对 `tetris_gymnasium` 环境的最小运行示例：创建环境、随机采样动作、渲染游戏画面，并在游戏结束后退出。

## 当前项目状态

目前已经包含的内容：

- 一个基于 `gymnasium` 和 `tetris_gymnasium` 的俄罗斯方块环境运行示例。
- 一个随机动作策略，用作后续实验的最基础 baseline。
- 项目展示材料 `TETRIS.pptx`。

目前尚未实现的内容：

- Expectimax 智能体。
- DQN 智能体。
- 训练脚本、评估脚本和实验结果统计。
- 自动化测试或完整的项目模块划分。

## 项目目标

本项目后续希望围绕同一个俄罗斯方块环境，逐步实现并比较不同 AI 策略的表现，包括：

- 随机策略：作为性能下限参考。
- 启发式或搜索策略：例如基于棋盘高度、空洞数量、平整度、消行数等特征进行决策。
- 强化学习策略：例如基于 DQN 的智能体训练与评估。

后续实验会尽量在统一的环境、状态定义、动作定义和评估指标下进行，便于比较不同方法在得分、存活步数、消行能力和决策效率上的差异。

## 环境要求

建议使用 Python 3.10。当前示例代码已经在 README 中按 Python 3.10.20 编写环境创建命令。

需要安装的主要依赖：

- `gymnasium`
- `tetris_gymnasium`
- `opencv-python`

其中，`opencv-python` 用于当前示例代码中的 `cv2.waitKey`。

## 环境配置

使用 conda 创建环境：

```bash
conda create -n tetris_env python=3.10.20
```

激活环境：

```bash
conda activate tetris_env
```

安装依赖：

```bash
pip install gymnasium tetris_gymnasium opencv-python
```

## 运行方式

在项目根目录下运行：

```bash
python ./tetris/main.py
```

运行后会打开俄罗斯方块窗口，程序会随机选择动作直到游戏结束。

## 目录结构

```text
TetrisAgent/
├── readme.md
├── TETRIS.pptx
└── tetris/
    └── main.py
```