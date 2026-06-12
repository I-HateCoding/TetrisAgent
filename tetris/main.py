import cv2
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris

# 创建俄罗斯方块环境
# - "tetris_gymnasium/Tetris": 环境名称
# - render_mode="human": 渲染模式
#   * "human": 显示游戏窗口（弹出可视化界面）
#   * "rgb_array": 返回图像数组（用于训练或录制）
#   * None: 不渲染（用于批量训练）
env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
# 重置环境到初始状态
# - seed: 设置随机种子，确保结果可重复
#   相同的种子会产生相同的游戏序列（方块顺序等）
env.reset(seed=42)

# 游戏状态标志
# - terminated: 游戏是否正常结束（如方块堆叠到顶部）
# - truncated: 游戏是否因外部条件提前结束（如超时、达到步数上限）
terminated = False
truncated = False

while not terminated:
    # 根据 render_mode 显示或生成当前游戏状态
    # 在 "human" 模式下，会弹出窗口显示俄罗斯方块游戏界面
    env.render()

    # 从动作空间中随机采样一个动作
    # action_space 定义了所有可能的动作
    action = env.action_space.sample()

    # step() 方法执行动作并返回五元组：
    #   - observation: 当前游戏状态（字典类型）
    #   - reward: 本步获得的奖励值
    #   - terminated: 游戏是否结束（如方块堆顶）
    #   - truncated: 游戏是否被截断（如达到最大步数）
    #   - info: 额外信息字典
    observation, reward, terminated, truncated, info = env.step(action)

    # cv2.waitKey(100) 等待 100 毫秒
    key= cv2.waitKey(100)

print("Game Over!")

env.close()