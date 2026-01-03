import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
'''
this file contains helper functions to create wrapped environments
for training and evaluation of RL agents on Atari games

it is the core function to create the atari enviroments with standard preprocessing


it has two main outputs:
either it will create the enviroment with rendering mode for visualization (deploy mode)
or it will create the enviroment without rendering mode for training (train mode)
'''
print("wrappers.py loaded")
#function to create atari env with standard preprocessing
def make_atari_env(
    env_id="ALE/BattleZone-v5",
    frame_stack=4,
    render=False
):
    """
    Creates an Atari environment with standard DQN/Rainbow/PPO preprocessing.

    Args:
        env_id (str): Atari environment id
        frame_stack (int): Number of stacked frames
        render (bool): If True, enable visualization (deploy mode)
    """

    render_mode = "human" if render else None

    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1
    )

    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        noop_max=30,
        terminal_on_life_loss=True
    )

    env = FrameStackObservation(env, frame_stack)

    return env

print("make_atari_env function defined")

