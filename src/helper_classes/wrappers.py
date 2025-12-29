'''
import ale_py
import gymnasium as gym
from gymnasium.envs.registration import registry
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
print("wrappers.py loaded")

def make_atari_env(env_id="BattleZone-v5", frame_stack=4):
    env = gym.make(env_id, render_mode=None, frameskip=1)

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
'''
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

print("wrappers.py loaded")

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

