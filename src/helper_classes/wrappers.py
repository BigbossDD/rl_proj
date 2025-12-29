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