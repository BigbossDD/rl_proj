import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import RescaleAction


def make_atari_env(env_id="ALE/BattleZone-v5" , frame_stack=4):
    
    """
    it creates a  Atari environment  
    and it will prepare it for the agent 
    
    it will take -->
    env_id :  environment name.
    frame_stack : Number of frames to stack.
    
    returns --> gym.Env: Wrapped and ready-to-use environment.
    """
    
    # Create the environment
    env = gym.make(env_id, render_mode=None)
    
    # DeepMind preprocessing
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        noop_max=30,
        terminal_on_life_loss=True
    )
    
   
    env = RescaleAction(env, num_stack=frame_stack)

    return env