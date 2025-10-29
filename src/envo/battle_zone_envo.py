
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/BattleZone-v5', render_mode="human")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        
        observation, info = env.reset()
env.close()



#note i activatted it through cmd 
#
#first--> venv\Scripts\activate --it should givve --> (venv) C:\Users\USER\OneDrive\Desktop\PSUT\RL - Special Topic in DS (2)\MISC\code\rl_proj>
#
#then--> python src/envo/battle_zone_envo.py 