from src.env.battlezone_env import make_env
from src.agent.network import DQN

def main():
    env = make_env()
    obs, info = env.reset()
    print("Environment ready:", env.unwrapped.spec.id)
    # Training logic will go here later

if __name__ == "__main__":
    main()
