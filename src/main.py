import argparse
from train import train_dqn
from train import train_ppo
from train import train_rainbowDQN

def main():
    
    #parser = argparse.ArgumentParser(description="RL Training Launcher")
    #parser.add_argument("--algo", type=str, required=True,
   #                     choices=["dqn", "rainbow", "ppo"],
   #                     help="Algorithm to train: dqn | rainbow | ppo")
   # args = parser.parse_args()

   # if args.algo == "dqn":
    print("\n=== Training DQN ===\n")
    train_dqn()

    #elif args.algo == "rainbow":
    #    print("\n=== Training Rainbow DQN ===\n")
    #    train_rainbowDQN()

    #elif args.algo == "ppo":
    #    print("\n=== Training PPO ===\n")
    #    train_ppo()

if __name__ == "__main__":
    main()
