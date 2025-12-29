import argparse
from train.train_rainbowDQN import train_rainbowDQN
#from src.train.train_ppo import train_ppo
from train.train_dqn import train_dqn
import matplotlib.pyplot as plt

def call_train(agent_type, args):
    '''
    Call the training function for the selected agent
    '''
    
    if agent_type == "DQN":
        stats = train_dqn(
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            start_learning=args.start_learning,
            train_freq=args.train_freq,
            target_update_freq=args.target_update_freq,
            gamma=args.gamma,
            lr=args.lr,
            device=args.device
        )

    #elif agent_type == "PPO":
     #   train_ppo(args)

    elif agent_type == "RAINBOW":
        stats = train_rainbowDQN(
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            start_learning=args.start_learning,
            train_freq=args.train_freq,
            target_update_freq=args.target_update_freq,
            gamma=args.gamma,
            lr=args.lr,
            device=args.device
        )

        
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return stats

def plot(stats , window=100):
    

    rewards = stats["episode_rewards"]
    episodes = range(len(rewards))

    # Rolling mean
    rolling_mean = [
        sum(rewards[max(0, i - window):i + 1]) / (i - max(0, i - window) + 1)
        for i in range(len(rewards))
    ]

    plt.figure(figsize=(12, 6))

    # Raw rewards (faded)
    plt.plot(
        episodes,
        rewards,
        color="gray",
        alpha=0.3,
        label="Episode Reward"
    )

    # Rolling average (main signal)
    plt.plot(
        episodes,
        rolling_mean,
        color="blue",
        linewidth=2,
        label=f"Rolling Mean ({window})"
    )

    # Final performance reference
    plt.axhline(
        y=sum(rewards[-window:]) / min(len(rewards), window),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Final Avg"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance (Convergence View)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on Atari")

    # ==== general ====
    parser.add_argument("--agent", type=str, default="RAINBOW",
                        choices=["DQN", "PPO", "RAINBOW"])
    parser.add_argument("--env_id", type=str, default="ALE/BattleZone-v5")
    parser.add_argument("--device", type=str, default="cuda")

    # ==== DQN hyperparameters ====
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--replay_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--start_learning", type=int, default=10_000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    print(f"Starting training for agent: {args.agent}")
    stats  = call_train(args.agent, args)
    print("Training finished.")
    
    
    if stats is not None:
        print("Plotting training curves...")
        plot(stats)

if __name__ == "__main__":
    main()
