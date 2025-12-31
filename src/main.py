
import argparse
import os
import numpy as np
from train.train_rainbowDQN import train_rainbowDQN
from train.train_ppo import train_PPO
from train.train_dqn import train_dqn
import matplotlib.pyplot as plt

#to change the agent that is being trained, in launch.json change the arg --agent to either DQN, PPO or RAINBOW


def call_train(agent_type, args):
    '''
    Call the training function for the selected agent
    '''
    print(f"[INFO] Calling training for agent: {agent_type}")
    if agent_type == "DQN":#DQN
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
            device=args.device,
            mode=args.mode        # Pass mode for rendering / resume support
        )

    elif agent_type == "PPO":#PPO
        stats = train_PPO(
            env_id=args.env_id,
            num_episodes=args.num_episodes,
            rollout_length=getattr(args, "rollout_length", 2048),
            batch_size=args.batch_size,
            mini_batch_size=getattr(args, "mini_batch_size", 32),
            epochs=getattr(args, "epochs", 10),
            gamma=args.gamma,
            lam=getattr(args, "lam", 0.95),
            clip_epsilon=getattr(args, "clip_epsilon", 0.2),
            lr=args.lr,
            device=args.device,
            mode=args.mode
        )

    elif agent_type == "RAINBOW":#Rainbow DQN
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
            device=args.device,
            mode=args.mode
        )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return stats


def plot(stats, window=50):
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


def plot_from_file(agent_name, window=1000):
    file_path = f"{agent_name}_episode_rewards.npy"
    rewards = np.load(file_path)

    episodes = np.arange(len(rewards))
    rolling_mean = np.array([
        rewards[max(0, i - window): i + 1].mean()
        for i in range(len(rewards))
    ])

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(episodes, rolling_mean, linewidth=2,
             label=f"Rolling Mean ({window})")

    plt.axhline(
        y=rolling_mean[-1],
        linestyle="--",
        label="Final Avg"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent_name.upper()} Training Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{agent_name}_convergence.png", dpi=200)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train RL agents on Atari")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "resume", "deploy"],
        help="train: start new, resume: continue training, deploy: run trained agent"
    )

    parser.add_argument("--agent", type=str, default="DQN", # to change the agent that is being trained, in launch.json change the arg --agent to either DQN, PPO or RAINBOW
                        choices=["DQN", "PPO", "RAINBOW"])
    parser.add_argument("--env_id", type=str, default="ALE/BattleZone-v5")
    parser.add_argument("--device", type=str, default="cuda")
    #------------------------
    # change the default values below to set hyperparameters for each agent 
    #as presented in the comments
    
    parser.add_argument("--num_episodes", type=int, default=3000)#for DQN --> 10000  /// Rainbow DQN -->  /// PPO -->3000
    parser.add_argument("--replay_size", type=int, default=100_000)#for DQN --> 100_000  /// Rainbow DQN -->  
    parser.add_argument("--batch_size", type=int, default=32)#for DQN --> 32  /// Rainbow DQN -->  
    parser.add_argument("--start_learning", type=int, default=2_000)#for DQN --> 2000
    parser.add_argument("--train_freq", type=int, default=4)#for DQN --> 4  /// Rainbow DQN -->  
    parser.add_argument("--target_update_freq", type=int, default=500)#for DQN --> 500 
    parser.add_argument("--gamma", type=float, default=0.99)#for DQN -->  0.99 /// Rainbow DQN -->  /// PPO -->99
    parser.add_argument("--lr", type=float, default=2.5e-4 )#for DQN --> 2.5e-4   /// Rainbow DQN -->  /// PPO -->2.5e-4
    #------------------------
    # PPO-specific optional args
    parser.add_argument("--rollout_length", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)

    args = parser.parse_args()

    # Create results directory early
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    WEIGHT_FILE = os.path.join(RESULT_DIR, f"{args.agent.lower()}_weights.pth")
    HISTORY_FILE = os.path.join(RESULT_DIR, f"{args.agent.lower()}_history.npy")

    stats = None

    if args.mode in ["train", "resume"]:
        print(f"Starting {args.mode} for agent: {args.agent}")
        stats = call_train(args.agent, args)

        if stats is not None:
            print("Plotting training curves...")
            plot(stats)

    elif args.mode == "deploy":
        print(f"Deploying agent: {args.agent}")
        # call_train with mode=deploy will handle loading weights and running agent visually
        call_train(args.agent, args)

if __name__ == "__main__":
    #main()
    #plot_from_file("ppo")
    #plot_from_file("dqn")
    plot_from_file("rainbow")
