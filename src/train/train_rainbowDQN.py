import numpy as np
import torch
import gymnasium as gym
import cv2
import os
from collections import deque, defaultdict
from models.rainbowDQN_agent import RainbowDQNAgent

# ======================================================
# Frame preprocessing
# ======================================================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


# ======================================================
# Training / Deployment Entry
# ======================================================
def train_rainbowDQN(
    env_id,
    num_episodes,
    replay_size,
    batch_size,
    start_learning,
    train_freq,
    target_update_freq,
    gamma,
    lr,
    device,
    mode="train"
):
    """
    mode:
        - 'train'  : training without rendering
        - 'deploy' : play using trained weights (rendered)
    """

    render_mode = "human" if mode == "deploy" else None
    env = gym.make(env_id, render_mode=render_mode)

    n_actions = env.action_space.n

    # ---- Agent ----
    

    agent = RainbowDQNAgent(
        obs_shape=(4, 84, 84),
        n_actions=n_actions,
        gamma=gamma,
        lr=lr,
        device=device,
        replay_size=replay_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )

    WEIGHT_FILE = "rainbow_weights.pt"
    REWARD_FILE = "rainbow_rewards.npy"

    frame_stack = deque(maxlen=4)

    # ======================================================
    # DEPLOY MODE
    # ======================================================
    if mode == "deploy":
        if not os.path.exists(WEIGHT_FILE):
            print("[ERROR] No trained weights found.")
            return None

        agent.q_net.load_state_dict(
            torch.load(WEIGHT_FILE, map_location=device)
        )
        agent.q_net.eval()

        print("[INFO] Loaded trained Rainbow DQN")

        for ep in range(5):
            obs, _ = env.reset()
            frame = preprocess_frame(obs)
            frame_stack.clear()
            for _ in range(4):
                frame_stack.append(frame)

            state = np.stack(frame_stack, axis=0)
            done = False
            ep_reward = 0

            while not done:
                action = agent.select_action(state)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                frame_stack.append(preprocess_frame(obs))
                state = np.stack(frame_stack, axis=0)

                ep_reward += reward

            print(f"[DEPLOY] Episode {ep+1} Reward: {ep_reward:.1f}")

        env.close()
        return None

    # ======================================================
    # TRAIN MODE
    # ======================================================
    stats = defaultdict(list)
    global_step = 0

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            frame = preprocess_frame(obs)
            frame_stack.clear()
            for _ in range(4):
                frame_stack.append(frame)

            state = np.stack(frame_stack, axis=0)
            done = False
            ep_reward = 0

            while not done:
                action = agent.select_action(state)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                frame_stack.append(preprocess_frame(obs))
                next_state = np.stack(frame_stack, axis=0)

                agent.store(state, action, reward, next_state, done)

                if global_step > start_learning and global_step % train_freq == 0:
                    loss = agent.learn()
                    if loss is not None:
                        stats["loss"].append(loss)

                if global_step % target_update_freq == 0:
                    agent.update_target()

                state = next_state
                ep_reward += reward
                global_step += 1

            stats["episode_rewards"].append(ep_reward)
            print(
                f"Episode {ep+1}/{num_episodes} | "
                f"Reward: {ep_reward:.1f}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted")

    finally:
        torch.save(agent.q_net.state_dict(), WEIGHT_FILE)
        np.save(REWARD_FILE, np.array(stats["episode_rewards"]))
        env.close()

        print(f"[INFO] Weights saved to {WEIGHT_FILE}")
        print(f"[INFO] Rewards saved to {REWARD_FILE}")

    return stats
