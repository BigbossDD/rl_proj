RL proj
==============================

atari battle zone project , test  on three algorithms , DQN , ranibowDQN , PPO

PSUT 
------------
src
│
├── envo                     <- Environment setup, API testing, and helper utilities.
│   ├── test_env.py          <- Confirms environment API (reset(), step(), spaces).
│   └── helper_classes
│       └── replay_buffer.py <- Replay buffer implementation for DQN-based agents.
│
├── models                   <- Agent classes for the reinforcement learning algorithms.
│   ├── dqn_agent.py         <- Baseline Deep Q-Network (DQN) agent skeleton.
│   ├── rainbowDQN_agent.py  <- Rainbow DQN agent skeleton (PER, N-step, etc.).
│   └── ppo_agent.py         <- Proximal Policy Optimization (PPO) agent skeleton.
│
├── train                    <- Training scripts for each RL algorithm.
│   ├── train_dqn.py         <- Training loop for the DQN baseline agent.
│   ├── train_rainbowDQN.py  <- Training loop for the Rainbow DQN agent.
│   └── train_ppo.py         <- Training loop for the PPO agent.
│
└── main.py                  <- Central launcher that selects which agent to train
                               (e.g., python main.py --algo dqn).



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
