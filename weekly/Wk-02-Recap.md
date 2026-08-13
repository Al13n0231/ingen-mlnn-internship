# Week 2 Recap — Reinforcement Learning for Aido Rover

The most important design decision this week was making the reward function depend on the rover's operational state rather than assigning a fixed reward to each action. In particular, `PATROL` receives a positive reward during normal operation but becomes negative when a simulated fault is active. This was necessary to avoid a simple reward-hacking failure mode in which the agent could maximize reward by continuously patrolling while ignoring abnormal sensor readings. The reward function also encourages appropriate fault responses and low-battery charging while penalizing unnecessary alerts, investigations, and battery depletion.

The resulting PPO policy behaved largely as expected. I trained the Stable Baselines3 PPO agent for 200,000 timesteps with a fixed training seed and TensorBoard logging. The training reward improved from strongly negative values early in training to approximately 190–200 per episode and became relatively stable after roughly 70,000–100,000 timesteps.

On 20 held-out evaluation episodes, PPO achieved **188.09 ± 35.26** mean total reward. The random baseline achieved **−84.89 ± 17.19**, while the rule-based baseline achieved **13.12 ± 50.45**. This indicates that the learned PPO policy substantially outperformed both baseline policies in the synthetic patrol environment.

The sample rollout also showed the intended connection between sensor conditions, actions, and reward. The agent generally selected `PATROL` during normal operation, but changed behavior when fault-related sensor patterns appeared. For example, continuing to patrol during a fault produced a negative reward, followed by an `INVESTIGATE` action that received a positive reward when the fault was addressed.

The main lesson from Week 2 is that transferring PPO from a game environment to a robot decision problem does not mainly change the learning algorithm. The harder design problem is defining observations, rewards, and termination conditions that represent useful operational behavior without creating unintended incentives.
