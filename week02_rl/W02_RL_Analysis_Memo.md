# Week 2 RL Analysis Memo — Aido Rover Decision Task

## 1. Reward Design Rationale

The Aido Rover environment uses a shaped reward function designed around operational outcomes rather than rewarding actions unconditionally. During normal operation, `PATROL` receives a small positive reward to encourage continued patrol behavior. When a fault is active, `ALERT` and `INVESTIGATE` receive larger positive rewards because the rover should respond to abnormal sensor conditions rather than continue normal patrol. Missed faults are penalized, unnecessary alerts or investigations are discouraged, and battery depletion receives a large negative penalty.

The most important reward-design decision was making the reward depend on both the current operational state and the selected action. A simpler design that rewarded `PATROL` at every timestep could lead to reward hacking: the agent could maximize cumulative reward by remaining in patrol even when sensor readings indicate a fault. The implemented reward therefore gives `PATROL` a positive reward only under normal conditions and penalizes it when a fault is active.

Battery management introduces another tradeoff. Charging is rewarded only when battery state of charge is low, while unnecessary charging is penalized. This prevents the policy from learning that remaining near the charging state is always safer than performing the patrol task. The PPO discount factor was explicitly set to γ = 0.99 so that future consequences such as delayed fault handling and battery depletion remain relevant over the 200-step episode horizon.

## 2. Training Behaviour and Evaluation

The PPO agent was trained with Stable Baselines3 for 200,000 timesteps. Training was seeded and logged using TensorBoard. The reward curve shows a clear learning trend: mean episode reward began substantially below zero, improved rapidly during the early stages of training, and reached approximately 150–180 reward by around 60,000–80,000 timesteps. After roughly 70,000–100,000 timesteps, performance stabilized near the 190–200 reward range. The final logged mean episode reward was approximately 195.

Held-out evaluation used 20 episodes that were not used during training or hyperparameter selection. PPO achieved a mean total reward of **188.09 ± 35.26**. The random baseline achieved **−84.89 ± 17.19**, while the rule-based baseline achieved **13.12 ± 50.45**. PPO therefore substantially outperformed both baselines in this synthetic environment.

The sample PPO rollout also showed behavior consistent with the intended reward design. During normal operation the policy primarily selected `PATROL`. When a fault appeared, continuing to patrol produced a negative reward, after which the policy selected `INVESTIGATE` and received a positive reward for responding to the fault. The sample held-out rollout completed the full 200-step episode with a total reward of **200.50**.

Compared with the earlier Breakout PPO project, the Aido Rover task converged much faster. This is expected because the rover environment uses a low-dimensional structured sensor vector and four high-level actions rather than high-dimensional game observations and more complex visual dynamics. The underlying PPO training loop remains similar, but the environment and reward function carry much more of the task-specific design responsibility.

## 3. InGen Deployment Implications

The results demonstrate that PPO can learn a meaningful high-level patrol policy in the simplified Aido Rover environment. However, the experiment should not be interpreted as evidence that the trained policy could directly control a real Aido Rover. The environment uses normalized synthetic sensor values and simplified transitions rather than actual robot telemetry.

A real deployment would introduce substantially more complexity. Sensor inputs such as motor current, IMU readings, proximity measurements, and RSSI would contain hardware noise, calibration error, missing values, latency, and changing outdoor conditions. The four-action discrete space also represents high-level operational decisions only. A real rover would likely require continuous or hybrid controls such as velocity, heading, steering, and motion planning.

The strongest conclusion from this experiment is therefore methodological rather than deployment-ready: the PPO training framework transfers successfully from the Breakout state-action-reward loop to a synthetic physical-AI decision environment, but successful real-world deployment would require more realistic sensor distributions, stronger safety constraints, richer action representations, and validation using real robot operating conditions.
