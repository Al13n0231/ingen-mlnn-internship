# W01 — RL / PIC 2.0 Bridge

## 1. PPO and GRPO: mathematical connection

Explain that PPO and GRPO are policy-gradient methods and compare their clipped policy-ratio objectives. Clearly distinguish PPO's learned value-function critic from GRPO's group-relative reward normalization.

> Scope note: InGen's internal GRPO implementation is not publicly disclosed. Any connection to PIC 2.0 should be presented as a structured hypothesis based on public information.

# W01 — RL / PIC 2.0 Bridge

## 1. PPO and GRPO: Mathematical Connection

Proximal Policy Optimization (PPO) and Group Relative Policy Optimization
(GRPO) are both on policy policy gradient methods. They directly update a
parameterized policy, $\pi_\theta(a \mid s)$, to increase the probability of
actions associated with higher estimated advantage while limiting excessively
large policy updates.

PPO commonly uses the clipped surrogate objective

$$
L^{PPO}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}
\left(
r_t(\theta), 1-\epsilon, 1+\epsilon
\right)\hat{A}_t
\right)
\right],
$$

where

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

is the probability ratio between the updated and previous policies, and
$\hat{A}_t$ estimates whether an action performed better or worse than
expected.

The main conceptual connection is that GRPO also uses a clipped policy-ratio
objective to control the size of policy updates. However, PPO usually estimates
advantages using a learned value-function critic, while GRPO estimates relative
advantages by comparing rewards across a group of sampled outputs. Therefore,
GRPO preserves the constrained policy-update principle of PPO while replacing
the learned critic with group-relative reward normalization.

InGen's internal GRPO implementation within PIC 2.0 is not publicly disclosed.
Therefore, the connection described here is a structured hypothesis based on
InGen's public description of PIC 2.0 and publicly available descriptions of
PPO and GRPO.

## 2. Breakout Environment vs. Physical Robot Environment

The Breakout PPO project and an InGen robot decision task share the same
state-action-reward structure, but the meaning and consequences of each
component differ substantially.

| Design Element | Breakout PPO Project | Physical Robot Context |
|---|---|---|
| Observation space | Pixel frames or game-state representations | Multi-sensor observations such as battery state of charge, motor current, proximity, IMU magnitude, and signal strength |
| Action space | Discrete game actions such as left, right, fire, and no-op | Discrete or hybrid operational actions such as PATROL, ALERT, CHARGE, and INVESTIGATE |
| Reward signal | Game score supplied by the environment | A shaped reward designed from operational goals and safety requirements |
| Transition dynamics | Deterministic or simulated game physics | Noisy, partially observable, and affected by hardware and environmental conditions |
| Episode termination | Loss of lives or completion of the game | Battery depletion, an unresolved fault, a safety event, or a maximum operating horizon |
| Consequence of failure | Loss of game reward | Possible hardware damage, missed hazards, or unsafe robot behavior |

The Breakout environment provides observations and rewards automatically. In
a physical robot context, the designer must decide which sensor readings form
the observation, how they are normalized, and how operational goals are
translated into a scalar reward. Real sensors may also contain noise, missing
values, or delayed measurements, making the robot environment less predictable
than the game environment.

## 3. Design Decisions That Must Be Reconsidered

### Reward shaping

Breakout provides a natural score-based reward, whereas the Aido Rover task
requires a reward function designed from first principles. A possible reward
could provide a small positive value for safe patrol behavior, a penalty for
false alerts, a larger penalty for failing to respond to an injected fault, and
a severe penalty for battery depletion without returning to charge.

However, reward shaping can produce unintended behavior. For example, if the
agent receives a positive reward mainly for remaining in PATROL, it may learn to
ignore abnormal sensor readings because continuous patrol accumulates more
reward. The reward must therefore represent successful operational outcomes
rather than simply rewarding the execution of a preferred action.

### Observation preprocessing

Breakout observations are commonly resized, normalized, and stacked across
multiple frames. Robot observations instead combine sensor channels with
different units and ranges. Battery percentage, motor current, proximity, and
IMU values must be normalized consistently. Temporal information may also be
needed because a single sensor reading may not distinguish normal variation
from a developing fault.

### Episode termination

In Breakout, termination occurs when the player loses all lives. In a robot
environment, termination conditions are part of the safety and reward design.
An episode may end when the battery reaches zero, a fault remains unresolved
for too long, or a maximum number of timesteps is reached. These conditions
must be designed carefully so that the policy cannot maximize reward by
intentionally ending episodes early.

### Physical deployment constraints

A simulated robot environment does not fully represent sensor noise, actuator
delay, communication loss, hardware wear, or safety constraints. A policy that
performs well in simulation would require additional testing, safety rules, and
human oversight before deployment on physical hardware.

## 4. Bridge to the Aido Rover Task

The Week 2 Aido Rover environment can preserve the implementation methodology
used in the Breakout PPO project: define an observation space, define a discrete
action space, design a reward function, train a PPO policy, log the training
process, and evaluate the learned policy on held-out episodes.

The domain transfer is therefore not a change in the core reinforcement
learning loop. It is a change in how the state, actions, rewards, termination
conditions, and safety constraints are defined. Breakout provides a controlled
game environment, while the Aido Rover task requires an operational model based
on robot sensor readings and patrol decisions. This makes reward design and
environment validity more important than simply obtaining a high training
reward.

