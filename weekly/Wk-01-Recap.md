# Week 01 Recap

During Week 1, I reviewed InGen Dynamics’ product portfolio and PIC 2.0
framework through reinforcement learning and multimodal AI perspectives. I
examined Fari, Senpai, Sentinel Prime AI, Aido Rover, and Aido Humanoid by
identifying the primary AI/ML task, the type of sensor input, the most important
constraint, and the most natural mapping to either reinforcement learning or
multimodal AI. I also completed the PPO/GRPO bridge, the multimodal bridge, and
the machine-learning environment check.

Among the InGen platforms, Aido Rover is the most analogous to the Breakout
reinforcement learning environment. Both can be represented through the same
state-action-reward loop. In Breakout, the agent receives a game observation,
selects a discrete action, and receives a scalar reward based on game
performance. In the proposed Aido Rover environment, the agent receives a
vector of sensor readings, selects an operational action such as PATROL, ALERT,
CHARGE, or INVESTIGATE, and receives a shaped reward based on patrol
performance, fault response, and battery management.

The main difference is that Breakout supplies its game mechanics and reward
signal automatically, while the Aido Rover environment requires these elements
to be designed. Observation selection, reward shaping, and episode termination
therefore become more important because a poorly designed environment may
produce unintended behavior.

Fari is the platform most analogous to the Multimodal Presentation Analysis
project. The previous project combined audio, facial-expression, and visual or
language-related information to evaluate presentation effectiveness. Fari
requires similar audio, visual, and language modalities to understand a user’s
interaction state. Voice characteristics, facial or body cues, and dialogue
content can be combined in a multimodal pipeline.

The prediction target changes from presentation effectiveness to companion
interaction state, and the real-time companion context introduces additional
requirements such as low latency, temporal alignment, missing modalities, and
safety-sensitive uncertainty. However, the basic feature-extraction and fusion
structure remains directly transferable.

Overall, Week 1 showed that the strongest reinforcement learning bridge is
between Breakout PPO and Aido Rover, while the strongest multimodal bridge is
between the presentation-analysis project and Fari.