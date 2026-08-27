# Week 4 Retrospective

## Reinforcement Learning: What Changed Most in the Physical-Robot Context

The most important difference between the Aido Rover reinforcement-learning task and the earlier Breakout PPO project was reward design. In Breakout, the environment already defines what success means through the game score. The agent can therefore optimize a reward signal supplied by the game itself. In the Aido Rover environment, the reward function had to be designed from operational requirements.

That design decision affected nearly every part of the environment. `PATROL` should be rewarded during normal operation, but the same action should be penalized when a fault is active. `ALERT` or `INVESTIGATE` should become valuable when abnormal conditions appear, while unnecessary alerts should remain costly. `CHARGE` should be encouraged when battery state of charge is low but discouraged when it interrupts normal patrol without need. A naive reward function could easily create reward hacking, such as a policy that remains in `PATROL` to accumulate reward while ignoring faults.

The Week 2 results showed that the final reward structure was learnable: PPO reached a held-out mean reward of **188.09 ± 35.26**, compared with **13.12 ± 50.45** for the rule-based baseline and **−84.89 ± 17.19** for the random baseline. However, the larger lesson is that high reward only has meaning if the reward function accurately represents the intended robot behavior. In a real physical system, safety constraints, sensor noise, delayed observations, and incomplete state information would make this alignment problem even more important.

## Multimodal AI: Which Modality Contributed Most

The Week 3 ablation study showed that the **visual and text modalities tied as the strongest individual feature groups** on the synthetic dataset. Both achieved **F1 = 1.000** and **AUROC = 1.000**, while audio alone achieved **F1 = 0.875** and **AUROC = 0.984**. The full audio + visual + text fusion model also achieved **F1 = 1.000** and **AUROC = 1.000**.

The result does not mean that visual or text signals should automatically be considered more important for a real Fari deployment. The dataset contained only 80 synthetic interactions generated from explicit distress-versus-normal rules, so some modalities were individually sufficient to separate the classes. In realistic interaction, speech, facial behavior, and language can each become ambiguous or unreliable for different reasons.

The more useful conclusion is that multimodal systems should preserve evidence from several channels rather than depend on a single modality. If one channel becomes noisy or conflicts with the others, that disagreement can itself provide useful information about uncertainty.

## Cognitive Science: The Most Useful Concept

The cognitive-science concept that proved most useful was **attentional capacity / cognitive load** because it translated most directly into a concrete HRI design decision. If Fari detects possible distress or confusion, the user may already have reduced attentional resources. A response that is long, rapid, complicated, or demanding could therefore make the interaction more difficult.

This leads to a specific response strategy: use short sentences, slow and clear speech, a small number of clarification questions, and low-effort choices. The system should avoid presenting a classifier output as a diagnosis and should escalate only through a cautious human-supervision process when evidence remains strong.

Theory of mind also provided an important modeling principle. Audio, visual, and language signals are observations of behavior, not direct measurements of a person's internal mental state. A companion system should therefore treat its predictions as uncertain inferences and choose low-risk responses when evidence is incomplete.

## Career Narrative

The strongest outcome of this internship is the combination of three areas that are often presented separately: reinforcement learning, multimodal machine learning, and human-centered AI. The Aido Rover work demonstrates the full RL engineering loop through custom environment design, reward shaping, PPO training, TensorBoard monitoring, held-out evaluation, and comparison with baselines. The Fari work demonstrates multimodal feature extraction, feature fusion, classification, ablation analysis, and careful interpretation of synthetic-data limitations. The cognitive-science and HCI component adds a third layer by connecting model outputs to how a physical AI system should interact with people under uncertainty.

Together, these projects support a profile oriented toward ML engineering and physical AI rather than toward a single narrow model type. The technical work shows experience building and evaluating learning systems, while the HRI analysis shows awareness that deployed AI must also account for reward alignment, uncertainty, safety, and user experience. That combination is particularly relevant to robotics AI, human-robot interaction, multimodal systems, and applied machine-learning roles where model decisions directly affect users or physical environments.
