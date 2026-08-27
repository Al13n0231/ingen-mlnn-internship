# InGen Dynamics ML & Neural Network Internship Capstone

## 1. Executive Summary

This four-week internship explored how reinforcement learning, multimodal machine learning, and cognitive-science-informed human-robot interaction can be combined within a physical-AI engineering workflow. The work was anchored to two InGen Dynamics platforms: Aido Rover for reinforcement learning and Fari for multimodal companion AI. Week 1 established the product and PIC 2.0 context, Week 2 implemented and evaluated a custom PPO-based rover decision environment, Week 3 implemented and evaluated a three-modality distress-classification pipeline, and Week 4 synthesized the technical findings into a portfolio-level physical-AI analysis.

The reinforcement-learning track produced a custom Gymnasium environment for a simplified Aido Rover patrol task with a structured sensor observation space and four high-level actions: `PATROL`, `ALERT`, `CHARGE`, and `INVESTIGATE`. A Stable Baselines3 PPO agent was trained for 200,000 timesteps with a discount factor of γ = 0.99. Training converged to a final logged mean episode reward of approximately 195. On 20 held-out evaluation episodes, PPO achieved **188.09 ± 35.26** mean total reward, compared with **13.12 ± 50.45** for the rule-based baseline and **−84.89 ± 17.19** for the random baseline. The central RL finding is therefore not only that PPO learned a strong policy in the synthetic environment, but that physical-robot RL places unusually high importance on reward alignment, state representation, safety, and environment validity.

The multimodal track implemented a synthetic Fari companion-state classifier using **26 MFCC audio features**, **six facial-geometry features**, and **384-dimensional MiniLM sentence embeddings**. Early concatenation produced a **416-dimensional** representation that was classified with logistic regression. On a held-out 16-example test set, the fused model achieved **precision 1.000, recall 1.000, F1 1.000, and AUROC 1.000**. Ablation analysis showed that **visual and text features tied as the strongest individual modalities**, each reaching F1 1.000 and AUROC 1.000, while audio alone reached F1 0.875 and AUROC 0.984. Because the dataset contained only 80 rule-generated synthetic examples, these perfect scores should be interpreted as evidence that the pipeline is internally coherent and highly separable under the synthetic generation rules, not as evidence of real-world elder-distress detection performance.

Across the product portfolio, the strongest recommendations are: Fari should prioritize uncertainty-aware multimodal state estimation and cautious response policies rather than treating classifier outputs as diagnoses; Senpai should reuse the multimodal architecture but optimize for learning-state evidence instead of superficial engagement; Sentinel Prime AI should emphasize uncertainty calibration, temporal confirmation, and safety gating before operational action; Aido Rover should keep high-level learned decision policies separated from hard safety constraints and lower-level motion control; and Aido Humanoid should be treated as a longer-horizon continuous-control and sim-to-real research problem rather than a direct extension of the four-action rover environment.

The cross-track conclusion is that physical AI is best understood as a pipeline rather than a single model: **sensor observations → perception and feature extraction → state understanding → decision policy → safety validation → physical or interactive action**. The Week 3 multimodal work focused on the perception and state-understanding side of this pipeline, while the Week 2 PPO work focused on decision policy. Cognitive science and HCI provide the interpretation layer connecting model outputs to human-centered behavior. In particular, attentional capacity, theory of mind, and the semantic-versus-pragmatic distinction in language all support a design in which uncertain human-state estimates lead to low-risk, cognitively appropriate responses rather than overconfident automation.

---

## 2. InGen Physical AI Context and Technical Bridges

### 2.1 Physical AI and PIC 2.0

InGen Dynamics' product portfolio can be viewed through the common problem of physical intelligence: converting observations from the physical world into safe and useful actions. Unlike software-only AI systems, physical AI must operate with noisy sensor inputs, real-time constraints, changing environments, and actions that may have direct physical or human consequences. Although Fari, Senpai, Sentinel Prime AI, Aido Rover, and Aido Humanoid serve different users and environments, each platform follows the same general machine-learning pipeline: sensor observations are transformed into perceptual features and state representations, which then support a decision or response that must be validated against safety constraints before execution.

The internship materials describe PIC 2.0 as a conceptual physical-intelligence architecture containing several specialized components, including GRPO for policy learning, STUM for state and temporal understanding, SEOM for semantic or embedding-based reasoning, AMDC for adaptive decision-making, HTD-IRL for hierarchical task reasoning and inverse reinforcement learning, and CRL-MRS for continual and multi-robot learning. Because the internal implementation of PIC 2.0 is not publicly disclosed, these components are treated in this project as architectural research anchors rather than claims about InGen's production system.

This architecture provides a useful framework for connecting the two technical tracks completed during the internship. The reinforcement-learning track focuses on how a robot selects actions from an estimated operational state, while the multimodal track focuses on how that state can be inferred from heterogeneous sensory information. Together, they represent two adjacent stages of the physical-AI pipeline: **perception and state understanding followed by decision and action**.

### 2.2 Reinforcement Learning Bridge: PPO to Physical Robot Decision-Making

The reinforcement-learning track extends the implementation methodology of a prior PPO Breakout project into a simplified Aido Rover patrol environment. Both problems can be represented using the same fundamental reinforcement-learning loop: an agent receives an observation, selects an action using a parameterized policy, receives a scalar reward, and uses experience to improve future decisions. The major change is therefore not the underlying RL formulation, but the meaning and consequences of the environment design.

PPO and GRPO provide the conceptual connection between the previous project and the PIC 2.0 context. Both belong to the policy-gradient family and constrain policy updates using probability-ratio objectives. PPO commonly estimates the advantage of an action using a learned value-function critic, while publicly described GRPO methods instead estimate relative advantage by comparing rewards among a group of sampled outputs. Thus, the connection used in this internship is not that PPO and InGen's GRPO implementation are identical, but that both illustrate constrained policy optimization in which higher-reward behavior becomes more probable while excessively large policy changes are limited.

The transition from Breakout to the Aido Rover also changes the environment-design problem substantially. Breakout supplies its game state, action rules, score, transition dynamics, and termination conditions. A robot environment requires these elements to be designed from operational requirements. The Week 2 Aido Rover environment therefore represents observations using synthetic robot sensor values such as battery state of charge, motor current, IMU magnitude, proximity, and communication strength. The action space represents operational decisions such as `PATROL`, `ALERT`, `CHARGE`, and `INVESTIGATE`. Most importantly, the reward function must encode the desired behavior rather than relying on a score supplied by the environment.

This distinction makes reward design and environment validity more important than maximizing training reward alone. A poorly designed reward can produce a numerically successful policy that behaves incorrectly, such as remaining in `PATROL` to accumulate positive reward while ignoring an anomaly. Physical deployment would introduce additional challenges including sensor noise, delayed observations, partial observability, actuator dynamics, communication failures, and safety constraints. The Week 2 environment therefore serves as a controlled abstraction for studying robot decision-policy design rather than as a direct simulation of the complete Aido Rover control system.

### 2.3 Multimodal Bridge: Presentation Analysis to Companion AI

The second technical bridge adapts the architecture of a previous Multimodal Presentation Analysis project to Fari's companion-robot context. The original project combined audio, visual, and content-related information to estimate presentation effectiveness. Fari requires the same general modality structure, but with different inputs and a different prediction target: voice characteristics, facial or body cues, and conversational language are combined to estimate an interaction state such as normal behavior or possible distress.

The audio component transfers most directly. Speech characteristics can be represented using features such as MFCC-based summaries that capture properties of vocal delivery. Visual analysis can similarly use facial or landmark-based features. The most significant change occurs in the third modality: presentation slide content is replaced by an embedding of the user's conversational language. Consequently, the high-level fusion architecture remains similar while the semantics of both the features and the classification target change.

A companion robot also introduces temporal and operational requirements that do not exist in an offline presentation-analysis setting. Fari would receive continuous audio and visual streams while language becomes available after speech recognition. These modalities therefore need to be aligned over short temporal windows, and individual channels may be temporarily unreliable because of background noise, lighting conditions, occlusion, or transcription errors. Disagreement between modalities is also potentially meaningful rather than simply erroneous. For example, a verbal statement indicating that everything is fine may conflict with vocal or facial cues associated with distress.

This observation creates the direct connection to cognitive science and human-robot interaction. Observable behavior should be treated as evidence about an underlying human state rather than as the state itself. A multimodal system therefore supports a more cautious form of state estimation than a text-only classifier, particularly when signals disagree or confidence is low. The Week 3 pipeline operationalizes this bridge by extracting audio, visual, and language features, fusing them into a common representation, and evaluating whether that combination can distinguish synthetic distress and normal-interaction examples.

---

## 3. Reinforcement Learning for the Aido Rover

### 3.1 Environment Design

The reinforcement-learning track modeled a simplified Aido Rover patrol task as a custom Gymnasium environment. The purpose of the environment was not to reproduce the rover's low-level physical dynamics, but to study a higher-level operational decision problem in which a robot must choose an appropriate behavior from its current sensor state.

The observation space consists of a structured vector of synthetic robot sensor readings representing quantities such as battery state of charge, motor current, IMU magnitude, proximity, and communication signal strength. This differs substantially from the high-dimensional visual observations used in the earlier Breakout PPO project. The lower-dimensional state representation makes the learning problem simpler while retaining the core state-action-reward structure required for reinforcement learning.

The action space contains four discrete operational decisions:

- `PATROL` — continue normal patrol behavior
- `ALERT` — stop normal operation and signal an abnormal condition
- `CHARGE` — respond to low battery state
- `INVESTIGATE` — respond directly to a detected anomaly

This action abstraction intentionally separates high-level decision-making from low-level navigation. A real rover would require additional continuous or hybrid controls for velocity, steering, heading, and motion planning, but those controls are outside the scope of this experiment.

Episodes are limited to a maximum horizon of 200 timesteps. Within an episode, the synthetic state evolves as battery energy is consumed and abnormal operating conditions may appear. The environment therefore provides enough temporal structure for the policy to learn that actions should depend on the current operational state rather than simply selecting the same action at every timestep.

### 3.2 Reward Function Design

Reward design was the most important environment-design decision in the RL track because, unlike Breakout, the Aido Rover task does not provide a naturally defined game score. The desired operational behavior therefore had to be translated into a scalar reward signal.

During normal operation, `PATROL` receives a small positive reward because continuing the patrol task is the desired default behavior. When a fault is active, however, the preferred behavior changes. `ALERT` and `INVESTIGATE` receive larger positive rewards for responding to abnormal conditions, while continuing to patrol during a fault is penalized. Missed faults, unnecessary alerts or investigations, and complete battery depletion are also penalized.

Battery management follows the same state-dependent principle. `CHARGE` is beneficial when battery state of charge is low, but unnecessary charging is penalized. This prevents the policy from learning that remaining in or repeatedly selecting a charging behavior is a risk-free way to accumulate reward while avoiding the actual patrol task.

This state-dependent reward structure was specifically designed to reduce reward hacking. For example, a naive reward function that assigned a positive value to every `PATROL` action could produce a policy that remained in patrol even when its sensors indicated a fault. Such a policy could obtain a high numerical return while failing at the actual operational objective. By conditioning the reward on both state and action, the implemented design attempts to align cumulative return more closely with the intended robot behavior.

The PPO discount factor was set to **γ = 0.99**. With a 200-step episode horizon, this allows future consequences such as delayed fault handling or eventual battery depletion to continue influencing earlier decisions instead of making the agent almost exclusively optimize immediate reward.

### 3.3 PPO Training Methodology

The policy was trained using Stable Baselines3's implementation of Proximal Policy Optimization for **200,000 timesteps**. Training was seeded for reproducibility and monitored using TensorBoard. The same general implementation discipline used in the previous Breakout project was retained: environment interaction generated experience, PPO updated the parameterized policy, training metrics were logged throughout the run, and final performance was evaluated separately on held-out episodes.

The primary difference from the Breakout experiment was therefore the environment rather than the learning algorithm. Breakout required the policy to extract useful information from a complex game observation, whereas the Aido Rover policy received a compact structured sensor vector. At the same time, the rover experiment placed more responsibility on the environment designer because the reward signal, operational state representation, and termination logic were defined specifically for the synthetic physical-AI task.

### 3.4 Training Behavior

The TensorBoard reward curve showed a clear learning trend during the 200,000-timestep training run. Mean episode reward began substantially below zero, indicating that the initial policy frequently selected actions that were poorly matched to the simulated operating state. Performance then improved rapidly during the early stages of training.

By approximately **60,000–80,000 timesteps**, mean episode reward had reached roughly **150–180**. Performance subsequently stabilized near the **190–200 reward range** after approximately **70,000–100,000 timesteps**, with a final logged mean episode reward of approximately **195**.

The relatively rapid convergence is consistent with the simplified structure of the environment. Unlike Breakout, where a policy must learn from high-dimensional game observations and more complicated visual dynamics, the Aido Rover environment provides a small sensor vector and only four high-level actions. The policy therefore has a much smaller state-action mapping to learn.

This comparison also illustrates an important physical-AI engineering distinction. In the Breakout project, much of the task complexity came from perception and game dynamics. In the Aido Rover experiment, more of the difficulty was moved into the design of the environment itself: deciding which sensor states should be represented, which actions should be available, how abnormal events should affect transitions, and which outcomes should receive positive or negative reward.

### 3.5 Held-Out Evaluation and Baseline Comparison

Final evaluation was performed on **20 held-out episodes** that were not used during training or hyperparameter selection. The trained PPO policy was compared against both a random policy and a deterministic rule-based baseline.

| Policy | Mean Episode Reward | Standard Deviation |
| --- | ---: | ---: |
| PPO | **188.09** | **35.26** |
| Rule-based | **13.12** | **50.45** |
| Random | **-84.89** | **17.19** |

The PPO policy substantially outperformed both comparison policies in the synthetic environment. Its mean held-out return was approximately **175 points higher than the rule-based baseline** and approximately **273 points higher than the random baseline**.

The random-policy result establishes that successful performance cannot be achieved by arbitrary action selection. More importantly, the comparison with the rule-based policy shows that the learned policy captured behavior that was more effective under the simulated dynamics than the simple hand-coded threshold strategy used as the baseline.

A sample held-out PPO rollout also demonstrated behavior consistent with the intended reward design. During normal conditions, the policy primarily selected `PATROL`. When a fault appeared, continuing to patrol produced a negative reward, after which the policy selected `INVESTIGATE` and received a positive reward for responding to the abnormal state. This rollout completed the full **200-step episode with a total reward of 200.50**.

These results should nevertheless be interpreted within the synthetic environment rather than as evidence of real-robot performance. Because both the learned policy and the baselines were evaluated under the same designed reward and transition system, the comparison demonstrates successful learning relative to that environment. It does not establish that the reward function perfectly represents all operational objectives of a physical Aido Rover.

### 3.6 Deployment Implications

The experiment demonstrates that PPO can learn a meaningful high-level patrol policy within the simplified Aido Rover environment, but the trained model is not intended to directly control a physical rover. The experiment uses normalized synthetic observations and simplified transition dynamics rather than telemetry collected from actual robotic hardware.

Real deployment would introduce several additional sources of uncertainty. Motor-current, IMU, proximity, battery, and wireless-signal measurements would contain sensor noise, calibration differences, missing observations, latency, and environmental variation. Outdoor operation would also introduce changing terrain, weather, lighting, obstacles, and communication conditions that are not represented in the synthetic environment.

The action representation would also need to become substantially richer. `PATROL`, `ALERT`, `CHARGE`, and `INVESTIGATE` represent useful high-level operational modes, but physical control ultimately requires lower-level decisions involving velocity, heading, steering, obstacle avoidance, localization, and trajectory planning. This suggests that a practical architecture may separate high-level policy selection from lower-level motion control rather than asking a single PPO policy to directly produce every actuator command.

Safety is another major difference between simulated and physical deployment. An incorrect action in the synthetic environment only decreases numerical reward, while an incorrect physical action could damage hardware or create risk for nearby people. A real system would therefore require hard safety constraints, fallback rules, uncertainty handling, and extensive validation in simulation and controlled physical environments before reinforcement-learning decisions were permitted to influence real robot behavior.

The main conclusion from the RL track is therefore methodological rather than deployment-ready. The PPO training framework successfully transferred from the Breakout state-action-reward loop to a synthetic physical-AI decision task, and the trained policy substantially outperformed both random and rule-based baselines under held-out evaluation. At the same time, the experiment shows that applying reinforcement learning to robotics shifts much of the engineering challenge toward **state representation, reward alignment, safety, and environment validity**. These design choices become at least as important as the optimization algorithm itself.

---

## 4. Multimodal AI for Fari Companion Intelligence

### 4.1 Three-Modality Pipeline

The Week 3 technical track adapted the earlier Multimodal Presentation Analysis workflow into a proof-of-concept companion-state classifier for Fari. The prediction task was binary: distinguish synthetic examples representing possible distress from examples representing normal interaction. The purpose was not to claim clinically valid distress recognition, but to test whether three heterogeneous input modalities could be extracted, aligned, fused, and evaluated within a reproducible companion-AI pipeline.

The pipeline used three distinct feature families. Audio was represented with **26 MFCC summary features**, formed by summarizing 13 MFCC coefficients across the synthetic speech waveform. Visual information was represented with **six facial-geometry features** intended to approximate expression-relevant spatial relationships. Dialogue content was represented with **384-dimensional sentence embeddings** from the `all-MiniLM-L6-v2` sentence-transformer model. Concatenating these vectors produced a **416-dimensional** feature representation for each interaction.

A logistic regression classifier was trained on the fused representation. Early concatenation was selected because the internship task emphasized a transparent baseline fusion architecture rather than a high-capacity neural fusion model. This choice keeps the role of the individual feature groups easier to inspect and makes ablation results straightforward to interpret.

The dataset contained **80 synthetic interactions** generated according to explicit distress-versus-normal class rules, with a held-out test set of **16 examples**. The audio waveforms, facial measurements, and dialogue examples were synthetic. This design allowed the full preprocessing and evaluation pipeline to run without using private or real eldercare data, but it also strongly limits how the resulting metrics should be interpreted.

### 4.2 Held-Out Classification Results

The fused classifier achieved the following held-out metrics:

| Metric | Fused Model |
| --- | ---: |
| Precision | **1.000** |
| Recall | **1.000** |
| F1 | **1.000** |
| AUROC | **1.000** |

On the synthetic test set, the classifier therefore separated the two classes perfectly. From an implementation perspective, this demonstrates that the three feature channels can be transformed into a single fixed-length representation and used successfully by a downstream classifier.

However, perfect synthetic performance is not the main result. The class-generation process intentionally created separable distress and normal distributions, so the test set shares the same synthetic assumptions as the training data. The observed performance therefore says more about internal pipeline consistency and feature separability than about performance in real human interaction. A production companion system would face much more ambiguous expressions, noisy speech, incomplete observations, changing individual behavior, and examples in which the same visible or linguistic signal can correspond to different underlying states.

The precision-recall tradeoff is especially important in a companion setting. A false positive could cause Fari to interrupt, escalate, or create unnecessary concern during a normal interaction. A false negative could allow a genuine distress signal to pass without an appropriate response. Real deployment would therefore require threshold calibration based on the relative cost of these errors, rather than selecting a model solely because it maximizes accuracy on a benchmark.

### 4.3 Ablation and Modal Contribution

Ablation experiments were used to determine how much predictive signal was available from each feature family when used independently.

| Feature Set | F1 | AUROC |
| --- | ---: | ---: |
| Audio only | **0.875** | **0.984** |
| Visual only | **1.000** | **1.000** |
| Text only | **1.000** | **1.000** |
| Audio + Visual + Text | **1.000** | **1.000** |

The strongest individual modalities were therefore **visual and text, tied at F1 1.000 and AUROC 1.000**. Audio alone remained strong but was weaker, with F1 0.875 and AUROC 0.984. The fused model matched the strongest single modalities rather than improving beyond them.

This result should not be interpreted as evidence that visual and text cues are inherently more important than audio for eldercare interaction. The synthetic label rules determine how class information is distributed among modalities, and the perfect visual and text scores indicate that those generated features were individually sufficient to separate the two synthetic classes. In real interaction, individual signals may be ambiguous or contradictory. A person may verbally minimize a problem while displaying vocal or facial changes, or background conditions may make one sensing channel unreliable.

The most useful conclusion from the ablation analysis is therefore architectural rather than hierarchical. A companion robot should not assume that one modality will always dominate. Instead, the system should be able to combine evidence, estimate reliability, and respond appropriately when channels disagree.

### 4.4 Why Fusion Still Matters

At first glance, fusion appears unnecessary in this experiment because visual-only and text-only models already achieved perfect synthetic test performance. That would be the wrong deployment conclusion. The result is an artifact of a small, rule-generated dataset in which some modalities are highly separable on their own.

In a realistic setting, the value of multimodal fusion comes from redundancy and complementarity. Audio describes how something is said, text describes what is said, and visual cues provide an independent behavioral channel. Each signal can fail for different reasons: microphones are affected by noise, cameras by lighting and occlusion, and text by speech-recognition errors or semantic ambiguity. A multimodal system can therefore become more robust if it learns when to trust each modality and when to treat disagreement as meaningful uncertainty.

A more advanced implementation could replace simple early concatenation with late fusion or attention-based fusion. Late fusion would allow each modality to produce an independent prediction before combining confidence scores. Attention-based fusion could dynamically weight the modalities according to their estimated reliability. The Week 3 pipeline does not test these architectures, but its ablation results provide the baseline needed to justify such future work.

### 4.5 HRI Design Implication

The technical classifier is only one part of a companion-AI system. The more important human-robot interaction question is what Fari should do after a model predicts possible distress.

A useful cognitive-science concept is **attentional capacity**, closely related to cognitive load. Human attentional resources are limited, and distress or confusion may already consume part of those resources. A robot that reacts with long explanations, rapid speech, repeated questioning, or visually complex interaction could increase the user's cognitive burden at the moment when the system is trying to help.

The response policy should therefore be conservative and low effort. If multimodal evidence suggests possible distress, Fari should use short sentences, slow and clear speech, and a small number of simple clarification questions. The classifier output should not be framed as a diagnosis. If evidence remains strong or the situation appears safety-critical, the system can escalate through an appropriate human-supervision protocol.

This design also connects to **theory of mind**. The model observes external behavior—speech acoustics, facial geometry, and words—but the target of interest is an underlying mental or emotional state. Observable behavior and internal state are not equivalent. The correct engineering stance is therefore cautious inference: multimodal evidence can support a hypothesis about a latent state, but the system should preserve uncertainty and choose actions that minimize harm when the hypothesis may be wrong.

---

## 5. NLP and Companion Dialogue Safety

### 5.1 Research-to-Deployment Mapping

The companion NLP bridge begins with the general problem of mental-state detection from conversational data: using observable language patterns to infer an underlying emotional or cognitive state. In a research setting, this can include patterns associated with distress, confusion, anxiety, or other atypical conditions. Fari presents a related but more operationally sensitive question: what signals in an elder user's conversational behavior suggest a state that may require a different companion response?

The two tasks share a machine-learning structure because both infer a latent state from observed language, but their deployment conditions differ substantially. Research-chat users may voluntarily engage with a system and may explicitly communicate emotional content. An eldercare companion would interact with a more heterogeneous population during everyday activities, where users may not intend to communicate distress and may not use direct linguistic indicators.

### 5.2 Distribution Shift

This difference creates a significant distribution shift. Age, conversational habits, cognitive state, environment, topic, speech characteristics, and social norms can all change the relationship between language and underlying state. A classifier trained primarily on explicit research-chat expressions could learn shortcuts such as direct distress vocabulary and then fail when deployed to users who communicate indirectly.

The Week 3 result reinforces this concern. Text alone achieved perfect performance on the synthetic dataset, but this is precisely why the result must be treated cautiously: the manually constructed language examples made class separation unusually easy. In real companion interaction, language will often be more ambiguous.

A stronger architecture should therefore avoid treating each utterance as an independent text-classification problem. A practical adaptation would combine sentence embeddings with recent audio and visual evidence, maintain a short interaction history, and calibrate confidence for safety-sensitive use. Temporal context matters because a single sentence may be uninformative while changes over several turns may reveal a meaningful pattern.

### 5.3 Semantics, Pragmatics, and Theory of Mind

A useful Language & Mind distinction is the difference between literal semantic content and intended or contextual meaning. A statement such as "I'm fine" has simple surface semantics, but its pragmatic meaning can change depending on tone, facial behavior, timing, and prior events. This is one reason a companion robot should not equate text content with mental state.

The same problem can be framed through theory of mind. Fari would observe words and behavior, but the engineering target is an unobservable internal state. The system therefore performs inference from evidence rather than direct measurement. This distinction is important for safety: confidence should be calibrated, and uncertain predictions should trigger clarification or low-risk support rather than an irreversible or high-consequence response.

### 5.4 Dialogue-Safety Policy

A companion dialogue-safety layer should treat model outputs as inputs to a response policy rather than as final decisions. When all modalities indicate a normal interaction with high confidence, the system can continue normally. When signals are weak or contradictory, the system can ask a short clarifying question. When multiple modalities consistently indicate possible distress or a predefined safety condition is triggered, the system can escalate according to a transparent human-supervision procedure.

This policy design also makes the asymmetric cost of errors explicit. Missing a genuine safety-relevant state may be more serious than creating one low-cost clarification, but excessive false alarms can reduce trust and create alarm fatigue. The appropriate threshold therefore depends on the action that follows the classifier. Low-risk supportive responses can use a lower threshold than high-consequence escalation.

The Week 3 work supports a broader design principle: **language should be one source of evidence inside a multimodal and temporally aware safety system, not the sole authority on the user's state**.

---

## 6. Cognitive Science and HCI Integration

### 6.1 Theory of Mind as a Modeling Constraint

Theory of mind is the ability to reason about mental states that are not directly observable from behavior. In a companion-AI context, this concept provides a useful warning against treating sensor measurements as direct labels of the user's internal condition.

Fari can observe acoustic features, facial geometry, text, and interaction history. None of these is equivalent to distress, confusion, intent, or belief. They are evidence from which a system may estimate an underlying state. A theory-of-mind framing therefore supports probabilistic and uncertainty-aware modeling rather than deterministic claims.

This distinction also changes the response objective. The purpose of the model is not to "prove" what the user feels, but to help the robot select a safe and appropriate interaction strategy under uncertainty.

### 6.2 Attentional Capacity and Cognitive Load

Attentional capacity is the most directly actionable cognitive-science concept in the Week 3 HRI design. If a person is already confused or distressed, their available attentional resources may be reduced. An interaction that requires several decisions, long explanations, or rapid responses can therefore make the situation worse.

For Fari, the design implication is concrete: responses to uncertain distress signals should minimize additional cognitive load. Short utterances, clear pacing, a small number of questions, and simple choices are preferable to complex dialogue trees.

The same principle also applies to Senpai. An educational companion should distinguish between productive challenge and excessive cognitive load. A system optimized only for continuous engagement could easily increase stimulation without improving understanding. Interaction design should therefore be tied to the user's task state rather than to visible activity alone.

### 6.3 Affect and Multimodal Evidence

The affect heuristic describes the way emotional state can influence judgments and decisions. In companion interaction, this matters because verbal responses may themselves be shaped by distress, fear, embarrassment, or confusion. A user may give a quick reassuring answer even when other signals suggest that the interaction state is not normal.

This provides a cognitive argument for multimodal sensing. Audio and visual evidence should not automatically override language, but they can reveal when literal verbal content may be incomplete. A robust system should model disagreement explicitly instead of forcing one modality to be treated as ground truth.

### 6.4 HCI and Human-Robot Interaction

Physical robots introduce interaction constraints beyond ordinary screen-based HCI. Movement, distance, gaze, timing, speech, and physical presence can all affect the user's experience. A companion robot's response is therefore both informational and embodied.

For Fari, this means that a multimodal classifier should not be evaluated separately from the behavior it triggers. A technically accurate model can still produce a poor system if its response policy is intrusive, confusing, or overconfident. Transparency is also important: the robot should avoid implying medical certainty when it is only responding to uncertain behavioral evidence.

For Senpai, the same principle applies to children. Engagement estimation must be combined with age-appropriate explanation, privacy, and pedagogical goals. For mobile or security robots, HRI also affects how systems signal intent, alerts, or movement around people.

The overall contribution of cognitive science and HCI to this internship is therefore not an additional prediction feature. It is a design framework for deciding **what the model output means, how confident the system should be, and what action is appropriate for the human context**.

---

## 7. Per-Platform Recommendations

### 7.1 Fari — Prioritize Uncertainty-Aware Multimodal State Estimation

The Week 3 experiment showed that visual and text modalities were individually sufficient to perfectly separate the small synthetic dataset, while audio was slightly weaker and full fusion matched the strongest individual models. This should not lead to a single-sensor design. Instead, it shows why Fari needs a pipeline capable of comparing heterogeneous evidence and identifying disagreement.

The recommended architecture is therefore a multimodal state-estimation layer that preserves modality-specific confidence and temporal context. Text should be treated as semantic evidence, audio as vocal-behavior evidence, and visual input as behavioral evidence. When modalities conflict, the system should lower confidence and prefer a low-risk clarification strategy. The response layer should use short, cognitively simple interactions and escalate only through explicit safety rules.

### 7.2 Senpai — Reuse the Multimodal Architecture, but Change the Objective

Senpai can reuse much of the same audio, visual, and language feature architecture as Fari, but the target labels should represent educationally meaningful states rather than distress. Possible targets include confusion, help-seeking, sustained attention, and task progress.

The most important recommendation is to avoid optimizing only for visible engagement. A child can look attentive without understanding, and a student who looks away may still be reasoning effectively. Sensor evidence should therefore be combined with lesson state, answer correctness, timing, and help requests. The system's objective should reflect learning support rather than maximizing interaction duration.

### 7.3 Sentinel Prime AI — Put Calibration and Safety Gating Before Automated Response

Sentinel's core challenge is not simply maximizing event-detection sensitivity. Excessive false positives create alert fatigue, while missed events create operational risk. The platform should therefore emphasize calibrated uncertainty, temporal confirmation, multimodal evidence, and abstention when confidence is insufficient.

The Week 2 RL experiment also provides a useful warning: reward or decision policies can exploit whatever objective is defined for them. Security-response automation should therefore retain hard operational safety rules and auditability rather than relying only on a learned policy to decide when and how to act.

### 7.4 Aido Rover — Separate High-Level Learned Policy from Low-Level Safety Control

The Aido Rover experiment provides the strongest quantitative result in the internship. PPO achieved **188.09 ± 35.26** held-out reward, substantially above the simple rule-based and random baselines. This supports continued investigation of learned high-level decision policies for tasks such as patrol, fault response, charging, and investigation.

However, the current action space is deliberately abstract. A practical rover should separate high-level mission decisions from lower-level navigation and actuator control. Hard safety constraints, collision avoidance, emergency stopping, battery protection, and validated motion-control rules should remain outside the unconstrained learned policy. The next experimental step should use more realistic noisy sensor distributions and simulation before attempting any real-hardware transfer.

### 7.5 Aido Humanoid — Treat as a Continuous-Control and Sim-to-Real Research Program

Aido Humanoid represents a fundamentally more difficult control problem than the Week 2 rover task. Humanoid locomotion and manipulation require continuous, high-dimensional control with strong safety constraints. The four-action discrete PPO environment should therefore not be presented as directly transferable to humanoid control.

The recommended direction is hierarchical control: high-level task or behavioral decisions can be separated from low-level locomotion and manipulation policies. Simulation, domain randomization, system identification, and conservative sim-to-real validation would be central. Multimodal perception and language would then provide the state and task context used by the high-level controller.

### 7.6 Portfolio-Level Recommendation

Across the five platforms, the most reusable architectural pattern is not a single algorithm but a division of responsibilities:

1. **Perception** extracts features from physical sensors.
2. **State understanding** integrates modalities and temporal context.
3. **Decision policy** selects an operational or interactive response.
4. **Safety validation** constrains or blocks unsafe actions.
5. **Execution and feedback** update the next observation.

This structure allows different products to reuse technical components while preserving domain-specific labels, rewards, safety rules, and human-interaction requirements.

---

## 8. Cross-Track Synthesis

### 8.1 Perception and Decision as Adjacent Problems

The RL and multimodal tracks initially appear to be separate projects. Week 2 trains a policy, while Week 3 trains a classifier. In a physical-AI system, however, they occupy adjacent stages of the same loop.

The multimodal system addresses the question: **what state is the world or user currently in?** The RL system addresses the next question: **given that state, what action should the system take?** This leads to a general architecture:

**sensor observations → multimodal perception → state representation → decision policy → safety validation → action → new observations**

The quality of downstream decisions is therefore limited by the quality of upstream state estimation. A policy trained on clean synthetic state vectors may perform well in simulation but degrade when a real perception system produces noisy or uncertain estimates. Conversely, an accurate perception model creates little value if the resulting action policy is poorly aligned with operational objectives.

### 8.2 Uncertainty Should Cross the Model Boundary

One of the strongest synthesis lessons is that state uncertainty should not disappear when information passes from perception to decision. The Week 3 pipeline currently outputs a class prediction, while the Week 2 environment assumes a defined state vector. A more realistic physical-AI architecture should preserve confidence information so that the decision layer knows when state estimation is unreliable.

For example, a companion system should behave differently when distress probability is high across all modalities than when text suggests distress but audio and visual evidence are neutral. Similarly, a rover should treat uncertain sensor values differently from confirmed fault conditions.

This suggests an architectural improvement for future work: policies should consume not only estimated states but also confidence, missingness, or modality-quality indicators.

### 8.3 Safety Cannot Be Reduced to Reward

The Week 2 reward-design analysis demonstrates that an optimization objective can be exploited. The Week 3 HRI analysis demonstrates that even a correct prediction can lead to a poor interaction if the response is inappropriate. Together, these results imply that safety should not be represented only as another reward term or classifier label.

Hard constraints, fallback behavior, abstention, human supervision, and domain-specific rules should exist alongside learned models. This is particularly important in eldercare, child interaction, security, and mobile robotics, where model errors can affect people directly.

### 8.4 PIC 2.0 as a Useful Conceptual Map

The internship's PIC 2.0 framing is useful because it separates several functions that are often collapsed into the phrase "AI model": perception, temporal understanding, semantic representation, decision policy, hierarchical reasoning, and continual learning.

The four-week work does not validate InGen's internal PIC 2.0 implementation. Instead, it provides a concrete open-source example of why a physical-intelligence platform may need multiple specialized components. The Week 3 pipeline represents a simplified perception and state-understanding stage. The Week 2 PPO policy represents a simplified decision stage. Cognitive science and HCI provide constraints on how those decisions should affect humans.

The central synthesis is therefore that physical intelligence depends on **coordination between models and safety layers**, not merely on selecting the highest-performing learning algorithm.

---

## 9. Limitations and Future Work

### 9.1 Synthetic Data and Simulation Bias

The largest limitation of both technical tracks is the use of synthetic environments and synthetic data. In the Aido Rover task, sensor dynamics and fault events are generated by a simplified Gymnasium environment rather than from real telemetry. In the Fari task, audio, facial features, and text examples are generated according to explicit class rules.

Synthetic data was appropriate for a four-week portfolio-safe internship because it enabled reproducible experimentation without access to proprietary hardware or sensitive user data. However, the same simplicity that makes experimentation possible also creates unrealistic separability and cleaner dynamics than real systems.

Future work should therefore focus on closing the distribution gap rather than simply increasing model size.

### 9.2 More Realistic Rover Simulation

The next RL iteration should introduce noise, delayed observations, missing sensor values, more variable fault processes, and richer battery and motion dynamics. Multiple environment configurations could be used to test robustness rather than evaluating on only one simulator distribution.

The rule-based baseline should also be strengthened. The current baseline is intentionally simple, so PPO's large advantage demonstrates successful learning relative to that heuristic but does not establish superiority over a carefully engineered operational controller. Future evaluation should compare PPO against several increasingly capable baselines and report confidence intervals or statistical tests across multiple seeds.

A further extension would separate high-level operational actions from continuous low-level motion control. PPO could remain responsible for high-level modes while a dedicated controller handles velocity and navigation, or continuous-control methods such as SAC or TD3 could be evaluated in simulation for lower-level decisions.

### 9.3 Realistic Multimodal Data and Temporal Modeling

The Fari pipeline should next be evaluated on realistic public datasets or newly collected consented data that contain natural variation rather than explicit synthetic class rules. The goal would be to test whether the architecture remains useful when individual modalities are ambiguous.

Temporal context should also be added. The current classifier treats each interaction as an independent example, but companion-state understanding is inherently sequential. Recurrent models, temporal transformers, or simpler rolling-window feature aggregation could capture changes in speech, expression, and dialogue over time.

Fusion should be upgraded from static concatenation to a method that represents modality reliability. Late fusion, confidence-weighted fusion, or attention-based fusion could allow the model to reduce the influence of a noisy channel without discarding the modality entirely.

### 9.4 Calibration and Uncertainty

Both tracks would benefit from explicit uncertainty analysis. For Fari, the classifier should be calibrated so that predicted probabilities correspond meaningfully to empirical likelihood. Thresholds can then be tied to different response costs.

For Aido Rover, uncertainty can be represented in observations or in model ensembles. A high-level policy should be able to abstain or fall back to a safe rule when sensor state is unreliable.

This is especially important because real physical systems experience out-of-distribution conditions that are difficult to enumerate during training.

### 9.5 Human Evaluation

The current work evaluates models, not human experience. A companion robot can have strong classification metrics and still fail because users find the interaction intrusive, confusing, or untrustworthy. Future Fari work should therefore include human-centered evaluation of response timing, wording, transparency, perceived safety, and cognitive load.

Similarly, Senpai would require studies of learning outcomes rather than only engagement prediction. Physical-AI evaluation must include the effects of model behavior on the human system around it.

### 9.6 What an Additional Month Would Explore

With an additional month, the highest-value continuation would be:

1. Make the rover simulator more realistic with stochastic sensor noise and multiple environment configurations.
2. Evaluate PPO across multiple random seeds and against stronger rule-based baselines.
3. Add calibrated uncertainty or abstention to the rover decision layer.
4. Replace the synthetic Fari data with a more realistic public multimodal dataset where possible.
5. Add temporal context to the multimodal classifier.
6. Compare early fusion with late or attention-based fusion.
7. Build a simple end-to-end prototype in which a multimodal state estimate, including confidence, feeds a constrained response policy.
8. Evaluate not only classification or reward metrics but also safety-oriented metrics such as false-negative rate, abstention frequency, and recovery behavior.

The purpose of this future work would not be to make the current prototype appear more production-ready than it is. It would be to test whether the architectural lessons from the four-week internship survive under more realistic uncertainty and interaction conditions.

---

## 10. References

1. InGen Dynamics, public company and product materials covering Origami AI / PIC 2.0, Fari, Senpai, Sentinel Prime AI, Aido Rover, and Aido Humanoid.
2. InGen Dynamics, *Machine Learning & Neural Network Analyst Internship — 4-Week Accelerated Remote Program Plan*, v1.0, July 2026.
3. InGen Dynamics, *ML & NN Analyst Internship Concepts Primer — RL, Multimodal AI, Cognitive Science*, 2026.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.
5. DeepSeek-AI. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” arXiv:2501.12599, 2025. Used as a public reference for Group Relative Policy Optimization concepts.
6. Stable Baselines3 documentation, PPO implementation and TensorBoard logging.
7. Gymnasium documentation, custom environment API.
8. McFee, B. et al., librosa documentation, MFCC feature extraction.
9. Google MediaPipe documentation, facial landmark / face mesh feature extraction.
10. Reimers, N. and Gurevych, I. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” EMNLP-IJCNLP, 2019.
11. Sentence Transformers documentation, `all-MiniLM-L6-v2` sentence embedding model.
12. Premack, D. and Woodruff, G. “Does the chimpanzee have a theory of mind?” *Behavioral and Brain Sciences*, 1978.
13. Slovic, P. et al. work on the affect heuristic and risk judgment.
14. Sweller, J. work on cognitive load theory and limited attentional capacity.
