# Companion NLP Bridge: Mental-State Research to Fari Dialogue Safety

## Research to Deployment Mapping

The Northeastern mental state detection research asks how conversational signals can be used to infer a person's underlying mental or emotional state from chat data. The central ML problem is to identify patterns in language that may correlate with states such as distress, confusion, anxiety, or other atypical conditions.

The analogous question for Fari is: **what signals in an elder user's conversational behavior indicate possible distress, confusion, or an atypical emotional state that may require a different companion response?**

The two problems share a common NLP structure because both attempt to infer latent human states from observable language. However, the deployment contexts differ substantially.

## Methodological Differences

Research chatbot data may come from users who voluntarily choose to engage with a system and may intentionally communicate their emotional state. A Fari deployment would instead involve a cognitively and behaviorally diverse elderly population interacting with a companion robot during everyday activities. Users may not explicitly communicate distress, and observable language may conflict with tone of voice or facial behavior.

This creates an important distribution shift. A model trained only on research-chat language may rely too heavily on direct semantic indicators such as explicit statements of anxiety or confusion. For a companion robot, the architecture should therefore incorporate multimodal and temporal context rather than treating each utterance as an independent text-classification problem.

A practical adaptation would combine sentence embeddings from recent dialogue with audio and visual features, maintain a short interaction history, and calibrate the decision threshold for safety-sensitive use. The model should also represent uncertainty rather than treating every prediction as a definitive statement about the user's condition.

## Cognitive Science Framing

A useful Language & Mind distinction is the difference between the literal semantic content of an utterance and the speaker's intended or underlying meaning. An elder may say that they are "fine" while their voice and facial behavior communicate a conflicting state. Language alone therefore provides an incomplete observation of the user's internal state.

This also connects to **theory of mind**: the system is not merely recognizing words but attempting to reason from observable behavior toward an unobservable mental state. For Fari, multimodal evidence should therefore support cautious inference rather than confident diagnosis.

The engineering implication is that companion dialogue safety should use language as one source of evidence within a larger multimodal model. When signals disagree or model confidence is low, Fari should seek clarification or use a low-risk supportive response rather than acting as though the inferred mental state were certain.
