# Companion AI Multimodal Memo

## 1. Technical Findings

The Week 3 experiment implemented a three-modality proof of concept classifier for a synthetic Fari companion interaction task. Audio information was represented using 26 MFCC summary features, visual information using six facial-geometry features, and dialogue content using 384-dimensional MiniLM sentence embeddings. Early concatenation produced a 416-dimensional multimodal representation for each interaction.

The fused classifier achieved a held out **precision of 1.000**, **recall of 1.000**, **F1 score of 1.000**, and **AUROC of 1.000** on the 16-example test set.

The precision recall tradeoff is especially important in a companion-robot setting. A false positive could cause Fari to interrupt or unnecessarily escalate a normal interaction, while a false negative could cause the system to miss a genuine signal of distress. For this reason, classifier quality cannot be reduced to accuracy alone. In a real deployment, the operating threshold would need to reflect the relative safety cost of these two error types.

The ablation experiment showed that **visual and text features tied as the strongest individual modalities**, each achieving an F1 score of **1.000** and an AUROC of **1.000**. Audio alone achieved an F1 score of **0.875** and an AUROC of **0.984**. The complete audio, visual, and text fusion also achieved an F1 score of **1.000** and an AUROC of **1.000**. Therefore, fusion matched the strongest individual modalities rather than improving beyond them in this synthetic experiment.

These results should be interpreted cautiously. The dataset contains only 80 synthetic interactions generated from explicit class rules. The audio waveforms are synthetic, the facial measurements are simulated geometry values, and the dialogue examples are manually constructed. The perfect performance of the visual, text, and fused classifiers likely reflects the separability of the synthetic distributions rather than realistic real-world performance. The experiment therefore validates the structure of the multimodal pipeline rather than the ability to identify distress in real elderly users.

## 2. Human-Robot Interaction Design Implication

A useful cognitive-science concept for Fari's response design is **attentional capacity**, closely related to cognitive load. Human attentional resources are limited, and distress or confusion may already consume part of those resources. A robot response that introduces additional questions, complex instructions, rapid speech, or excessive interface activity could therefore worsen the interaction rather than help the user.

When the multimodal classifier detects a potential distress signal, Fari should consequently avoid treating the classification as a definitive diagnosis. A safer response strategy would use short sentences, slow and clear speech, and a small number of low-effort clarification questions. If multimodal evidence remains strong or the situation appears safety critical, the system could then escalate according to an appropriate human-supervision protocol.

This design also connects to **theory of mind**. The classifier observes external signals—speech acoustics, facial geometry, and words—but the target concept is an underlying mental state. Observable behavior and internal state are not identical. For example, a user could verbally state that they are fine while their acoustic and visual behavior suggest otherwise.

Multimodal sensing is therefore useful not because any one signal directly measures distress, but because different modalities provide complementary evidence about an uncertain latent state. A companion robot should use that evidence cautiously, preserve uncertainty, and choose responses that minimize harm when the prediction may be incorrect.

The Week 3 experiment suggests that companion AI should be designed as a combination of multimodal perception and human-centered response policy. Classification performance is important, but the value of the system ultimately depends on how uncertainty in the model is translated into safe and cognitively appropriate interaction.
