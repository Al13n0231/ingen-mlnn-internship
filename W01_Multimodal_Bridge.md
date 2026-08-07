# W01 — Multimodal Presentation Analysis to Fari Companion AI Bridge

## 1. Three-Modality Mapping

The Multimodal Presentation Analysis project combined audio, visual, and
language-related information to evaluate presentation effectiveness. Fari's
companion AI requires the same general modality types, but the inputs and
prediction target change from presentation performance to the user's
interaction or emotional state.

| Presentation Analysis Project | Fari Companion AI Analogue |
|---|---|
| Audio speech analysis, including vocal delivery and speaking patterns | Voice audio from the elder, including tone, speech rate, hesitation, pauses, pitch variation, and vocal energy |
| Facial-expression analysis from recorded video | Camera observations of facial expression and body posture |
| Slide or visual-content analysis | Conversational dialogue text representing what the elder says |
| Presentation-effectiveness prediction | Companion interaction-state prediction, such as distress or normal interaction |

The audio and facial feature-extraction components can remain conceptually
similar. Audio features such as MFCC-based summaries can represent properties
of speech, while facial landmarks or expression features can represent visual
behavior. The original slide-content component must be replaced with a
language representation, such as a sentence embedding of the user's latest
dialogue turn. The principal architectural change is therefore not the number
of modalities, but the meaning of the third modality and the classification
target. The primer describes this as a change from presentation effectiveness
to companion interaction state while retaining a similar three-modality fusion
structure. :contentReference[oaicite:1]{index=1}

## 2. Adaptation for Real-Time Companion Interaction

The original presentation-analysis pipeline operated on a completed or
recorded presentation. Fari would instead receive continuous audio, video, and
dialogue inputs during an ongoing interaction. The pipeline must therefore
process short temporal windows rather than waiting for an entire session.

Latency becomes a central constraint because the robot must respond while the
interaction is still relevant. Audio, visual, and text features may also arrive
at different times. Audio and video are continuous streams, while text becomes
available only after speech recognition produces a dialogue transcript.
Consequently, the system must align the modalities using timestamps and define
how much recent context should be included in each prediction.

The system must also remain useful when one modality is missing or unreliable.
Poor lighting may weaken visual features, background noise may affect audio,
and speech recognition may produce inaccurate text. Rather than treating all
modalities as equally reliable at every moment, a production system should
track modality availability and confidence.

Finally, contradictory signals must not be treated as an error automatically.
An elder might say “I am fine” while speaking with a strained voice and showing
a tense facial expression. Such disagreement may contain more useful
information than agreement between all three channels.

## 3. Cognitive Science Framing: Theory of Mind

Theory of mind is the capacity to attribute underlying mental states, such as
beliefs, intentions, knowledge, and emotions, to another person. Applied to
Fari, this means that the robot should not treat observable audio, visual, and
linguistic signals as the user's mental state itself. Instead, those signals are
evidence from which the system attempts to infer an underlying state such as
confusion, distress, loneliness, or comfort.

This distinction matters because observable behavior and internal state do not
always match. A user may suppress distress, provide a socially expected verbal
response, or lack the words needed to explain their condition. A text-only
system could interpret “I am fine” literally, whereas multimodal analysis may
detect hesitation, reduced vocal energy, or a conflicting facial expression.
The theory-of-mind framing therefore provides a cognitive-science justification
for combining the three modalities instead of relying on a single observable
signal. The primer specifically requires cognitive-science integration to name
the concept, define it, and connect it to a concrete Fari design decision.
:contentReference[oaicite:2]{index=2}

## 4. Design Implication for Fari

The multimodal model should produce a confidence-based interaction-state
assessment rather than an unconditional diagnosis. When several modalities
indicate possible distress, Fari should respond conservatively by asking a
short clarifying question, reducing additional cognitive load, or escalating
according to an approved safety process.

When the modalities disagree or confidence is low, Fari should avoid making a
strong conclusion from a single signal. It should gather additional context
through continued observation or dialogue. This design treats multimodal
fusion as support for safer interaction and mental-state reasoning, rather than
as proof that the system can directly observe a person's internal state.