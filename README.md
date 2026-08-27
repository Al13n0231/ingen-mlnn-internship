# InGen MLNN Internship

Machine Learning & Neural Network Analyst Internship — 4-week accelerated remote program focused on reinforcement learning, multimodal AI, cognitive science, and physical AI.

## Project Overview

This repository contains the technical work completed across four internship weeks. The project connects three complementary areas:

- **Reinforcement Learning** — custom Gymnasium environment design, reward shaping, PPO training, TensorBoard logging, and held-out evaluation for a simplified Aido Rover patrol task.
- **Multimodal AI** — audio, visual, and language feature extraction and fusion for a synthetic Fari companion-state classification task.
- **Cognitive Science / HCI** — theory of mind, attentional capacity, cognitive load, language pragmatics, and human-robot interaction considerations applied to companion-AI design.

The work uses only public or synthetic data and does not rely on InGen internal systems, proprietary datasets, or confidential technical documentation.

## Technical Arc

### Week 1 — InGen Landscape and Technical Bridges

Week 1 established the physical-AI context and mapped prior ML experience to InGen product scenarios.

Artifacts:

- `W01_InGen_MLNN_Landscape.md` — ML/NN analysis of Fari, Senpai, Sentinel Prime AI, Aido Rover, Aido Humanoid, and PIC 2.0.
- `W01_RL_PIC2_Bridge.md` — conceptual bridge between PPO and GRPO and between game RL and physical-robot RL.
- `W01_Multimodal_Bridge.md` — bridge from a multimodal presentation-analysis architecture to Fari companion AI.
- `W01_env_check.ipynb` — Python/ML dependency and device validation.
- `weekly/Wk-01-Recap.md` — Week 1 recap.

### Week 2 — Reinforcement Learning for Aido Rover

Week 2 implemented a synthetic high-level Aido Rover patrol environment and trained a PPO policy.

The observation state represents robot-style sensor values such as battery state of charge, motor current, IMU magnitude, proximity, and RSSI. The discrete action space contains:

- `PATROL`
- `ALERT`
- `CHARGE`
- `INVESTIGATE`

The PPO agent was trained for **200,000 timesteps** with seed `42` and `gamma = 0.99`.

Held-out evaluation across 20 episodes:

| Policy | Mean Reward | Std. Dev. |
| --- | ---: | ---: |
| PPO | **188.09** | **35.26** |
| Rule-based | **13.12** | **50.45** |
| Random | **-84.89** | **17.19** |

Artifacts:

- `week02_rl/W02_RL_Environment.ipynb`
- `week02_rl/W02_PPO_Training.ipynb`
- `week02_rl/W02_RL_Analysis_Memo.md`
- `week02_rl/models/`
- `week02_rl/tensorboard_logs/`
- `configs/W02_ppo_config.json`
- `weekly/Wk-02-Recap.md`

### Week 3 — Multimodal Sensor Fusion for Fari

Week 3 implemented a three-modality pipeline for a synthetic distress-vs-normal companion interaction task.

Feature representation:

- **Audio:** 26 MFCC summary features
- **Visual:** six facial-geometry features
- **Text:** 384-dimensional `all-MiniLM-L6-v2` sentence embeddings
- **Fusion:** 416-dimensional early-concatenated representation

The fused logistic-regression classifier was evaluated on a held-out 16-example test set.

| Feature Set | F1 | AUROC |
| --- | ---: | ---: |
| Audio only | **0.875** | **0.984** |
| Visual only | **1.000** | **1.000** |
| Text only | **1.000** | **1.000** |
| Audio + Visual + Text | **1.000** | **1.000** |

The perfect fused, visual-only, and text-only scores should be interpreted only within the small rule-generated synthetic dataset. They demonstrate pipeline coherence and feature separability, not validated real-world elder-distress detection performance.

Artifacts:

- `W03_Multimodal_Pipeline.ipynb`
- `W03_Companion_NLP_Note.md`
- `W03_Multimodal_Memo.md`
- `weekly/Wk-03-Recap.md`

### Week 4 — Capstone Synthesis

Week 4 combines the RL, multimodal, and cognitive-science tracks into a single physical-AI engineering narrative.

The central architecture is:

**sensor observations → perception / feature extraction → state understanding → decision policy → safety validation → action**

Artifacts:

- `week04_capstone/W04_Capstone_Report.md` — full capstone source report.
- `week04_capstone/W04_Capstone_Report.docx` — final formatted report.
- `week04_capstone/W04_Capstone_Deck.pptx` — 12-slide executive readout.
- `week04_capstone/W04_Retrospective.md` — technical and career retrospective.
- `weekly/Wk-04-Final-Recap.md` — final weekly recap.

## Key Findings

### RL

The PPO policy substantially outperformed both the random and simple rule-based baselines in the synthetic rover environment. The more important result is methodological: when reinforcement learning is transferred from a game to a robot-like task, **reward alignment and environment design become central engineering problems**.

### Multimodal AI

The Week 3 pipeline successfully combined heterogeneous audio, visual, and language representations into a common classifier. Visual and text features tied as the strongest individual modalities in the synthetic experiment. Because the data-generation rules were intentionally separable, the result motivates more realistic data, temporal modeling, and uncertainty-aware fusion rather than a claim that one modality is universally dominant.

### Cognitive Science / HCI

The strongest HRI implication is that model outputs should not be treated as direct measurements of a person's internal state. Theory of mind, attentional capacity, and cognitive-load reasoning support a cautious response design in which uncertain state estimates trigger low-risk clarification and simple interaction rather than overconfident automation.

## Repository Structure

```text
ingen-mlnn-internship/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── W02_ppo_config.json
├── W01_InGen_MLNN_Landscape.md
├── W01_RL_PIC2_Bridge.md
├── W01_Multimodal_Bridge.md
├── W01_env_check.ipynb
├── week02_rl/
│   ├── W02_RL_Environment.ipynb
│   ├── W02_PPO_Training.ipynb
│   ├── W02_RL_Analysis_Memo.md
│   ├── models/
│   └── tensorboard_logs/
├── W03_Multimodal_Pipeline.ipynb
├── W03_Companion_NLP_Note.md
├── W03_Multimodal_Memo.md
├── week04_capstone/
│   ├── W04_Capstone_Report.md
│   ├── W04_Capstone_Report.docx
│   ├── W04_Capstone_Deck.pptx
│   └── W04_Retrospective.md
└── weekly/
    ├── Wk-01-Recap.md
    ├── Wk-02-Recap.md
    ├── Wk-03-Recap.md
    └── Wk-04-Final-Recap.md
```

## Environment

The project was developed with **Python 3.11.9**.

The Week 1 environment check confirmed the core stack:

- PyTorch
- Stable Baselines3
- Gymnasium
- NumPy
- pandas
- matplotlib
- librosa
- OpenCV
- scikit-learn
- transformers
- sentence-transformers

OpenCV is used as the visual-processing backend. MediaPipe is not required by the final Week 3 implementation.

## Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/Al13n0231/ingen-mlnn-internship.git
cd ingen-mlnn-internship
```

### 2. Create a Python 3.11 virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the environment

Run:

```text
W01_env_check.ipynb
```

The notebook checks the required ML dependencies and reports the selected PyTorch device.

### 5. Run the Week 2 RL notebooks

Run from the repository's `week02_rl` directory in this order:

```text
W02_RL_Environment.ipynb
W02_PPO_Training.ipynb
```

The training notebook uses the committed configuration in `configs/W02_ppo_config.json` and writes model / TensorBoard outputs under `week02_rl/`.

To inspect the TensorBoard logs:

```bash
tensorboard --logdir week02_rl/tensorboard_logs
```

### 6. Run the Week 3 multimodal notebook

From the repository root, run:

```text
W03_Multimodal_Pipeline.ipynb
```

The notebook uses a fixed random seed and generates the synthetic multimodal data programmatically, so no private dataset is required.

> The first run of the sentence-transformer component may download the public `all-MiniLM-L6-v2` model.

## Reproducibility Notes

- Random seed `42` is used for the Week 2 PPO experiment and Week 3 synthetic classifier workflow.
- PPO hyperparameters are externalized in `configs/W02_ppo_config.json`.
- Held-out episodes are used for the final RL evaluation.
- Week 3 uses an 80/20 train/test split with a fixed seed.
- TensorBoard logs are committed for inspection.
- All Week 3 data are synthetic and generated within the notebook.
- Reported metrics should be interpreted within the corresponding synthetic environment or dataset.

## Scope and Limitations

This repository is an open-source internship portfolio artifact. It does **not** contain:

- InGen confidential information
- internal product architecture
- proprietary customer or robot telemetry
- real eldercare interaction data
- production robot-control software

References to PIC 2.0 and InGen products are based on internship materials and public product context. The implementations are research abstractions designed to explore transferable ML engineering concepts.

## Capstone Takeaway

The project demonstrates that physical AI is not only a model-selection problem. A complete system must connect **perception, state estimation, decision-making, reward alignment, uncertainty, safety, and human interaction**.

The Week 2 RL track demonstrates sequential decision-policy engineering. The Week 3 multimodal track demonstrates multimodal state estimation. The cognitive-science/HCI track provides the design constraints needed to convert uncertain model outputs into safer human-facing behavior.
