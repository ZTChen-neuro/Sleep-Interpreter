# Interpreting Sleep Activity Through Neural Contrastive Learning

## Summary
Spontaneous memory replay during sleep is crucial for cognition but challenging to capture because distinct sleep rhythms hinder the generalization of wake-trained electroencephalogram (EEG) decoders. To address this, we developed the Sleep Interpreter (SI), which uses neural contrastive learning to isolate shared semantic content from background rhythms. We collected a dataset of 135 participants undergoing targeted reactivation of 15 semantic categories, yielding approximately 1,000 h of overnight sleep and 400 h of wake EEG. During non-rapid eye movement (NREM) sleep, SI achieved high decoding accuracy for cue-evoked semantic responses, with accuracy peaking during slow oscillation and spindle coupling at 40.02% top-1 accuracy on unseen participants (chance 6.7%). We demonstrated SI generalizability in two independent nap experiments involving targeted and spontaneous reactivation, where decoded reactivations correlated with post-sleep memory performance. Finally, we implemented SI for real-time sleep staging and stage-specific NREM and REM decoding. The dataset and codebase are shared as open resources for future clinical applications.

## Overview
This repository contains code and resources for our **real‑time sleep decoding system “Sleep Interpreter”**. It includes:
- A **real-time sleep decoding system demo code**.
- Training and evaluation scripts for **NREM 2/3** and **REM** models (sleep decoding).
- Example **data** and **model checkpoints** folders (data and checkpoints not included).

---

## Repository Layout

```
.
├─ Real-time decoding system/
│  ├─ Model/                 # Models/checkpoints used by the real‑time system
│  ├─ Parameters/            # Config files for the system (paths, thresholds, etc.)
│  ├─ Utils/                 # Helper functions for the system
│  ├─ Main_Decode.py         # Entry point for the real‑time system
│  └─ Subject.py             # Subject configuration
│
├─ data/
│  ├─ Model checkpoint/      # Pretrained checkpoints (not included, in "OSF")
│  ├─ SI Main Test/          # Test data (not included, in "OSF")
│  ├─ SI Main Train/         # Preprocessed training data for Lasso GLM model (not included, in "OSF")
│  ├─ SI Main Train_pickle/  # Preprocessed training data in pickle format for REM model SISD (not included, in "OSF")
│  ├─ SI Main Train_tfrecords/ # Preprocessed training data in tfrecord format for NREM and REM model (not included, in "dropbox")
│  └─ SI Staging Label/      # Expert-labelled sleep stages
│
├─ environment/
│  └─ environment.yaml       # Environment file
│
├─ lasso_glm/
│  ├─ lasso_glm_audio2audio.py   # Within‑domain (audio→audio) linear model result
│  ├─ lasso_glm_image2image.py   # Within‑domain (image→image) linear model result
│  ├─ lasso_glm_awake2sleep.py   # Cross‑domain (awake→sleep) linear model result
│  ├─ lasso_glm_N2N3N2N3.py      # Within‑domain (NREM→NREM)  linear model result
│  ├─ lasso_glm_REM2REM.py       # Within‑domain (REM→REM) linear model result
│  ├─ lasso_glm_NREM2awake.py    # Cross‑domain (NREM→awake) linear model result
│  └─ lasso_glm_REM2awake.py     # Cross‑domain (REM→awake) linear model result

│
├─ model/
│  ├─ lasso_glm/             # Lasso GLM model for sleep decoding
│  ├─ N23SIMD/               # NREM2/3 & awake data trained model for sleep decoding
│  ├─ N23SISD/               # NREM2/3 data trained model for sleep decoding
│  ├─ REMSIMD/               # REM & awake data trained model for sleep decoding
│  └─ REMSISD/               # REM data trained model for sleep decoding
│
├─ params/
│  └─ lasso_glm_params.py    # Hyperparameters for Lasso GLM experiments
│
├─ test/
│  ├─ N23SIMD.py             # Evaluation: sleep decoding on NREM2/3 data
│  ├─ N23SIMD_retrained_SO.py# Evaluation: sleep decoding on SO‑phases data
│  ├─ N23SIMD_SO.py          # Evaluation: sleep decoding on SO‑phases data
│  ├─ N23SISD.py             # Evaluation: sleep decoding on NREM2/3 data
│  ├─ REMSIMD.py             # Evaluation: sleep decoding on REM data
│  └─ REMSISD.py             # Evaluation: sleep decoding on REM data
│
├─ train/
│  ├─ N23_finetune.py        # Fine‑tuning for N23SIMD model on NREM2/3 data
│  ├─ N23SIMD.py             # Training: NREM2/3 and awake data to NREM2/3 stage sleep decoder
│  ├─ N23SISD.py             # Training: NREM2/3 data to NREM2/3 stage sleep decoder
│  ├─ REM_finetune.py        # Fine‑tuning for REMSIMD model on REM data
│  ├─ REMSIMD.py             # Training: REM and awake data to REM stage sleep decoder
│  └─ REMSISD.py             # Training: REM data to REM stage sleep decoder
│
└─ utils/
   ├─ __init__.py
   ├─ DotDict.py             # Simple dict helper
   └─ model.py               # Common model utilities
```

---
## Setup

1. **Create the environment**
   ```bash
   conda env create -f environment/environment.yaml
   conda activate tensorflow
   ```

2. **Prepare data**
   - Place the data and model checkpoints in the "OSF" link following the structure shown under `data/`.

---

## How to Run

### 1) Lasso GLM model
From the corresponding folder path, run any of the scripts below:
```bash
cd lasso_glm
python lasso_glm_image2image.py
python lasso_glm_audio2audio.py
python lasso_glm_awake2sleep.py
python lasso_glm_N2N3N2N3.py
python lasso_glm_NREM2awake.py
python lasso_glm_REM2REM.py
python lasso_glm_REM2awake.py
```
Output metrics are saved under the directory `results/`.

### 2) Train Sleep‑Interpreter models
```bash
cd train
# NREM2/3 family
python N23SIMD.py
python N23SISD.py
python N23_finetune.py

# REM family
python REMSIMD.py
python REMSISD.py
python REM_finetune.py

```
Checkpoints are saved under the directory `results/Decoding checkpoint/`.

### 3) Evaluate / Test
```bash
cd test
python N23SIMD.py
python N23SISD.py
python REMSIMD.py
python REMSISD.py
python N23SIMD_SO.py
python N23SIMD_retrained_SO.py
```
Output metrics are saved under the directory `results/`.

