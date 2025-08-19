# Interpreting Sleep Activity Through Neural Contrastive Learning

## Abstract
Memories are spontaneously replayed during sleep, a process thought to support memory consolidation. However, capturing this replay in humans has been challenging because unlike wakefulness, sleep EEG is dominated by slow, rhythmic background activity. Moreover, each sleep stage (e.g., NREM, REM) has distinct rhythms, hindering generalisation of models trained on wake-state data. To overcome these challenges, we developed the Sleep Interpreter (SI), a neural network model that decodes memory replay from sleep EEG. In a large dataset comprising 135 participants (~1,000 h of overnight sleep; ~400 h of wake), we employed a TMR-like paradigm with 15 semantically congruent cue-image pairs to tag specific memories. SI was trained separately for NREM and REM using contrastive learning to align neural patterns across wake and sleep, filtering out stage-specific background rhythms. We also examined how slow oscillations and spindle coupling influence decoding in NREM sleep. In a 15-way classification, SI achieved up to 40.02% Top-1 accuracy on unseen subjects. To test generalisability, we followed up with two independent nap experiments in separate samples and applied the trained SI model off-the-shelf. The first probed spontaneous reactivation without auditory cues, while the second used semantic-free sounds with new images. In both, SI successfully decoded reactivation during sleep that correlated with post-nap memory performance. By openly sharing our dataset and SI system, we provide a unique resource for advancing research on memory and learning during sleep, and related disorders.

## Overview
This repository contains code and resources for our **real‑time sleep decoding system “Sleep Interpreter”**. It includes:
- A **real-time sleep decoding system demo code**.
- Training and evaluation scripts for **NREM 2/3** and **REM** models (staging & sleep decoding).
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
│  ├─ Model checkpoint/      # Pretrained checkpoints (not included, in "dropbox")
│  ├─ SI Main Test/          # Test data (not included, in "dropbox")
│  ├─ SI Main Train/         # Preprocessed training data for Lasso GLM model (not included, in "dropbox")
│  ├─ SI Main Train_pickle/  # Preprocessed training data in pickle format for REM model SISD (not included, in "dropbox")
│  ├─ SI Main Train_tfrecords/ # Preprocessed training data in tfrecord format for NREM and REM model (not included, in "dropbox")
│  ├─ SI Staging/            # Preprocessed staging data in pickle format for staging model (not included, in "dropbox")
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
│  ├─ REMSISD/               # REM data trained model for sleep decoding
│  └─ SIStaging/             # Sleep staging model
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
│  ├─ REMSISD.py             # Evaluation: sleep decoding on REM data
│  └─ SIStagingtest.py       # Evaluation: sleep staging
│
├─ train/
│  ├─ N23_finetune.py        # Fine‑tuning for N23SIMD model on NREM2/3 data
│  ├─ N23SIMD.py             # Training: NREM2/3 and awake data to NREM2/3 stage sleep decoder
│  ├─ N23SISD.py             # Training: NREM2/3 data to NREM2/3 stage sleep decoder
│  ├─ REM_finetune.py        # Fine‑tuning for REMSIMD model on REM data
│  ├─ REMSIMD.py             # Training: REM and awake data to REM stage sleep decoder
│  ├─ REMSISD.py             # Training: REM data to REM stage sleep decoder
│  └─ SIStaging.py           # Training: sleep staging model
│
└─ utils/
   ├─ __init__.py
   ├─ DotDict.py             # Simple dict helper
   └─ model.py               # Common model utilities
```
---

## Example data for staging label

Raw data from one train subject and one test subject were released at the following link to verify the precision of the staging label: [example_data](https://www.dropbox.com/scl/fo/117q6pzzhyh1m2ss2qef2/AOWz6zXSqRg30ETmXO--syQ?rlkey=7nmsmiogb3qpqrjxx9asd4bc8&st=swrviidk&dl=0)

---
## Setup

1. **Create the environment**
   ```bash
   conda env create -f environment/environment.yaml
   conda activate tensorflow
   ```

2. **Prepare data**
   - Place the data and model checkpoints in the "dropbox" link following the structure shown under `data/`.

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

# Sleep staging
python SIStaging.py
```
Checkpoints are saved under the directory `results/Decoding checkpoint/` and `results/Staging checkpoint/`.

### 3) Evaluate / Test
```bash
cd test
python N23SIMD.py
python N23SISD.py
python REMSIMD.py
python REMSISD.py
python N23SIMD_SO.py
python N23SIMD_retrained_SO.py
python SIStagingtest.py
```
Output metrics are saved under the directory `results/`.

