# EMART: Emotion Recognition with Multimodal Audio-Text Representation

This repository contains the implementation of EMART, a multimodal emotion recognition model that leverages both audio and text modalities for conversational emotion recognition.

**Note:** This implementation is based on and modified from the [PEFT-SER](https:/AnonymousAnonymous/github.com/usc-sail/peft-ser) codebase.

## Table of Contents
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Testing/Inference](#testinginference)
- [Project Structure](#project-structure)

---

## Quick Start

### Reproduce Main Results
To reproduce the main results reported in our paper, simply run:
```bash
bash bash/run_test.sh
```

This script will:
- Run inference on both IEMOCAP and MELD test sets using our pretrained models
- Generate prediction results and save them as CSV files
- Calculate and display evaluation metrics (Weighted Accuracy, Unweighted Accuracy, Macro F1)

---

## Dataset Setup

Before training or testing, you need to download the datasets and place them in the correct directory structure.

**Required Directory Structure:**
```
ERC/
├── EMART/              # Model code (this repository)
└── dataset/            # Dataset directory (sibling to EMART)
    ├── IEMOCAP_full_release/
    │   └── telme_data/
    │       ├── IEMOCAP_train.csv
    │       ├── IEMOCAP_dev.csv
    │       └── IEMOCAP_test.csv
    └── MELD.Raw/
        └── valid_label_csv/
            ├── train_sent_emo.csv
            ├── dev_sent_emo.csv
            └── test_sent_emo.csv
```

**Important:** The dataset folder should be at `../dataset` relative to the EMART folder.

---

## Data Preprocessing

Before training, you need to preprocess the text data by concatenating previous utterances with special tokens and tokenizing them.

### Step 1: Navigate to Preprocessing Directory
```bash
cd train_split_gen
```

### Step 2: Run Preprocessing Scripts

#### For IEMOCAP (max_txt_len=256):
```bash
python iemocap_preprocessing.py
```

**What this does:**
- Reads raw CSV files from `../dataset/IEMOCAP_full_release/telme_data/`
- Concatenates previous utterances with special tokens (`[Current]`, `<SELF>`, `<OTHER>`)
- Dynamically fits utterances within max_txt_len (256 tokens)
- Tokenizes using RoBERTa tokenizer
- Saves intermediate CSV to `./previous_utt_csv/previous_current_dynamic_self_other_256/`
- Saves final tokenized `.pt` files to:
  ```
  ../dataset/IEMOCAP_full_release/previous_utt_pt/previous_current_dynamic_self_other_256/
  ├── IEMOCAP_train.pt
  ├── IEMOCAP_dev.pt
  └── IEMOCAP_test.pt
  ```

#### For MELD (max_txt_len=128):
```bash
python meld_preprocessing.py
```

**What this does:**
- Reads raw CSV files from `../dataset/MELD.Raw/valid_label_csv/`
- Concatenates previous utterances with special tokens (`[Current]`, `<SELF>`, `<OTHER>`)
- Dynamically fits utterances within max_txt_len (128 tokens)
- Tokenizes using RoBERTa tokenizer
- Saves intermediate CSV to `./csv_with_previous_utt_self_other_dynamic/3/`
- Saves final tokenized `.pt` files to:
  ```
  ../dataset/MELD.Raw/pt_with_previous_utt_self_other_dynamic/3/
  ├── train_sent_emo.pt
  ├── dev_sent_emo.pt
  └── test_sent_emo.pt
  ```

---

## Training

To train the model from scratch:

```bash
bash bash/run_train.sh
```

**Training Configuration:**
- **Datasets:** IEMOCAP-6, MELD-7
- **Audio Models:** Wav2Vec2.0, WavLM
- **Text Model:** RoBERTa-base
- **Batch Size:** 64 (IEMOCAP), 32 (MELD)
- **Learning Rate:** 2e-5
- **Epochs:** 20
The training script will automatically:
- Load preprocessed `.pt` files
- Train multimodal fusion model
- Save checkpoints to `finetune/{dataset}/multimodal/`
- Log metrics to WandB (if configured)

---

## Testing/Inference

### Using Bash Script
```bash
bash bash/run_test.sh
```

### Manual Inference
```bash
cd inference
python test_inference.py \
    --model_dir ../finetune/iemocap6/multimodal/pretrained \
    --fold_idx 1 \
    --dataset iemocap6 \
    --modal multimodal \
    --split_data_dir train_split/train_split_iemocap \
    --text_csv_path ../dataset/IEMOCAP_full_release/previous_utt_csv/previous_current_5/IEMOCAP_test.csv \
    --output_dir ../test_results/mm_test_iemocap \
    --device cuda
```

**Output:**
Results will be saved as CSV files in the specified output directory.

---

## Project Structure

```
EMART/
├── bash/                       # Shell scripts for training and testing
│   ├── run_train.sh           # Training script
│   └── run_test.sh            # Inference script
│
├── config/                     # Configuration files
│   ├── config.py              # Python config
│   └── config.yml             # YAML config
│
├── dataloader/                 # Data loading utilities
│   ├── __init__.py
│   └── dataloader.py          # Custom dataset and dataloader implementations
│
├── experiment/                 # Training and evaluation
│   ├── finetune.py            # Main training script
│   ├── evaluation.py          # Evaluation metrics (WA, UA, mF1, etc.)
│   └── wandb/                 # WandB logging directory
│
├── inference/                  # Inference and testing
│   ├── inference.py           # Inference wrapper class
│   └── test_inference.py      # Test script for batch inference
│
├── model/                      # Model architectures
│   ├── wav2vec.py             # Wav2Vec2.0 wrapper
│   ├── wavlm_plus.py          # WavLM wrapper
│   ├── custom_roberta.py      # Custom RoBERTa with cross-attention
│   └── prediction.py          # Multimodal fusion and prediction heads
│
├── train_split_gen/            # Data preprocessing scripts
│   ├── iemocap_preprocessing.py  # IEMOCAP preprocessing
│   ├── meld_preprocessing.py     # MELD preprocessing

├── utils/                      # Utility functions
│   ├── __init__.py
│   └── utils.py               # Loss functions, metrics, argument parsers
│
├── finetune/                   # Saved model checkpoints
│   ├── iemocap6/
│   └── meld7/
│
├── test_results/               # Inference results
│
└── README.md                   # This file
```

## Requirements

```
torch>=1.10.0
transformers>=4.20.0
torchaudio
pandas
numpy
tqdm
scikit-learn
wandb (optional, for experiment tracking)
```

