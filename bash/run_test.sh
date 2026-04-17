#!/bin/bash

# IEMOCAP
# This script runs inference on the MELD test set and saves results to CSV
cd ../inference
# Set variables (paths relative to parent directory)
MODEL_DIR="../finetune/iemocap6/multimodal/pretrained"
FOLD_IDX=1
DATASET="iemocap6"
SPLIT_DATA_DIR="train_split/train_split_iemocap"
TEXT_CSV_PATH="../dataset/IEMOCAP_full_release/previous_utt_csv/previous_current_5/IEMOCAP_test.csv"
OUTPUT_DIR="../test_results/mm_test_iemocap"

DEVICE="cuda"

# Run inference
python test_inference.py \
    --model_dir ${MODEL_DIR} \
    --fold_idx ${FOLD_IDX} \
    --dataset ${DATASET} \
    --modal multimodal \
    --split_data_dir ${SPLIT_DATA_DIR} \
    --text_csv_path ${TEXT_CSV_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE}

echo ""
echo "================================================================"
echo "Test inference completed!"
echo "Results saved to: ${OUTPUT_DIR}/${DATASET}_fold${FOLD_IDX}_test_results.csv"
echo "================================================================"

# MELD
MODEL_DIR="../finetune/meld7/pretrained"
FOLD_IDX=1
DATASET="meld7"
SPLIT_DATA_DIR="train_split/train_split_meld"
TEXT_CSV_PATH="../dataset/MELD.Raw/test_sent_emo.csv"
OUTPUT_DIR="../test_results/mm_test_meld"
DEVICE="cuda"

# Run inference
python test_inference.py \
    --model_dir ${MODEL_DIR} \
    --fold_idx ${FOLD_IDX} \
    --dataset ${DATASET} \
    --split_data_dir ${SPLIT_DATA_DIR} \
    --text_csv_path ${TEXT_CSV_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE}

echo ""
echo "================================================================"
echo "Test inference completed!"
echo "Results saved to: ${OUTPUT_DIR}/${DATASET}_fold${FOLD_IDX}_test_results.csv"
echo "================================================================"
