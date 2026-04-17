"""
MELD Preprocessing Script
- Concatenates previous utterances with special tokens
- Respects max_txt_len (128 for MELD)
- Saves tokenized data as .pt files

Input:  ../dataset/MELD.Raw/valid_label_csv/{split}_sent_emo.csv
CSV Output: ./csv_with_previous_utt_self_other_dynamic/3/{split}_sent_emo.csv
PT Output:  ../dataset/MELD.Raw/pt_with_previous_utt_self_other_dynamic/3/{split}_sent_emo.pt
"""

import csv
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer


def preprocess_dialogue_maxlen(df, tokenizer, max_txt_len=128, mode="self_other"):
    """
    Dynamic utterance concatenation respecting max_txt_len.

    Args:
        df: DataFrame with columns [Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion]
        tokenizer: RobertaTokenizer
        max_txt_len: maximum token length (128 for MELD)
        mode: speaker encoding mode ('self_other' or 'spk_idx')
    """
    processed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing utterances"):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        target_speaker = row['Speaker']

        # Collect all previous utterances in this dialogue
        prev_utts = df[(df['Dialogue_ID'] == dialogue_id) & (df['Utterance_ID'] < utterance_id)]

        # === Speaker map (for spk_idx mode) ===
        speaker_map = {}
        if mode == "spk_idx":
            speaker_map[target_speaker] = "<SPK0>"
            spk_counter = 1
            for spk in prev_utts['Speaker'].unique():
                if spk not in speaker_map:
                    speaker_map[spk] = f"<SPK{spk_counter}>"
                    spk_counter += 1

        # === Current utterance ===
        if mode == "spk_idx":
            curr_text = f"[Current] <SPK0> {row['Utterance']}"
        elif mode == "self_other":
            curr_text = f"[Current] <SELF> {row['Utterance']}"
        else:
            curr_text = f"[Current] {row['Utterance']}"

        combined = curr_text
        encoded_len = len(tokenizer.encode(combined, add_special_tokens=True))

        # === Add previous utterances (from most recent, stop if exceeds max_txt_len) ===
        for _, prev_row in reversed(list(prev_utts.iterrows())):
            spk = prev_row['Speaker']
            utt = prev_row['Utterance']

            if mode == "spk_idx":
                spk_token = speaker_map.get(spk, "<SPK_UNK>")
                prev_seg = f"{spk_token} {utt}"
            elif mode == "self_other":
                spk_token = "<SELF>" if spk == target_speaker else "<OTHER>"
                prev_seg = f"{spk_token} {utt}"
            else:
                prev_seg = f"{utt}"

            candidate = f"{prev_seg} </s></s> {combined}"
            cand_len = len(tokenizer.encode(candidate, add_special_tokens=True))

            if cand_len <= max_txt_len:
                combined = candidate
                encoded_len = cand_len
            else:
                break  # Stop if exceeds max_txt_len

        row_out = row.copy()
        row_out['Utterance'] = combined.strip()
        processed_rows.append(row_out)

    return pd.DataFrame(processed_rows)


def save_split_as_pt(csv_path, pt_path, tokenizer, max_txt_len=128):
    """
    Read CSV and save as .pt file with tokenized data.

    Args:
        csv_path: path to CSV file
        pt_path: output .pt file path
        tokenizer: RobertaTokenizer
        max_txt_len: maximum token length
    """
    df = pd.read_csv(csv_path)
    dataset_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {csv_path.name}"):
        utt_id = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
        text = row['Utterance']
        label = row['Emotion']

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_len,
            return_tensors="pt"
        )

        dataset_dict[utt_id] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label
        }

    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset_dict, pt_path)
    print(f"✓ Saved {len(dataset_dict)} samples to {pt_path}")


if __name__ == "__main__":
    # Configuration
    MAX_TXT_LEN = 128
    SPEAKER_MODE = "self_other"  # or "spk_idx"
    MODEL_KEY = "roberta-base"
    K = 3  # Number of previous utterances to consider (dynamic within max_txt_len)

    # Paths
    INPUT_DIR = Path("../dataset/MELD.Raw/valid_label_csv")
    CSV_OUTPUT_DIR = Path(f"./csv_with_previous_utt_{SPEAKER_MODE}_dynamic/{K}")
    PT_OUTPUT_DIR = Path(f"../dataset/MELD.Raw/pt_with_previous_utt_{SPEAKER_MODE}_dynamic/{K}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_KEY)

    # Add special tokens
    special_tokens = ['[Current]']
    if SPEAKER_MODE == "spk_idx":
        special_tokens += [f"<SPK{i}>" for i in range(9)]
    elif SPEAKER_MODE == "self_other":
        special_tokens += ["<SELF>", "<OTHER>"]

    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added special tokens: {special_tokens}")
    print(f"Vocab size: {len(tokenizer)}")

    # Process each split
    for split in ['train', 'dev', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")

        # Step 1: Read original CSV
        input_csv = INPUT_DIR / f"{split}_sent_emo.csv"
        if not input_csv.exists():
            print(f"⚠ Warning: {input_csv} not found, skipping...")
            continue

        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} samples from {input_csv}")

        # Step 2: Create CSV with concatenated previous utterances
        new_df = preprocess_dialogue_maxlen(
            df,
            tokenizer=tokenizer,
            max_txt_len=MAX_TXT_LEN,
            mode=SPEAKER_MODE
        )

        # Save intermediate CSV
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_output_path = CSV_OUTPUT_DIR / f"{split}_sent_emo.csv"
        new_df.to_csv(csv_output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"✓ Saved CSV to {csv_output_path}")

        # Step 3: Tokenize and save as .pt
        pt_output_path = PT_OUTPUT_DIR / f"{split}_sent_emo.pt"
        save_split_as_pt(csv_output_path, pt_output_path, tokenizer, max_txt_len=MAX_TXT_LEN)

    print(f"\n{'='*60}")
    print("✓ MELD preprocessing completed!")
    print(f"{'='*60}")
