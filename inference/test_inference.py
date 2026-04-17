"""
Test set inference script for IEMOCAP emotion recognition
Runs inference on the entire test set and saves results to CSV
"""

import json
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
import sys
import os

# Add paths
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from inference import EmotionInference
from dataloader.dataloader import load_finetune_audios

LABEL_SETS = {
    "iemocap6": {
        0: "neu",
        1: "sad",
        2: "fru",
        3: "ang",
        4: "hap",
        5: "exc",
    },
    "meld7": {
        0: "neutral",
        1: "sadness",
        2: "anger",
        3: "joy",
        4: "surprise",
        5: "fear",
        6: "disgust",
    },
    "meld": {
        0: "neutral",
        1: "sadness",
        2: "anger",
        3: "joy",
    },
}


def get_emotion_labels(dataset):
    if dataset not in LABEL_SETS:
        raise ValueError(f"Unsupported dataset for labels: {dataset}")
    return LABEL_SETS[dataset]


def load_text_csv(csv_path, dataset):
    """
    Load text CSV and create instance_id to text mapping

    Args:
        csv_path: Path to CSV file containing texts
        dataset: Dataset name

    Returns:
        Dictionary mapping instance_id to text
    """
    if not Path(csv_path).exists():
        print(f"Warning: Text CSV not found at {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    text_map = {}

    for _, row in df.iterrows():
        if dataset in ['iemocap', 'iemocap6']:
            instance_id = row['Speaker']
        else:
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            instance_id = f"dia{dialogue_id}_utt{utterance_id}"

        # Extract text after [Current] if present
        full_text = str(row['Utterance'])
        if '[Current]' in full_text:
            # Split by [Current] and take the part after it
            current_text = full_text.split('[Current]')[1].strip()
            # Remove speaker name if present (format: "Speaker: text")
            if ':' in current_text:
                current_text = current_text.split(':', 1)[1].strip()
            text_map[instance_id] = current_text
        else:
            text_map[instance_id] = full_text

    return text_map



def run_test_inference(args):
    """
    Run inference on test set and save results to CSV

    Args:
        args: Command line arguments
    """
    print("="*70)
    print(f"{args.dataset.upper()} Test Set Inference")
    print("="*70)

    # Load config
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    split_dir = str(Path(config["project_dir"]).joinpath(args.split_data_dir))
    audio_dir = str(Path(config["project_dir"]).joinpath("audio"))

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Fold: {args.fold_idx}")
    print(f"  Split dir: {split_dir}")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Device: {args.device}")

    # Load text CSV for extracting actual text
    text_map = {}
    if hasattr(args, 'text_csv_path') and args.text_csv_path:
        print(f"  Text CSV: {args.text_csv_path}")
        text_map = load_text_csv(args.text_csv_path, args.dataset)
        print(f"  Loaded {len(text_map)} text entries")

    # Load test file list
    print(f"\n{'='*70}")
    print("Loading test data...")
    print(f"{'='*70}")

    train_file_list, dev_file_list, test_file_list = load_finetune_audios(
        split_dir,
        audio_path=audio_dir,
        dataset=args.dataset,
        fold_idx=args.fold_idx
    )

    print(f"\nTest set size: {len(test_file_list)}")

    # Initialize inference model
    print(f"\n{'='*70}")
    print("Loading model...")
    print(f"{'='*70}\n")

    inference_model = EmotionInference(
        model_dir=args.model_dir,
        fold_idx=args.fold_idx,
        device=args.device,
        dataset=args.dataset,
        modal=args.modal
    )

    print(f"\n{'='*70}")
    print("Running inference on test set...")
    print(f"{'='*70}\n")

    # Prepare results list
    results = []

    # Run inference
    emotion_labels = get_emotion_labels(args.dataset)

    for idx, data_item in enumerate(tqdm(test_file_list, desc="Processing test samples")):
        # Extract data
        instance_id = data_item[0]  # Speaker ID or dialogue ID
        speaker_id = data_item[1]
        file_name = data_item[2]
        audio_path = data_item[3]
        text_path = data_item[4]
        speaker_idx = data_item[5]
        ground_truth_idx = data_item[-1]
        ground_truth_label = emotion_labels[ground_truth_idx]

        # Load text
        # Check if text_path is PT file (tokenized) or CSV file (raw text)
        if text_path.endswith('.pt'):
            # Check if text path exists
            if not Path(text_path).exists():
                print(f"\nWarning: Text file not found: {text_path}")

            # Load PT file containing tokenized texts
            try:
                if not hasattr(run_test_inference, 'pt_data_cache'):
                    run_test_inference.pt_data_cache = {}

                if text_path not in run_test_inference.pt_data_cache:
                    run_test_inference.pt_data_cache[text_path] = torch.load(text_path)

                pt_data = run_test_inference.pt_data_cache[text_path]

                # Extract text from PT data
                if instance_id in pt_data:
                    utt_dict = pt_data[instance_id]
                    # PT files contain tokenized data (input_ids, attention_mask)
                    # Use these directly for inference
                    text = (utt_dict['input_ids'], utt_dict['attention_mask'])
                else:
                    text = ""
            except Exception as e:
                print(f"\nWarning: Could not load PT file {text_path}: {e}")
                text = ""
        elif text_path.endswith('.csv'):
            # Load CSV file
            if args.dataset in ['iemocap', 'iemocap6']:
                files = pd.read_csv(text_path)
                text = files[files['Speaker'] == instance_id]['Utterance'].values.tolist()
                if len(text) > 0:
                    text = text[0]
                else:
                    text = ""
            elif args.dataset in ['meld', 'meld7']:
                files = pd.read_csv(text_path)
                files['dia_utt'] = files.apply(
                    lambda row: f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}", axis=1
                )
                text = files[files['dia_utt'] == instance_id]['Utterance'].values.tolist()
                if len(text) > 0:
                    text = text[0]
                else:
                    text = ""
        else:
            text = ""

        if args.modal == "text":
            try:
                pred_label, probabilities = inference_model.predict(
                    audio_path=None,
                    text=text,
                    return_probabilities=True
                )
                pred_idx = [k for k, v in emotion_labels.items() if v == pred_label][0]
            except Exception as e:
                print(f"\nError processing {instance_id}: {e}")
                pred_idx = -1
                pred_label = "ERROR"
                probabilities = {label: 0.0 for label in emotion_labels.values()}
        else:
            # Check if audio file exists
            if not Path(audio_path).exists():
                print(f"\nWarning: Audio file not found: {audio_path}")
                pred_idx = -1
                pred_label = "NOT_FOUND"
                probabilities = {label: 0.0 for label in emotion_labels.values()}
            else:
                try:
                    # Run inference
                    pred_label, probabilities = inference_model.predict(
                        audio_path=audio_path,
                        text=text,
                        return_probabilities=True
                    )

                    # Get predicted index
                    pred_idx = [k for k, v in emotion_labels.items() if v == pred_label][0]

                except Exception as e:
                    print(f"\nError processing {instance_id}: {e}")
                    pred_idx = -1
                    pred_label = "ERROR"
                    probabilities = {label: 0.0 for label in emotion_labels.values()}

        # Store results
        # Handle text display
        # If text_map is available, use it; otherwise use the original text or placeholder
        if instance_id in text_map:
            text_display = text_map[instance_id]
        elif isinstance(text, str):
            text_display = text
        else:
            text_display = "[Tokenized data]"

        result = {
            'instance': instance_id,
            # 'audio_path': audio_path,
            'text': text_display,
            'ground_truth_idx': ground_truth_idx,
            'pred_idx': pred_idx,
            'ground_truth_label': ground_truth_label,
            'pred_label': pred_label,
            'correct': int(pred_idx == ground_truth_idx),
        }

        # Add probability for each emotion
        # for emotion_idx, emotion_label in EMOTION_LABELS.items():
        #     result[f'prob_{emotion_label}'] = probabilities.get(emotion_label, 0.0)

        results.append(result)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Filter out error/not found samples for metric calculation
    max_label_idx = len(emotion_labels) - 1
    valid_df = df_results[(df_results['pred_idx'] >= 0) & (df_results['pred_idx'] <= max_label_idx)]

    # Calculate metrics
    accuracy = (valid_df['correct'] == 1).mean() * 100

    # Calculate macro F1 and UAR using sklearn
    y_true = valid_df['ground_truth_idx'].values
    y_pred = valid_df['pred_idx'].values

    mf1 = f1_score(y_true, y_pred, average='weighted') * 100
    uar = recall_score(y_true, y_pred, average='macro') * 100

    # Print summary
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    print(f"\nTotal samples: {len(df_results)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {mf1:.2f}%")
    print(f"UAR (Unweighted Average Recall): {uar:.2f}%")
    print(f"\nCorrect predictions: {valid_df['correct'].sum()}")
    print(f"Incorrect predictions: {len(valid_df) - valid_df['correct'].sum()}")

    # Calculate per-class metrics
    print(f"\n{'='*70}")
    print("Per-class Metrics")
    print(f"{'='*70}")
    print(f"{'Class':10s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Correct/Total':>15s}")
    print(f"{'-'*70}")

    per_class_metrics = {}
    for idx, label in emotion_labels.items():
        # Ground truth samples for this class
        class_df = valid_df[valid_df['ground_truth_idx'] == idx]
        # Predicted samples for this class
        pred_df = valid_df[valid_df['pred_idx'] == idx]

        if len(class_df) > 0:
            # Accuracy (recall for this class)
            class_acc = (class_df['correct'] == 1).mean() * 100
            correct = class_df['correct'].sum()
            total = len(class_df)

            # True Positives, False Positives, False Negatives
            tp = len(valid_df[(valid_df['ground_truth_idx'] == idx) & (valid_df['pred_idx'] == idx)])
            fp = len(valid_df[(valid_df['ground_truth_idx'] != idx) & (valid_df['pred_idx'] == idx)])
            fn = len(valid_df[(valid_df['ground_truth_idx'] == idx) & (valid_df['pred_idx'] != idx)])

            # Precision
            precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0

            # Recall (same as class accuracy)
            recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

            # F1 Score
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            print(f"{label:10s} {class_acc:6.2f} {precision:6.2f} {recall:6.2f} {f1:6.2f} {correct:6d}/{total:<6d}")

            per_class_metrics[label] = {
                'accuracy': float(class_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'correct': int(correct),
                'total': int(total),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }

    # Save to CSV
    output_path = Path(args.output_dir).joinpath(
        f"{args.dataset}_fold{args.fold_idx}_test_results.csv"
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")

    # Also save a summary file
    summary = {
        'dataset': args.dataset,
        'fold': args.fold_idx,
        'total_samples': len(df_results),
        'valid_samples': len(valid_df),
        'acc': float(accuracy),
        'mf1': float(mf1),
        'uar': float(uar),
        'accuracy': float(accuracy),  # Keep for backward compatibility
        'correct': int(valid_df['correct'].sum()),
        'incorrect': int(len(valid_df) - valid_df['correct'].sum()),
        'per_class_accuracy': {}
    }

    # Use the already calculated per_class_metrics
    summary['per_class_metrics'] = per_class_metrics

    # Keep per_class_accuracy for backward compatibility
    for label, metrics in per_class_metrics.items():
        summary['per_class_accuracy'][label] = {
            'accuracy': metrics['accuracy'],
            'correct': metrics['correct'],
            'total': metrics['total']
        }

    summary_path = Path(args.output_dir).joinpath(
        f"{args.dataset}_fold{args.fold_idx}_test_summary.json"
    )
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}\n")

    return df_results, summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='IEMOCAP/MELD Test Set Inference')

    # Model arguments
    parser.add_argument('--model_dir', type=str,
                        default='finetune/iemocap6/multimodal/best',
                        help='Directory containing model files')
    parser.add_argument('--fold_idx', type=int, default=1,
                        help='Fold index')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--modal', type=str, default='multimodal',
                        help='Modality (multimodal, text, audio)')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='iemocap6',
                        help='Dataset name (iemocap6, iemocap, meld7, meld, etc.)')
    parser.add_argument('--split_data_dir', type=str,
                        default='train_split/train_split_pt_self_other_256',
                        help='Directory containing split data')
    parser.add_argument('--text_csv_path', type=str, default=None,
                        help='Path to CSV file containing actual text (to extract text after [Current])')

    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='test_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Run inference
    df_results, summary = run_test_inference(args)

    print("Done!")


if __name__ == '__main__':
    main()
