import json
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torchaudio
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add paths
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from model.wav2vec import Wav2VecWrapper
from model.wavlm_plus import WavLMWrapper
from model.custom_roberta import RobertaCrossAttn
from model.prediction import TextAudioClassifierForCrossModalAttn, TextAudioClassifier
from transformers import RobertaTokenizer
from utils import tokenize_texts

# Model hidden dimensions
hid_dim_dict = {
    "wav2vec2_0": 768,
    "wav2vec2_0-large": 1024,
    "roberta-base": 768,
    "wavlm": 768,
    "wavlm-large": 1024,
}

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

class EmotionInference:
    def __init__(self, model_dir, fold_idx=1, device='cuda', dataset='iemocap6', modal=None):
        """
        Initialize inference model for emotion recognition

        Args:
            model_dir: Directory containing saved model files
            fold_idx: Fold index (default: 1)
            device: Device to run inference on ('cuda' or 'cpu')
            dataset: Dataset name (iemocap6, meld7, meld)
            modal: Modality (multimodal, text, audio)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fold_idx = fold_idx
        self.model_dir = Path(model_dir)
        self.dataset = dataset
        self.emotion_labels = get_emotion_labels(dataset)

        # Set up arguments (matching training configuration)
        self.args = self._setup_args(dataset, modal)
        self._autodetect_model_args()

        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.args.text_model)

        # Build model
        self.model = self._build_model()

        # Load weights
        self._load_weights()

        # Set to eval mode
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    def _setup_args(self, dataset, modal):
        """Setup arguments matching training configuration"""
        args = argparse.Namespace()

        if dataset in ["iemocap", "iemocap6"]:
            # Model configuration
            args.audio_model = "wavlm"
            args.text_model = "roberta-base"
            args.dataset = "iemocap6"
            args.modal = "multimodal"
            args.cross_modal_atten = False
            args.hidden_dim = 256
            args.dr = 0.5
            args.speaker = "None"
            args.speaker_dim = 10
            args.multimodal_pooling = "cls"

            # Audio model configuration
            args.finetune_audio = True
            args.adapter_hidden_dim = 128
            args.embedding_prompt_dim = 5
            args.lora_rank = 16
            args.downstream = False

            # Text model configuration
            args.max_txt_len = 256
            args.ws = False
            args.self_attn = False
            args.pooling_mode = "curr_only"
            args.clamp = 1e-6
            args.tokenize_mode = "default"
            args.load_pt = False  # Default: False, will be dynamically set based on input type
            args.num_hidden_layers = None
            args.finetune_roberta = True
            args.lora_target_modules = "full"
            args.speaker_mode = "self_other"
            args.truncation_side = "right"
            args.padding = None
        else:
            # Model configuration (MELD defaults)
            args.audio_model = "wav2vec2_0"
            args.text_model = "roberta-base"
            args.dataset = dataset
            args.modal = "multimodal"
            args.cross_modal_atten = True
            args.hidden_dim = 256
            args.dr = 0.5
            args.speaker_dim = 10
            args.multimodal_pooling = 'curr_only'

            # Audio model configuration
            args.finetune_audio = True
            args.adapter_hidden_dim = 128
            args.embedding_prompt_dim = 5
            args.lora_rank = 8
            args.downstream = False

            # Text model configuration
            args.max_txt_len = 128
            args.ws = False
            args.self_attn = False
            args.pooling_mode = "curr_only"
            args.clamp = False
            args.tokenize_mode = "default"
            args.load_pt = False  # Default: False, will be dynamically set based on input type
            args.num_hidden_layers = None
            args.finetune_roberta = False
            args.lora_target_modules = "none"
            args.speaker_mode = "self_other"
            args.truncation_side = "right"
            args.padding = "max_length"

        if modal is not None:
            args.modal = modal
            if modal == "text":
                args.cross_modal_atten = False
            elif modal == "audio":
                args.cross_modal_atten = False

        return args

    def _build_model(self):
        """Build model architecture"""
        num_class = len(self.emotion_labels)

        audio_model = None
        text_model = None
        audio_dim = None
        text_dim = None

        if self.args.modal in ["multimodal", "audio"]:
            if self.args.audio_model in ["wavlm", "wavlm-large"]:
                audio_model = WavLMWrapper(self.args).to(self.device)
            else:
                audio_model = Wav2VecWrapper(self.args).to(self.device)
            audio_dim = hid_dim_dict[self.args.audio_model]

        if self.args.modal in ["multimodal", "text"]:
            text_model = RobertaCrossAttn(self.args, audio_model).to(self.device)
            text_dim = hid_dim_dict[self.args.text_model]

        if self.args.modal == "multimodal":
            model = TextAudioClassifierForCrossModalAttn(
                audio_model=audio_model,
                text_model=text_model,
                audio_dim=audio_dim,
                text_dim=text_dim,
                hidden_dim=self.args.hidden_dim,
                num_classes=num_class,
                dropout_prob=self.args.dr,
                cross_modal_atten=self.args.cross_modal_atten,
                modal=self.args.modal,
                multimodal_pooling=self.args.multimodal_pooling
            ).to(self.device)
        else:
            model = TextAudioClassifier(
                audio_model=audio_model,
                text_model=text_model,
                audio_dim=audio_dim,
                text_dim=text_dim,
                hidden_dim=self.args.hidden_dim,
                num_classes=num_class,
                dropout_prob=self.args.dr,
                cross_modal_atten=self.args.cross_modal_atten,
                modal=self.args.modal
            ).to(self.device)

        return model

    def _autodetect_model_args(self):
        """Override args based on model files when possible."""
        model_files = [p.name for p in self.model_dir.glob("*.pt")]
        if not model_files:
            return

        audio_candidates = ["wavlm-large", "wavlm", "wav2vec2_0-large", "wav2vec2_0"]
        for candidate in audio_candidates:
            if any(candidate in name for name in model_files):
                if self.args.audio_model != candidate:
                    print(f"Detected audio model '{candidate}' from checkpoints.")
                    self.args.audio_model = candidate
                break

    def _load_weights(self):
        """Load model weights from saved files"""
        # Find model files
        model_files = list(self.model_dir.glob(f"*_fold_{self.fold_idx}.pt"))

        if len(model_files) == 0:
            raise FileNotFoundError(f"No model files found for fold {self.fold_idx} in {self.model_dir}")

        # Always load individual components to avoid vocab size mismatch
        print("Loading model components separately...")

        # Load audio model
        if self.args.modal in ["multimodal", "audio"]:
            audio_model_file = [f for f in model_files if self.args.audio_model in f.name]
            if audio_model_file:
                print(f"Loading audio model from {audio_model_file[0]}")
                state_dict = torch.load(audio_model_file[0], map_location=self.device, weights_only=False)
                self.model.audio_model.load_state_dict(state_dict, strict=False)

        # Load text model
        text_model_file = [f for f in model_files if "roberta-base" in f.name]
        if text_model_file and self.args.modal in ["multimodal", "text"]:
            print(f"Loading text model from {text_model_file[0]}")
            state_dict = torch.load(text_model_file[0], map_location=self.device, weights_only=False)

            # Get saved vocab size from checkpoint
            saved_vocab_size = state_dict['semantic_model.embeddings.word_embeddings.weight'].shape[0]
            current_vocab_size = self.model.text_model.semantic_model.embeddings.word_embeddings.weight.shape[0]

            if saved_vocab_size != current_vocab_size:
                print(f"   Adjusting vocab size from {current_vocab_size} to {saved_vocab_size}")
                # Resize token embeddings to match saved model
                self.model.text_model.semantic_model.resize_token_embeddings(saved_vocab_size)

            self.model.text_model.load_state_dict(state_dict, strict=False)

        # Load prediction layer
        pred_file = [f for f in model_files if "pred" in f.name]
        if pred_file:
            print(f"Loading prediction layer from {pred_file[0]}")
            state_dict = torch.load(pred_file[0], map_location=self.device, weights_only=False)
            self.model.pred_linear.load_state_dict(state_dict, strict=False)

    def preprocess_audio(self, audio_path, max_length=6):
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file
            max_length: Maximum audio length in seconds

        Returns:
            audio tensor and length
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Take first channel if stereo
        if audio.shape[0] > 1:
            audio = audio[0:1, :]

        audio = audio.squeeze(0)

        # Crop if too long
        max_samples = max_length * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        length = torch.tensor(len(audio))

        # Pad to max length
        if len(audio) < max_samples:
            audio = torch.nn.ConstantPad1d(padding=(0, max_samples - len(audio)), value=0)(audio)

        return audio.unsqueeze(0), length.unsqueeze(0)

    def preprocess_text(self, text):
        """
        Preprocess text input

        Args:
            text: Input text string, list of text strings, or tuple of (input_ids, attention_mask)

        Returns:
            Tokenized text or tuple for tokenized data
        """
        # Check if text is already tokenized (tuple of tensors)
        if isinstance(text, tuple) and len(text) == 2:
            # Already tokenized data from PT file
            # Need to ensure tensors have batch dimension
            input_ids, attention_mask = text
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            return (input_ids, attention_mask)
        elif isinstance(text, str):
            text = [text]

        return text

    @torch.no_grad()
    def predict(self, audio_path, text, return_probabilities=False):
        """
        Perform emotion prediction

        Args:
            audio_path: Path to audio file
            text: Input text (string or list)
            return_probabilities: If True, return class probabilities

        Returns:
            Predicted emotion label and optionally probabilities
        """
        # Preprocess inputs
        if self.args.modal == "text":
            audio = None
            length = None
        else:
            audio, length = self.preprocess_audio(audio_path)
            audio = audio.to(self.device)
            length = length.to(self.device)

        text_input = self.preprocess_text(text)
        speaker_id = torch.tensor([0]).to(self.device) if self.args.modal != "text" else None

        # Dynamically set load_pt based on input type
        is_tokenized = isinstance(text_input, tuple) and len(text_input) == 2
        original_load_pt = self.model.text_model.load_pt
        if is_tokenized:
            self.model.text_model.load_pt = True

        # Forward pass
        outputs, _, _, _ = self.model(
            audio_input=audio,
            text_input=text_input,
            # speaker_ID=speaker_id,
            length=length
        )

        # Restore original load_pt value
        self.model.text_model.load_pt = original_load_pt

        # Get predictions
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_emotion = self.emotion_labels[predicted_class]

        if return_probabilities:
            probs_dict = {
                self.emotion_labels[i]: probabilities[0, i].item()
                for i in range(len(self.emotion_labels))
            }
            return predicted_emotion, probs_dict
        else:
            return predicted_emotion

    @torch.no_grad()
    def predict_batch(self, data_list, batch_size=8):
        """
        Perform batch prediction

        Args:
            data_list: List of tuples (audio_path, text)
            batch_size: Batch size for inference

        Returns:
            List of predictions
        """
        results = []

        for i in tqdm(range(0, len(data_list), batch_size), desc="Processing batches"):
            batch_data = data_list[i:i+batch_size]

            batch_texts = []

            for audio_path, text in batch_data:
                batch_texts.append(text)

            if self.args.modal == "text":
                batch_audios = None
                batch_lengths = None
                speaker_ids = None
            else:
                batch_audios = []
                batch_lengths = []
                for audio_path, _ in batch_data:
                    audio, length = self.preprocess_audio(audio_path)
                    batch_audios.append(audio)
                    batch_lengths.append(length)
                batch_audios = torch.cat(batch_audios, dim=0).to(self.device)
                batch_lengths = torch.cat(batch_lengths, dim=0).to(self.device)
                speaker_ids = torch.zeros(len(batch_data), dtype=torch.long).to(self.device)

            # Forward pass
        outputs, _, _, _ = self.model(
            audio_input=batch_audios,
            text_input=batch_texts,
            speaker_ID=speaker_ids,
            length=batch_lengths
        )

        # Get predictions
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

        for j, pred_class in enumerate(predicted_classes):
            pred_emotion = self.emotion_labels[pred_class]
            probs_dict = {
                self.emotion_labels[k]: probabilities[j, k].item()
                for k in range(len(self.emotion_labels))
            }
            results.append({
                'audio_path': batch_data[j][0],
                'text': batch_data[j][1],
                'predicted_emotion': pred_emotion,
                'probabilities': probs_dict
            })

        return results


def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model files')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Path to audio file (required unless --modal text)')
    parser.add_argument('--text', type=str, required=True,
                        help='Text transcription')
    parser.add_argument('--fold_idx', type=int, default=1,
                        help='Fold index (default: 1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--dataset', type=str, default='iemocap6',
                        help='Dataset name (iemocap6, meld7, meld)')
    parser.add_argument('--modal', type=str, default='multimodal',
                        help='Modality (multimodal, text, audio)')

    args = parser.parse_args()

    # Initialize inference model
    print("Initializing model...")
    if args.modal != "text" and not args.audio_path:
        raise ValueError("--audio_path is required unless --modal text")

    inference_model = EmotionInference(
        model_dir=args.model_dir,
        fold_idx=args.fold_idx,
        device=args.device,
        dataset=args.dataset,
        modal=args.modal
    )

    # Perform prediction
    print("\nPerforming inference...")
    predicted_emotion, probabilities = inference_model.predict(
        audio_path=args.audio_path,
        text=args.text,
        return_probabilities=True
    )

    # Print results
    print("\n" + "="*50)
    print(f"Audio: {args.audio_path}")
    print(f"Text: {args.text}")
    print(f"\nPredicted Emotion: {predicted_emotion}")
    print("\nProbabilities:")
    for emotion, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion:10s}: {prob:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
