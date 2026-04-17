# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import os
import pdb
import copy
import torch
import argparse
import numpy as np
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from functools import lru_cache
from torchaudio.compliance import kaldi

from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, AutoProcessor, WavLMModel, WhisperModel, AutoFeatureExtractor


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        i
    ):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        self.i = i

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states) 
        
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Wav2VecWrapper(nn.Module):
    def __init__(
        self, 
        args, 
        hidden_dim=256,
        output_class_num=4
    ):
        super(Wav2VecWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.args = args
        if args.audio_model == 'wav2vec2_0': 
            model_name = "facebook/wav2vec2-base-960h"
        elif args.audio_model == 'wav2vec2_0-large':
            model_name = "facebook/wav2vec2-large-960h"
            
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True
        )   
        state_dict = self.backbone_model.state_dict()

        # 2. Read the model config
        self.model_config                 = self.backbone_model.config
        self.model_config.finetune_audio  = self.args.finetune_audio
    
        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList([Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers)])
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if self.args.finetune_audio:
            self.backbone_model.freeze_feature_extractor()
        else:
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False    
            
        
    def forward(self, x, length=None, return_feature=False):
        # 1. feature extraction and projections
        with torch.no_grad():
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2) # New version of huggingface
        x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
        
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()
            
        # 3. transformer encoding features
        features = self.backbone_model.encoder(x, output_hidden_states=True)
        encoded_feature = features.last_hidden_state
        
        if return_feature:
            mask = (torch.arange(encoded_feature.size(1), device=encoded_feature.device)[None, :] < length[:, None]).float()
            return encoded_feature, mask
        
        # # 7. Pooling
        if length is not None:
            masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)
        
        # 8. Output predictions
        # B x D
        # predicted = self.out_layer(features)
        return features
        
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--finetune_method', 
        default='none',
        type=str, 
        help='finetune method: adapter, embedding prompt, input prompt'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim', 
        default=128,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--embedding_prompt_dim', 
        default=5,
        type=int, 
        help='adapter dimension'
    )
    
    args = parser.parse_args()
    model = Wav2VecWrapper(args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(output.shape)