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
import transformers.models.wavlm.modeling_wavlm as wavlm

from functools import lru_cache
from torchaudio.compliance import kaldi

from torch import nn
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, AutoProcessor, WavLMModel, WhisperModel, AutoFeatureExtractor, WavLMPreTrainedModel


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):

        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
 
class WavLMWrapper(nn.Module):
    def __init__(
        self, 
        args, 
        hidden_dim=256,
        output_class_num=4
    ):
        super(WavLMWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.args = args
        if args.audio_model == 'wavlm': 
            model_name = "microsoft/wavlm-base-plus"
        elif args.audio_model == 'wavlm-large':
            model_name = "microsoft/wavlm-large"
            
        self.backbone_model = WavLMModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        state_dict = self.backbone_model.state_dict()

        self.args.speaker = None
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_audio        = self.args.finetune_audio

        self.backbone_model.encoder.layers = nn.ModuleList(
            [WavLMEncoderLayer(self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
        )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        
        # 5. Freeze the weights
        if self.args.finetune_audio:
            self.backbone_model.freeze_feature_extractor() # freeze only feature extractor 
        else:
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False    
        self.projector  = nn.Sequential(nn.Linear(self.model_config.hidden_size, hidden_dim))
            
    def forward(self, x, length=None, return_feature=False):
        # 1. feature extraction and projections
        x = self.backbone_model.feature_extractor(x)
        x = x.transpose(1, 2) # New version of huggingface
        x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
            
        # 2. get length and mask
        attn = None
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()
            attn = (torch.arange(x.size(1), device=x.device)[None, :] < length[:, None]).cuda()
            
        features = self.backbone_model.encoder(
            x, attention_mask=attn, output_hidden_states=True
        )

        encoded_feature = features.last_hidden_state
        h = self.projector(encoded_feature)
        
        if return_feature:
            mask = (torch.arange(h.size(1), device=h.device)[None, :] < length[:, None]).float()
            return encoded_feature, mask
        else: 
            if length is not None:
                mask = (torch.arange(h.size(1), device=h.device)[None, :] < length[:, None]).float()
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                h = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
                return h, mask
            else:
                h = h.mean(dim=1)
                return h
            
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
    