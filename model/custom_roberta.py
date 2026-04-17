

import torch
from torch import nn
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaSelfOutput
from transformers.models.roberta.modeling_roberta import RobertaIntermediate, RobertaOutput


from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer, find_pruneable_heads_and_indices

from torch import nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers import RobertaModel, RobertaTokenizer, AutoConfig

from transformers import RobertaModel
from utils import tokenize_texts, tokenize_texts_with_current
import inspect



import torch
import numpy as np
import loralib as lora 



from torch import nn
from torch.nn import functional as F
from transformers import RobertaModel, RobertaTokenizer, AutoConfig


from transformers import RobertaModel
from utils import tokenize_texts

from peft import get_peft_model, LoraConfig
import loralib as lora 

# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta

'''
[last update: 09/22/2025]

'''

class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class RobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # if encoder_hidden_states is not None: 
            # self.cross atten layer
            # print("[Roberta Attention]")
            # print(f"encoder_hidden_states: {encoder_hidden_states.shape}, hidden_states:{hidden_states.shape}")
            # print(f"encoder_attention_mask: {encoder_attention_mask.shape}, ")
            
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
            
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            # print("crossattention")
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

# Roberta encoder layer for cross attention 
class RobertaLayerWithCross(nn.Module):
    def __init__(self, base_layer: RobertaLayer, layer_idx: int, fusion_layer: int, config):
        super().__init__()
        self.attention     = base_layer.attention
        self.intermediate  = base_layer.intermediate
        self.output        = base_layer.output
        
        self.enable_cross = (layer_idx >= fusion_layer)
        if self.enable_cross:
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # 1) self-attn
        sa_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = sa_outputs[0]
        attn_outputs = sa_outputs[1:]  # tuple of (self_attn_probs) if requested

        # 2) cross-attn 
        if self.enable_cross and (encoder_hidden_states is not None):
            ca_outputs = self.crossattention(
                hidden_states,
                attention_mask,              # query mask
                head_mask,
                encoder_hidden_states,       # key/value
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = ca_outputs[0]
            if output_attentions:
                attn_outputs = attn_outputs + (ca_outputs[1],)

        # 3) FFN
        # hidden_states = apply_chunking_to_forward(   
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_outputs
        # )
        
        hidden_states = apply_chunking_to_forward(
            lambda x: self.output(self.intermediate(x), x),
            0, 1, hidden_states
        )
        return (hidden_states,) + attn_outputs
    

class RobertaEncoder(nn.Module):
    def __init__(self, base_encoder, config):
        super().__init__()
            
        self.config = config
        config.fusion_layer = int(config.num_hidden_layers/2)
        print("Custom RobertaEncoder config: ", config)
        self.layer = nn.ModuleList([RobertaLayerWithCross(base_encoder.layer[i], i, config.fusion_layer, config) for i in range(config.num_hidden_layers)])
        # self.layer = nn.ModuleList([RobertaLayerWithCross(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        mode='text'
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        B, T, H = hidden_states.shape

        if encoder_hidden_states is not None: # Audio mask 
            extended_attention_mask = self._expand_mask(attention_mask, hidden_states.dtype)
            encoder_extended_attention_mask = self._expand_mask(encoder_attention_mask, hidden_states.dtype)
        else:
            encoder_extended_attention_mask = None
            
        # Select layer range based on mode
        if mode == 'text':
            start_layer = 0
            output_layer = self.config.fusion_layer
        elif mode == 'fusion':
            start_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers
        elif mode == 'multi_modal':
            start_layer = 0
            output_layer = self.config.num_hidden_layers
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        # for i, layer_module in enumerate(self.layer):
        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask if i >= self.config.fusion_layer else attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    encoder_hidden_states=encoder_hidden_states if i >= self.config.fusion_layer else None,
                    encoder_attention_mask=encoder_extended_attention_mask if i >= self.config.fusion_layer else None,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
 
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # mask: (B, S) with 1 for keep, 0 for pad
        # returns (B, 1, 1, S) with 0 for keep, large negative for pad
        expanded = mask[:, None, None, :].to(dtype)              # (B,1,1,S)
        return (1.0 - expanded) * -10000.0

class RobertaCrossAttn(nn.Module):
    def __init__(self, args, acoustic_model=None):
        super(RobertaCrossAttn, self).__init__()
        
        self.config                 = AutoConfig.from_pretrained(args.text_model)
        self.semantic_model         = RobertaModel.from_pretrained(args.text_model, self.config)
        self.semantic_config         = self.semantic_model.config 
        # torch.save(self.semantic_model.state_dict(), 'roberta.pt')
        # add config 
        self.semantic_config.modal = args.modal 
        self.semantic_config.cross_modal_atten = args.cross_modal_atten
        
        if args.modal in ['multimodal']: 
            # Using Custom Robertaㅍ Encoder for cross modal attention 
            base_encoder                = self.semantic_model.encoder
            self.semantic_model.encoder = RobertaEncoder(base_encoder, self.semantic_config)
      
        self.max_txt_len        = args.max_txt_len
        self.ws                 = args.ws
        self.self_attn          = args.self_attn
        self.pooling_mode       = args.pooling_mode
        self.clamp              = args.clamp
        self.tokenize_mode      = args.tokenize_mode
        self.load_pt            = args.load_pt

        if args.num_hidden_layers is None:
            self.num_layers       = self.semantic_config.num_hidden_layers
        else: 
            self.num_layers       = args.num_hidden_layers
            
        # PEFT roberta  
        if args.finetune_roberta:
            self.semantic_config.finetune_roberta = args.finetune_roberta
            self.semantic_config.lora_target_modules = args.lora_target_modules
  
            # Full Finetuning  
            if args.lora_target_modules == 'full':
                state_dict = self.semantic_model.state_dict()
                msg = self.semantic_model.load_state_dict(state_dict, strict=False)
                for name, p in self.semantic_model.named_parameters():
                    p.requires_grad = True
  
        # Frozen 
        else: 
            state_dict = self.semantic_model.state_dict()
            msg = self.semantic_model.load_state_dict(state_dict, strict=False)
            for name, p in self.semantic_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
            
        self.tokenizer = RobertaTokenizer.from_pretrained(args.text_model)


        special_tokens = ['[Previous]', '[Current]']
        special_tokens += ["<SELF>", "<OTHER>"]
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.semantic_model.resize_token_embeddings(len(self.tokenizer))
        
        self.cur_id = self.tokenizer.convert_tokens_to_ids("[Current]")
        self.sep_id = self.tokenizer.sep_token_id
        
        self.truncation_side = args.truncation_side
        self.padding = args.padding

        if acoustic_model is not None:
            self.acoustic_config    = acoustic_model.backbone_model.config 

        if self.pooling_mode in ["weighted_pool"]:
            init_alpha = torch.normal(mean=torch.tensor(0.6), std=torch.tensor(0.05))
            init_alpha = torch.clamp(init_alpha, 0.0, 1.0)  
            self.alpha = nn.Parameter(init_alpha)
        print(self.semantic_model)
        
    def span_pool(self, last_hidden_state, input_ids, attention_mask):
        B, L, H = last_hidden_state.shape
        pos = torch.arange(L, device=input_ids.device).expand(B, L)
        cur_mask = (input_ids == self.cur_id)
        start = cur_mask.float().argmax(dim=1)
        sep = (input_ids == self.sep_id)
        after = pos >= start[:, None]
        first_sep = (sep & after)
        end = torch.where(first_sep.any(1), first_sep.float().argmax(1), torch.full_like(start, L-1))
        cur_span = (pos >= start[:, None]) & (pos < end[:, None])
        prev_span = (pos < start[:, None])

        def pool(span_mask):
            mask = (span_mask & attention_mask.bool()).float().unsqueeze(-1)  # [B,L,1]
            num = (last_hidden_state * mask).sum(1)
            den = mask.sum(1).clamp_min(1e-6)
            return num / den

        cur_vec  = pool(cur_span)
        prev_vec = pool(prev_span)

        out = self.alpha * cur_vec + (1 - self.alpha) * prev_vec
        return out
    
    def current_span_pool(self, last_hidden_state, input_ids, attention_mask):
        
        B, L, H = last_hidden_state.shape
        device = input_ids.device

        cur_mask = (input_ids == self.cur_id)        # [B, L]
        has_cur  = cur_mask.any(dim=1)               # [B]
        start    = cur_mask.float().argmax(dim=1)    # [B]

        sep_id   = self.sep_id
        is_sep   = (input_ids == sep_id)             # [B, L]

        pos = torch.arange(L, device=device).expand(B, L)  # [B, L]
        after     = pos >= start[:, None]            # [B, L]
        first_sep = (is_sep & after)                 # [B, L]
        end = torch.where(
            first_sep.any(dim=1),
            first_sep.float().argmax(dim=1),         # [B]
            torch.full_like(start, L-1)
        )                                            # [B]
        span = (pos >= start[:, None]) & (pos < end[:, None])  # [B, L]
        mask = (span & attention_mask.bool() & has_cur[:, None]).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (last_hidden_state * mask).sum(dim=1)                 # [B, H]
        denom  = mask.sum(dim=1).clamp_min(1e-12)                       # [B, 1]
        pooled = pooled / denom                                        # [B, H]
        return pooled
    
    
    def forward(self, 
                embeddings=None,  # text input 
                s_token_type_ids=None,
                s_position_ids=None,
                s_head_mask=None,
                s_labels=None,
                a_inputs=None,
                a_attention_mask=None,
                a_token_type_ids=None,
                a_position_ids=None,
                a_head_mask=None,
                a_labels=None,
                a_mask_labels=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None, 
                acoustic_encode=None, 
                mode=None,
                s_attention_mask=None,
                input_ids=None,
                multimodal_pooling=None
                ): 
        
        ## Multimodal fusion 
        if acoustic_encode is not None and mode == 'fusion':
            return_dict = return_dict if return_dict is not None \
            else (self.acoustic_config.use_return_dict and self.semantic_config.use_return_dict)
            
            # fusion feature 
            semantic_encode = self.semantic_model.encoder(
                        hidden_states=embeddings,
                        attention_mask=s_attention_mask,
                        encoder_hidden_states=acoustic_encode,
                        encoder_attention_mask=a_attention_mask,
                        mode=mode
            ) 
            
            if multimodal_pooling: 
                if multimodal_pooling in ['curr_only']:
                    encoded_feature =  self.current_span_pool(semantic_encode.last_hidden_state, 
                                                            input_ids, s_attention_mask)  
                elif multimodal_pooling in ['weighted_pool']:
                    encoded_feature =  self.span_pool(semantic_encode.last_hidden_state, 
                                                    input_ids, s_attention_mask)   
                elif multimodal_pooling in ['mean']:
                    encoded_feature =  torch.mean(semantic_encode.last_hidden_state, dim=1) # mean pooling 
                else: 
                    encoded_feature =  semantic_encode.last_hidden_state[:, 0, :] # CLS return 
            else:
                if self.pooling_mode in ['curr_only']:
                    encoded_feature =  self.current_span_pool(semantic_encode.last_hidden_state, 
                                                            input_ids, s_attention_mask)  
                elif self.pooling_mode in ['weighted_pool']:
                    encoded_feature =  self.span_pool(semantic_encode.last_hidden_state, 
                                                    input_ids, s_attention_mask)   
                elif self.pooling_mode in ['mean']:
                    encoded_feature =  torch.mean(semantic_encode.last_hidden_state, dim=1) # mean pooling 
                else: 
                    encoded_feature =  semantic_encode.last_hidden_state[:, 0, :] # CLS return 
                
            return encoded_feature
        
        ## Text feature encode 
        else: 
            if self.load_pt: 
                input_ids, attention_mask = embeddings
            else: 
                if self.tokenize_mode in ['default']:
                    input_ids, attention_mask = tokenize_texts(embeddings, self.tokenizer, self.max_txt_len, self.truncation_side, self.padding)
                else: 
                    input_ids, attention_mask = tokenize_texts_with_current(embeddings, self.tokenizer, self.max_txt_len)
            
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            semantic_encode = self.semantic_model(input_ids=input_ids, attention_mask=attention_mask)    
                                                    
            if self.pooling_mode in ['curr_only']:
                encoded_feature =  self.current_span_pool(semantic_encode.last_hidden_state, input_ids, attention_mask)  
            elif self.pooling_mode in ['weighted_pool']:
                encoded_feature =  self.span_pool(semantic_encode.last_hidden_state, input_ids, attention_mask)   
            elif self.pooling_mode in ['mean']:
                encoded_feature =  torch.mean(semantic_encode.last_hidden_state, dim=1) # mean pooling 
            else: 
                encoded_feature =  semantic_encode.last_hidden_state[:, 0, :] # CLS return    
            
            return encoded_feature, attention_mask, semantic_encode.last_hidden_state, input_ids

