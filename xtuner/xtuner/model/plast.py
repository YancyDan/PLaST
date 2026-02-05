from collections import OrderedDict

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from xtuner.dataset.plast import prepare_inputs_labels_for_plast
from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad, traverse_dict)
from geomloss import SamplesLoss

np.random.seed(42)


""" correct mha"""
class MHA(nn.Module):
    def __init__(
        self, 
		qdim=2048, 
		kvdim=768,
		num_heads=8,
        hidden_act='gelu'
	):
        super().__init__()
        self.embed_dim = qdim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim: {self.embed_dim} but num_heads: {num_heads}"

        self.q_proj = nn.Linear(qdim, qdim)
        self.k_proj = nn.Linear(kvdim, qdim)
        self.v_proj = nn.Linear(kvdim, qdim)
        self.out_proj = nn.Linear(qdim, qdim)
        
        self.ffn = nn.Sequential(
            nn.Linear(qdim, qdim),
            ACT2FN[hidden_act],
            nn.Linear(qdim, qdim),
            ACT2FN[hidden_act],
            nn.Linear(qdim, qdim),
        )

    def forward(self, query, key, value, key_padding_mask=None):
        """
        	query: [batch, tgt_len, embed_dim]
			key: [batch, src_len, kdim]
			value: [batch, src_len, vdim]
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # 1. input proj
        Q_proj = self.q_proj(query)        # [batch, tgt_len, embed_dim]
        K_proj = self.k_proj(key)          # [batch, src_len, embed_dim]
        V_proj = self.v_proj(value)        # [batch, src_len, embed_dim]

        # 2. split multi-heads
        Q = self._split_heads(Q_proj)      # [batch, num_heads, tgt_len, head_dim]
        K = self._split_heads(K_proj)      # [batch, num_heads, src_len, head_dim]
        V = self._split_heads(V_proj)      # [batch, num_heads, src_len, head_dim]

        # 3. attention
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Q*K^T / sqrt(d_k)

        # 4. masking
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.view(batch_size, 1, 1, -1),  # [batch, 1, 1, src_len]
                float('-inf')
            )

        # 5. Softmax + Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)  	 # [batch, num_heads, tgt_len, src_len]
        attn_output = torch.matmul(attn_weights, V)        	 # [batch, num_heads, tgt_len, head_dim]
        attn_output = self._combine_heads(attn_output)       # [batch, tgt_len, embed_dim]
        
        # 6. output proj
        output = self.ffn(attn_output)
        return output, attn_weights
        

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.embed_dim)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class AudioProjectorConfig(PretrainedConfig):
    model_type = 'projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        audio_hidden_size=4096,
        llm_hidden_size=4096,
        depth=3,
        conv_depth=0,
        hidden_act='gelu',
        bias=True,
        **kwargs,
    ):
        self.audio_hidden_size = audio_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        self.conv_depth = conv_depth
        super().__init__(**kwargs)


class AudioEncoder(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = AudioProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: AudioProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False
        print('*' * 30)
        print(config.audio_hidden_size, config.llm_hidden_size)
        modules = []

        for _ in range(config.conv_depth):
            # [B, L, D] -> [B, D, L]
            modules.append(Permute(0, 2, 1))
            modules.append(
                nn.Conv1d(
                    in_channels=config.audio_hidden_size,
                    out_channels=config.audio_hidden_size,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )

            # [B, D, L] -> [B, L, D]
            modules.append(Permute(0, 2, 1))
            modules.append(ACT2FN[config.hidden_act])

        modules.append(
            nn.Linear(config.audio_hidden_size, config.llm_hidden_size))
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model = nn.Sequential(*modules)

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        # if isinstance(module, AudioProjectorConfig):
        if isinstance(module, nn.Module):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs


class PLaSTModel(BaseModel):
    """Implementation of plast.

    Acknowledge: LLaVA: Visual Instruction Tuning
    (llava-vl.github.io/)
    """

    def __init__(
        self,
        llm,
        speech_encoder,
        freeze_llm=False,
        freeze_speech_encoder=False,
        freeze_projector=False,
        speech_select_layer=-1,
        pretrained_pth=None,
        projector_depth=2,
        llm_lora=None,
        speech_encoder_lora=None,
        llm_lora_trainable=True,
        speech_encoder_lora_trainable=True,
        use_activation_checkpointing=True,
        conv_depth=0,
        training_stage=0,
        use_hidden_state=False,
        layers_hidden_state=[-1],
        w_hidden_states=[1],
        use_ot=False,
        w_ot=0.001,
        use_contrast=False,
        w_contrast=0.3,
        use_cutoff=False,
        p_cutoff=0.1,
        use_feature=False,
        feature_detach=True
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_speech_encoder = freeze_speech_encoder
        self.freeze_projector = freeze_projector
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.speech_encoder = self._build_from_cfg_or_module(
                speech_encoder)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = AudioProjectorConfig(
            audio_hidden_size=self.speech_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth,
            conv_depth=conv_depth)
        self.projector = AudioEncoder(projector_config).to(
            self.speech_encoder.dtype)

        if use_feature and training_stage != 1:
            self.mha = MHA(qdim=self.llm.config.hidden_size).to(
                self.speech_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_speech_encoder:
            self.speech_encoder.requires_grad_(False)
        if self.freeze_projector:
            self.projector.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.speech_encoder, 'enable_input_require_grads'):
                self.speech_encoder.enable_input_require_grads()
            else:
                self.speech_encoder.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = (llm_lora is not None) and (training_stage != 1)
        self.use_speech_encoder_lora = speech_encoder_lora is not None
        
        self.llm_lora_trainable = llm_lora_trainable
        self.speech_encoder_lora_trainable = speech_encoder_lora_trainable

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_speech_encoder_lora:
            self._prepare_speech_encoder_for_lora(
                speech_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            if 'mha.ffn.4' in pretrained_state_dict:
                out_str = self.load_state_dict(pretrained_state_dict, strict=False)
            else:
                new_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'mha' not in k}
                out_str = self.load_state_dict(new_state_dict, strict=False)
            assert len(out_str.unexpected_keys) == 0, out_str.unexpected_keys
            print(f'Load pretrained weight from {pretrained_pth}')

        if not self.llm_lora_trainable:
            self.llm.requires_grad_(False)
        if not self.speech_encoder_lora_trainable:
            self.speech_encoder.requires_grad_(False)
        
        self.speech_select_layer = speech_select_layer
        self._is_init = True

        self.use_feature = use_feature
        self.feature_detach = feature_detach
       
        self.training_stage = training_stage 
        self.use_hidden_state = use_hidden_state
        self.layers_hidden_state = layers_hidden_state
        self.w_hidden_states = w_hidden_states
        
        self.use_ot = use_ot
        self.w_ot = w_ot
        self.use_contrast = use_contrast
        self.w_contrast = w_contrast
        self.use_cutoff = use_cutoff
        self.p_cutoff = p_cutoff

        
        if self.use_hidden_state:
            assert len(self.layers_hidden_state) == len(self.w_hidden_states), \
                "layers and weights length not match"
                
        if self.training_stage == 1:
            assert pretrained_pth is None, \
                "stage 1 should not load checkpoint"

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.speech_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.projector.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.speech_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. speech_encoder
        if self.use_speech_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.speech_encoder, state_dict=state_dict))
        elif not self.freeze_speech_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'speech_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        # Step 4. mha
        if self.use_feature:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'mha.' in k}
            )
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_speech_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.speech_encoder)
            lora_config.target_modules = modules
        self.speech_encoder = get_peft_model(self.speech_encoder, lora_config)

    def forward(self, data, data_samples=None, mode='loss'):
        if 'audio_tokens' in data:
            data['audio_tokens'] = data['audio_tokens'].to(
                self.speech_encoder.encoder.conv1.weight.dtype)
            batch_size = data['audio_tokens'].shape[0]
            decoder_input_ids = torch.tensor([
                [1] * batch_size
            ]) * self.speech_encoder.config.decoder_start_token_id

            audio_outputs = self.speech_encoder(
                data['audio_tokens'],
                decoder_input_ids=decoder_input_ids.to(
                    data['audio_tokens'].device),
                output_hidden_states=True).encoder_last_hidden_state

            audio_outputs = audio_outputs[:, :max(data['audio_lens']), :]
            audio_tokens = self.projector(audio_outputs)
            data['audio_tokens'] = audio_tokens

            # audio feature
            audio_feature = audio_tokens
            audio_lens = data['audio_lens']
            max_audio_len = max(audio_lens)
            audio_mask = torch.arange(max_audio_len, device=audio_feature.device).expand(batch_size, max_audio_len) \
                < audio_lens.unsqueeze(1)

            # src content
            src_ids = data['src_ids']
            src_mask = (data['src_ids'] != 0).long()

            # auxilary information by cross-attn 
            if self.use_feature:
                auxi_feature = data['audio_features'].to(audio_tokens.dtype)
                auxi_lens = data['feature_lens']
                max_auxi_len = max(auxi_lens)
                auxi_mask = torch.arange(max_auxi_len, device=auxi_feature.device).expand(batch_size, max_auxi_len) \
                    >= auxi_lens.unsqueeze(1)

                # detach feature from whisper
                if self.feature_detach:
                    detached_audio_tokens = audio_tokens.detach()  
                    data['audio_features'], _ = self.mha(detached_audio_tokens, auxi_feature, auxi_feature, auxi_mask)
                else:
                    data['audio_features'], _ = self.mha(audio_tokens, auxi_feature, auxi_feature, auxi_mask)
            
            data.pop('src_ids', None)
            data.pop('src_ids_rep', None)                
            data.pop('feature_lens', None)
            data = prepare_inputs_labels_for_plast(llm=self.llm, **data)

        if mode == 'loss':
            # only alignment
            if self.training_stage == 1:
                if self.use_hidden_state and self.use_ot:
                    src_hiddens = self.llm(input_ids=src_ids.to(self.llm.device),
                                           attention_mask=src_mask.to(self.llm.device),
                                           output_hidden_states=True)['hidden_states']

                    audio_hiddens = self.llm(inputs_embeds=audio_feature.to(self.llm.device),
                                             attention_mask=audio_mask.to(self.llm.device),
                                             output_hidden_states=True)['hidden_states']
                    
                    src_hidden_states = [src_hiddens[i] for i in self.layers_hidden_state]
                    audio_hidden_states = [audio_hiddens[i] for i in self.layers_hidden_state]
                    
                    ot_loss = 0
                    for text, audio, w in zip(src_hidden_states, audio_hidden_states, self.w_hidden_states):
                        ot_loss += w * self.wasserstein_loss(audio*audio_mask.unsqueeze(-1), 
                                                             text*src_mask.unsqueeze(-1), 
                                                             mask_audio=audio_mask,
                                                             mask_text=src_mask)
                    return {'loss': ot_loss}
            # only translation
            elif self.training_stage == 2:
                return self.compute_loss(data, data_samples)
            else:
                raise NotImplementedError
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def info_nce_loss(
        self,
        features,  # [2B, D]
        temperature=0.02,
        bi_view=False  # True for 2B-2 samples, False for B-1 samples
    ):
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size)
                           for _ in range(2)], dim=0).to(features.device)

        # cos similarity
        features_normalized = F.normalize(features, dim=1)
        similarity_matrix = torch.mm(
            features_normalized, features_normalized.T)  # [2B, 2B]

        # pair mask
        eye_mask = torch.eye(
            2 * batch_size, dtype=torch.bool, device=features.device)
        sims = similarity_matrix[~eye_mask].view(
            2 * batch_size, -1)  # [2B, 2B-1]

        # positive similarity
        pos_indices = (labels.unsqueeze(
            0) == labels.unsqueeze(1)).to(features.device)
        pos_indices = pos_indices[~eye_mask].view(2 * batch_size, -1)
        positive_similarities = sims[pos_indices].view(
            2 * batch_size, -1)  # [2B, 1]

        # negative similarity
        if bi_view:
            # using all
            neg_mask = ~pos_indices
        else:
            # using the other spec
            view = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).to(
                features.device)
            view_comparison = (view.unsqueeze(1) != view.unsqueeze(0)).to(
                features.device)  # [2B, 2B]
            view_comparison = view_comparison[~eye_mask].view(
                2 * batch_size, -1)  # [2B, 2B-1]
            neg_mask = ~pos_indices & view_comparison  # negative & cross-view

        negative_similarities = sims[neg_mask].view(2 * batch_size, -1)

        # InfoNCE Loss
        logits = torch.cat(
            [positive_similarities, negative_similarities], dim=1)
        logits = logits / temperature

        labels = torch.zeros(
            2 * batch_size, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def wasserstein_loss(
        self,
        h_audio: torch.Tensor,  # [batch_size, max_len_audio, d]
        h_text: torch.Tensor,  # [batch_size, max_len_text, d]
        mask_audio=None,
        mask_text=None,
        lambda_reg: float = 1.0,  # λ
        use_position_reg=False,
        position_scale: float = 1.0,  # α
        eps: float = 1e-8,
        norm_dim=2
    ) -> torch.Tensor:

        batch_size, max_len_audio, d = h_audio.shape
        _, max_len_text, _ = h_text.shape

        h_audio = (h_audio - h_audio.mean(dim=norm_dim, keepdim=True)) / \
            (h_audio.std(dim=norm_dim, keepdim=True) + eps)
        h_text = (h_text - h_text.mean(dim=norm_dim, keepdim=True)) / \
            (h_text.std(dim=norm_dim, keepdim=True) + eps)
        
        if mask_text is not None:
            weights_text = mask_text.float() / (mask_text.sum(dim=1, keepdim=True) + eps)
            weights_audio = mask_audio.float() / (mask_audio.sum(dim=1, keepdim=True) + eps)

        if use_position_reg:
            def _get_pos_enc(max_len: int) -> torch.Tensor:
                positions = torch.linspace(
                    0, 1, steps=max_len, dtype=torch.float32, device=h_audio.device)
                return positions.view(1, max_len, 1).expand(batch_size, -1, -1)

            # [batch, max_len_audio, 1]
            pos_audio = _get_pos_enc(max_len_audio)
            pos_text = _get_pos_enc(max_len_text)    # [batch, max_len_text, 1]

            h_audio = torch.cat(
                [h_audio, position_scale * pos_audio], dim=-1)  # [batch, a, d+1]
            h_text = torch.cat(
                [h_text, position_scale * pos_text], dim=-1)     # [batch, t, d+1]

        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=lambda_reg,
            scaling=0.9,
            backend="tensorized",
        )
        
        if mask_text is not None:
            return sinkhorn_loss(weights_audio, h_audio, weights_text, h_text).mean()
        else:
            return sinkhorn_loss(h_audio, h_text).mean()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
