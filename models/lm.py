from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import Transformer, TransformerConfig, RMSNorm
from models.mamba.mamba import Mamba, MambaConfig

# todo : inference function, with no grad, with kv cache for transformer, step() for mamba

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig, MambaConfig], vocab_size: int):
        super().__init__()

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        else:
            self.core = Mamba(self.config)

        self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)
        x = self.out_norm(x)
        logits = self.lm_head(x)

        return logits
    
    def forward_up_to(self, tokens, layer):
        # tokens : (B, L)
        # layer (1->n_layers): will stop the forward pass just after this layer

        # x : (B, L, D) activations after {layer}

        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=layer)

        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
