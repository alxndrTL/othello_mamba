from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import Transformer, TransformerConfig
from models.mamba.mamba import Mamba, MambaConfig

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig, MambaConfig], vocab_size: int):
        super().__init__()

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model)
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        else:
            self.core = Mamba(self.config)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)
        logits = self.lm_head(x)

        return logits
