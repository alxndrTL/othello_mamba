import math

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from data import OthelloDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

d_model = 512
n_layers = 8
n_heads = 8

B = 1

dropout = 0.
bias = False

lr = 5e-4 # todo : up BS and LR
lr_min = 1e-5
lr_warmup_iter = 100
lr_decay_iter = 10000 # max_iters as in chinchilla

adam_b1 = 0.9
adam_b2 = 0.95

clip_value_grad = 1.0
weight_decay = 0.1

# -------------------------------------------------------

