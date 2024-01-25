"""
main training script.

todo : 
-gradient acc ?

"""

import os
import math

import torch
import torch.nn.functional as F

import wandb

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from data import OthelloDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

run_name = "test0"

d_model = 512
n_layers = 8
n_heads = 8

batch_size = 1

num_iters = 500 # todo : up iters and lr_decays_iters

lr = 5e-4 # todo : up BS and LR
lr_min = 1e-5
lr_warmup_iters = 100
lr_decay_iters = 10000 # max_iters as in chinchilla

dropout = 0.
bias = False

adam_b1 = 0.9
adam_b2 = 0.95

clip_value_grad = 1.0
weight_decay = 0.1

load_checkpoint = False
checkpoint_dir = "ckpt.pth"

data_dir = "data/"

# -------------------------------------------------------

wandb.init(project="othello",
           config={
               "architecture": "Transformer",
               "d_model": d_model,
               "n_layers": n_layers,
               "n_heads": n_heads,
               "num_iters": num_iters,
               "batch_size": batch_size,
               "lr": lr,
               "lr_min": lr_min,
               "lr_warmup_iters": lr_warmup_iters,
               "lr_decay_iters": lr_decay_iters,
               "dropout": dropout,
               "bias": bias,
               "adam_b1": adam_b1,
               "adam_b2": adam_b2,
               "clip_value_grad": clip_value_grad,
               "weight_decay": weight_decay,
           })

# cosine with warmup (taken from @karpathy)
def get_lr(it):
    if lr_decay_iters == 0:
        return lr
    
    # 1) linear warmup for warmup_iters steps
    if it < lr_warmup_iters:
        return lr * it / lr_warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return lr_min
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - lr_warmup_iters) / (lr_decay_iters - lr_warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return lr_min + coeff * (lr - lr_min)

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

ds = OthelloDataset(train_dir)
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

ds_val = OthelloDataset(val_dir)
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=1, num_workers=0, pin_memory=True) # todo : bs de 1 ici, quand on l'augmentera attention a eval.py

config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, bias=bias, max_len=60, flash=True)
model = LM(config, vocab_size=65).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(adam_b1, adam_b2), weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()

#sum([p.numel() for p in model.parameters()])

if load_checkpoint:
    checkpoint = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])


for iter, data in enumerate(loader):
    x, y = data
    x = x.int().to(device)
    y = y.long().to(device)

    with torch.autocast(device, torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)

    scaler.scale(loss).backward()

    if clip_value_grad != 0.0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value_grad)
    
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

    # lr decay
    lr_iter = get_lr(iter)
    for param_group in optim.param_groups:
        param_group['lr'] = lr_iter

    if iter >= num_iters:
        break

checkpoint = {"model": model.state_dict(),
              "optimizer": optim.state_dict(),
              "scaler": scaler.state_dict()}
torch.save(checkpoint, checkpoint_dir)

from eval import eval
print(eval(model, 50, loader_val))

wandb.finish()