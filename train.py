"""
main training script.

todo : 
-gradient acc
-multiple gpus (DDP)

-flops util (in final log)
-weight decay different pour certains params

-SAVE THE UNCOMPILED MODEL
-save for HF (model_export dans llama2.c/train.py ????)

-mamba en float32 : attention ! justement, garder les .float() ? regarder le code officiel

"""

import os
import math
import time
import random
import string
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import wandb

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from data import OthelloDataset
from eval import eval

# -------------------------------------------------------

log_wandb = False

d_model = 512
n_layers = 8
n_heads = 8

batch_size = 16

num_iters = 500 # 1000 = 1 min
train_log_interval = 50
eval_acc_interval = 1000
eval_val_interval = 200
eval_iters = 50

lr = 1e-3
lr_min = 1e-5 # as in Mamba paper
lr_warmup_iters = 100
lr_decay_iters = 10000 # num_iters as in Chinchilla

dropout = 0.
bias = False

adam_b1 = 0.9
adam_b2 = 0.95

clip_value_grad = 1.0
weight_decay = 0.1

use_torch_compile = False
use_flash_attention = True

device = "cuda" # cpu, cuda:0, cuda:1, ...
dtype = "float32" # float32, float16 or bfloat16 (float16 will use a GradScaler)

load_checkpoint = False
checkpoint_load_dir = "runs/sleek-water-17.pth" # where to load from (if load_checkpoint)

data_dir = "data/"
save_dir = "runs/" # where to save to

# -------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if log_wandb:
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
    
if log_wandb:
    run_name = wandb.run.name
else:
    run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

save_dir = os.path.join(save_dir, run_name + '.pth')

print(f"Run name: {run_name}.")

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
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=True) # todo : bs de 1 ici, quand on l'augmentera attention a eval.py
iter_val = iter(loader_val)

#config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, bias=bias, max_len=60, flash=use_flash_attention)
config = MambaConfig(d_model=d_model, n_layers=n_layers)
model = LM(config, vocab_size=65).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(adam_b1, adam_b2), weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))

print(f"Model initialized. Number of parameters : {sum([p.numel() for p in model.parameters()])}.")

unoptimized_model = model
if use_torch_compile:
    print("Compiling the model...")
    model = torch.compile(model)
    print("Done compiling.")

if load_checkpoint:
    checkpoint = torch.load(checkpoint_load_dir, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

    print(f"Successfully loaded checkpoint from {checkpoint_load_dir}.")

print("Training is starting.")
start_time = time.time()

for iter, data in enumerate(loader):
    x, y = data
    x, y = x.to(device), y.to(device)

    with dtype_ctx:
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

    # logging : print and wandb
    to_log = {}
    if iter % train_log_interval == 0:
        to_log.update({"train_loss": loss.item()})

    if iter % eval_val_interval == 0:
        with torch.no_grad():
            model.eval()
            eval_loss = 0
            for i in range(eval_iters):
                data = next(iter_val)
                x, y = data
                x, y = x.to(device), y.to(device)

                with dtype_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                eval_loss += loss.item()

            eval_loss /= eval_iters
            model.train()
        
        to_log.update({"val_loss": eval_loss})
    
    if iter % eval_acc_interval == 0:
        with torch.no_grad():
            model.eval()
            acc = eval(unoptimized_model, device, 10, loader_val) # evaluate on 10 games
            model.train()
        to_log.update({"accuracy": acc})

    if to_log:
        to_log.update({"lr": lr_iter})

        # printing
        if "val_loss" in to_log:
            num_digits = len(str(num_iters))
            formatted_iter = f"{iter:0{num_digits}d}"
            print(f"Step {formatted_iter}/{num_iters}. train loss : {loss.item():.3f}. valid loss : {eval_loss:.3f}. lr : {lr_iter:.5f}. uptime : {(time.time()-start_time)/60:.2f} minutes.")

        # logging
        if log_wandb:
            wandb.log(to_log, step=iter)

    if iter >= num_iters:
        break

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")

checkpoint = {"model": model.state_dict(),
              "optimizer": optim.state_dict(),
              "scaler": scaler.state_dict()}
torch.save(checkpoint, save_dir)

print(f"Successfully saved checkpoint in {save_dir}.")

model.eval()
final_acc = eval(unoptimized_model, device, 50, loader_val)
model.train()
print(f"Final accuracy: {100.*final_acc:.2f}%")

# final log
num_params = sum([p.numel() for p in model.parameters()])
num_tokens_processed = num_iters * batch_size * 60

to_log = {"final_accuracy": final_acc,
          "num_params": num_params,
          "tokens_per_s": int(num_tokens_processed/(end_time-start_time)),
          "iter_per_s": int(num_iters/(end_time-start_time)),
          "use_torch_compile": use_torch_compile,
          "use_flash_attn": use_flash_attention,
          "dtype": dtype,}

if log_wandb:
    wandb.log(to_log)
    wandb.finish()
