"""
main training script.
Depending on the hyperparameters and your machine, can take between 30 min to a few hours.

When launching the training, the script will assign a name to the run (same as wandb run if log_wandb is enabled).
It will place the config, model and checkpoints in runs/{run_name} dir.
The config file stores all hyperparams relative to the model (architecture, d_model, bias, n_heads if Transformer...).
It is used by downstream scripts to load the model.

not implemented:
-gradient acc
-multiple gpus (DDP)
-save for HF
-training from a checkpoint : it works, but lr must be manually set to cst=lr_min, and wandb logging is restarted
-flops utilization

Notes :
-given the same d_model, Mamba should use 2x more layers than a Transformer to match its number of params
(its not actually deeper than the Transformer, it's just that the definition of a layer is not the same in the 2 architectures)
-for a Transformer on A100 80GB (d_model=512, n_layers=8, n_heads=8, batch_size=256), 30,000 steps = 18 min
"""

import os
import math
import time
import random
import string
from contextlib import nullcontext
from dataclasses import asdict
import json

import torch
import torch.nn.functional as F

import wandb

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.mamba.jamba import JambaConfig
from models.configuration_jamba import JambaConfig as JambaConfig_hf

from data import OthelloDataset
from eval import eval_legal_moves

# -------------------------------------------------------

# model parameters
architecture = "Jamba" # Transformer or Mamba
d_model = 288
n_layers = 8
bias = False

# Jamba specific
mlp_size = 864
inner_layernorms = False # not compatible with use_cuda

num_attn_heads = 6
num_key_value_heads = 6

num_experts = 1
num_experts_per_tok = 1

attn_layer_offset = 8
attn_layer_period = 200
expert_layer_offset = 1
expert_layer_period = 2

use_cuda = True # choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)

# Mamba specific
use_cuda = True # choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)

# Transformer specific
n_heads = 6
dropout = 0.
use_flash_attention = True

# training parameters
num_iters = 50000
batch_size = 256

lr = 1e-3
lr_min = 4e-5 # as in Mamba paper and Chinchilla
lr_warmup_iters = 100
lr_decay_iters = num_iters # num_iters as in Chinchilla

adam_b1 = 0.9
adam_b2 = 0.95

clip_value_grad = 1.0
weight_decay = 0.1

use_torch_compile = True # do not toggle if using Mamba

device = "cuda" # cpu, cuda:0, cuda:1, ...
dtype = "bfloat16" # float32, float16 or bfloat16 (float16 will use a GradScaler)

load_checkpoint = False
load_dir = "" # where to load from (if load_checkpoint is set)

save_dir = "runs/" # where to save to (ignored if load_checkpoint is set)

data_dir = "data/"

#checkpointing parameters
ckpt_interval = 10000

# logging parameters
log_wandb = True

train_log_interval = 50
eval_acc_interval = 1000
eval_val_interval = 200
eval_iters = 50

# -------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if log_wandb:
    wandb.init(project="othello",
            config={
                "architecture": architecture,
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
                "mlp_size": mlp_size,
                "inner_layernorms": inner_layernorms,
                "num_attn_heads": num_attn_heads,
                "num_key_value_heads": num_key_value_heads,
                "num_experts": num_experts,
                "num_experts_per_tok": num_experts_per_tok,
                "attn_layer_offset": attn_layer_offset,
                "attn_layer_period": attn_layer_period,
                "expert_layer_offset": expert_layer_offset,
                "expert_layer_period": expert_layer_period,
            })

if log_wandb:
    run_name = wandb.run.name
else:
    run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

if load_checkpoint:
    save_dir = load_dir
    print(f"Running with a loaded checkpoint. Will be saved in {save_dir}")
else:
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

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

# dataset
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

ds = OthelloDataset(train_dir)
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

ds_val = OthelloDataset(val_dir)
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=True)
iter_val = iter(loader_val)

# model
if architecture == "Transformer":
    config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, bias=bias, max_len=60, flash=use_flash_attention)
elif architecture == "Mamba":
    config = MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=use_cuda)
elif architecture == "Jamba":
    config = JambaConfig(d_model=d_model, n_layers=n_layers, mlp_size=mlp_size, inner_layernorms=inner_layernorms,
                         num_attention_heads=num_attn_heads, num_key_value_heads=num_key_value_heads, 
                         num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                         attn_layer_offset=attn_layer_offset, attn_layer_period=attn_layer_period,
                         expert_layer_offset=expert_layer_offset, expert_layer_period=expert_layer_period, use_cuda=use_cuda)
elif architecture == "Jamba_hf":
    config = JambaConfig_hf(vocab_size=65, hidden_size=d_model, intermediate_size=mlp_size, num_hidden_layers=n_layers,
                            num_attention_heads=num_attn_heads, num_key_value_heads=num_key_value_heads, use_cache=False, n_ctx=100,
                            num_experts_per_tok=num_experts_per_tok, num_experts=num_experts, expert_layer_offset=1, expert_layer_period=2,
                            attn_layer_offset=1, attn_layer_period=2, use_mamba_kernels=True)
else:
    raise NotImplementedError

# vocab_size being equal to 65 is a vestigial feature
# it should actually be 60 (moves) + 1 (padding) = 61
# (60 moves because the four center moves are never used as every game start with the same 4 pieces at center)
# but the OthelloGame from the original OthelloGPT recorded the 64 moves, so OthelloGame from othello.py here also do that
# in short, 4 tokens are never used
model = LM(config, vocab_size=65).to(device)
optim = model.configure_optimizers(weight_decay, lr, (adam_b1, adam_b2), device_type) # AdamW optim with weight_decay except for 1D params (biases, norms)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=="float16")) # needed when training with float16

print(f"Model initialized. Number of parameters : {sum([p.numel() for p in model.parameters()])}.")

if load_checkpoint:
    config_dir = os.path.join(load_dir, 'config.json')
    checkpoint_dir = os.path.join(load_dir, 'model.pth')

    config_json = json.load(open(config_dir))

    assert config_json['architecture'] == architecture, f"Hyperparameters in train.py are different than those found in loaded config (from {config_dir})"
    del config_json['architecture']

    if architecture == "Transformer":
        config_loaded = TransformerConfig(**config_json)
    elif architecture == "Mamba":
        config_loaded = MambaConfig(**config_json)
    else:
        raise NotImplementedError

    assert config == config_loaded, f"Hyperparameters in train.py are different than those found in loaded config (from {config_dir})"

    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

    checkpoint = None
    print(f"Successfully loaded checkpoint from {load_dir}.")

unoptimized_model = model # the unoptimized model is kept for saving
if use_torch_compile:
    print("Compiling the model...")
    model = torch.compile(model)
    print("Done compiling.")

print("Training is starting.")
start_time = time.time()

for iter, data in enumerate(loader):
    x, y = data
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
            acc = eval_legal_moves(unoptimized_model, device, 10) # evaluate on 10 games (unoptimized_model is faster)
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

    # checkpointing
    if iter % ckpt_interval == 0:
        os.makedirs(os.path.join(save_dir, f"ckpt_{iter}/"), exist_ok=True)
        checkpoint = {"model": unoptimized_model.state_dict()}
        torch.save(checkpoint, os.path.join(save_dir, f"ckpt_{iter}/model.pth"))

    if iter >= num_iters:
        break

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")

# saving : config + model checkpoint (model+optim+scaler)
config_dict = asdict(config)

if isinstance(config, TransformerConfig):
    config_dict['architecture'] = "Transformer"
elif isinstance(config, MambaConfig):
    config_dict['architecture'] = "Mamba"
else:
    raise NotImplementedError

json.dump(config_dict, open(os.path.join(save_dir, 'config.json'), 'w'))

checkpoint = {"model": unoptimized_model.state_dict(),
              "optimizer": optim.state_dict(),
              "scaler": scaler.state_dict()}
torch.save(checkpoint, os.path.join(save_dir, "model.pth"))

print(f"Successfully saved checkpoint and config in {save_dir}.")

model.eval()
final_acc = eval_legal_moves(unoptimized_model, device, 50)
model.train()
print(f"Final accuracy: {100.*final_acc:.2f}%")

# final logging (some metrics for wandb)
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
