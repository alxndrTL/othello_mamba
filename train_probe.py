import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ProbingDataset
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.lm import LM
from eval import eval_probe_accuracy

# -------------------------------------------------------

layer = 7
load_dir = None # run directory
dir_activations = None # if None, will default to load_dir/data_probing/layer_{layer}
dir_boards = None # if None, will default to load_dir/data_probing

save_dir = None # if None, will default to load_dir/probe_{layer}.pth

batch_size = 256
num_iters = 120000

n_games = 500 # number of games to compute acc

# probe training parameters
lr = 1e-4
weight_decay = 0.01
adam_b1 = 0.9
adam_b2 = 0.99

print_interval = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", type=str, default=None, help="something like runs/name_run/")
args = parser.parse_args()

if args.load_dir is not None:
    load_dir = args.load_dir

assert load_dir is not None, "Please provide the run path (either as an argument or in the file)"
    
if dir_activations is None:
    dir_activations = os.path.join(load_dir, "data_probing", f"layer_{layer}")

if dir_boards is None:
    dir_boards = os.path.join(load_dir, "data_probing")

ds = ProbingDataset(dir_activations=dir_activations, dir_boards=dir_boards)
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)

config_dir = os.path.join(load_dir, 'config.json')
checkpoint_dir = os.path.join(load_dir, 'model.pth')

config_json = json.load(open(config_dir))
architecture = config_json['architecture']
del config_json['architecture']

if architecture == "Transformer": 
    config = TransformerConfig(**config_json)
elif architecture == "Mamba":
    config = MambaConfig(**config_json)
else:
    raise NotImplementedError

model = LM(config, vocab_size=65).to(device)

checkpoint = torch.load(checkpoint_dir, map_location=device)
model.load_state_dict(checkpoint['model'])
print(f"Successfully loaded checkpoint from {load_dir}.")
model.eval()

class Probe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        
        self.fc = nn.Linear(d_model, 3*8*8, bias=True)
        # 3 = number of cell types (empty=0, yours=1, mine=2)
        # 8*8 = board size

    def forward(self, x):
        # x : (B, 512) -> y : (B, 3*8*8)
        return self.fc(x)
    
probe = Probe(config.d_model).to(device)
optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_b1, adam_b2))

print("Starting training...")

for iter, data in enumerate(loader):
    activations, boards = data
    activations, boards = activations.to(device), boards.to(device)
    boards = boards.long()

    logits = probe(activations)

    loss = F.cross_entropy(logits.view(-1, 3), boards.view(-1), ignore_index=-100)

    optim.zero_grad()
    loss.backward()
    optim.step()

    # printing
    if iter % print_interval == 0:
        cell_acc, board_acc = eval_probe_accuracy(model, probe, layer, device, n_games=10)

        num_digits = len(str(num_iters))
        formatted_iter = f"{iter:0{num_digits}d}"
        print(f"Step {formatted_iter}/{num_iters}. train loss = {loss.item():.3f}. mean cell acc = {100*cell_acc:.2f}%. mean board acc = {100*board_acc:.2f}%")

    if iter >= num_iters:
        break

print("Training done.")

save_dir = os.path.join(load_dir, f"probe_{layer}.pth")
checkpoint = {"probe": probe.state_dict()}
torch.save(checkpoint, save_dir)

print(f"Sucessfully saved trained probe in {save_dir}")

cell_acc, board_acc = eval_probe_accuracy(model, probe, layer, device, n_games=500)

print(f"Mean cell accuracy: {100*cell_acc:.2f}% (vs {66}% for an untrained model)")
print(f"Mean board accuracy: {100*board_acc:.2f}% (vs {0}% for an untrained model)")

# its important to compare the results with the "trained probe on an untrained model" setup :
# 1) untrained probe on trained model   : 33%, 0%
# 2) untrained probe on untrained model : 33%, 0%
# 3) trained probe on untrained model   : 66%, 0% (most important)
