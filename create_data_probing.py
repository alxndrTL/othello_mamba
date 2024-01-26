import os

import math
import numpy as np
import torch

from data import OthelloDataset
from othello import OthelloGame
from models.transformer.transformer import TransformerConfig
from models.lm import LM

# -------------------------------------------------------

total_games = 1000
batch_size = 128 # each file will contain batch_size games
layer = 7
save_dir = "data_probing/"
data_dir = "data/val"

d_model = 512
n_layers = 8
n_heads = 8

dropout = 0.
bias = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, bias=bias, max_len=60, flash=True)
model = LM(config, vocab_size=65).to(device)

ds_val = OthelloDataset(data_dir, seed=47)
loader_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=True)

# ************** ACTIVATIONS dataset **************

save_dir_activations = os.path.join(save_dir, f"layer_{layer}")
os.makedirs(save_dir_activations, exist_ok=True)

print(f"Creating and saving dataset of activations for layer={layer} (saving dir: {save_dir_activations})...")

print(f"Number of files : {math.ceil(total_games/batch_size)}")
print(f"Size of each file: {4*batch_size*59*config.d_model/1e6:.1f} MB")
print(f"Size of activations dataset : {math.ceil(total_games/batch_size)*4*batch_size*59*config.d_model/1e6:.1f} MB")

num_games = 0
for i, data in enumerate(loader_val):
    x, _ = data # (B, 59)
    x = x.to(device)

    activations = model.forward_up_to(x, layer).detach().cpu().numpy() # (B, 59, d_model)
    np.save(os.path.join(save_dir_activations, f"batch_{i+1}_activations.npy"), activations)

    num_games += batch_size
    if num_games >= total_games:
        break

print(f"Done creating and saving the activations dataset.")

# ************** BOARDS dataset **************

print("Creating and saving dataset of boards...")

print(f"Number of files : {math.ceil(total_games/batch_size)}")
print(f"Size of each file: {4*batch_size*59*64/1e6:.1f} MB")
print(f"Size of boards dataset : {math.ceil(total_games/batch_size)*4*batch_size*59*64/1e6:.1f} MB")

num_games = 0
for i, data in enumerate(loader_val):
    x, _ = data # (B, 59)
    x = x.to(device)

    boards = np.zeros((batch_size, 59, 8*8), dtype=np.int32)

    for k in range(batch_size):
        game_transcript = x[k] # (59)

        game = OthelloGame()
        for t in range(0, 59):
            move = game_transcript[t].item() - 1
            if move == -1:
                print("ee")
                boards[k, t] = -100 * np.ones((8*8,), dtype=np.int32)
            else:
                game.play_move(game_transcript[t].item() - 1)
                boards[k, t] = np.copy(game.state).flatten()

    np.save(os.path.join(save_dir, f"batch_{i+1}_boards.npy"), boards)

    num_games += batch_size
    if num_games >= total_games:
        break

print(f"Done creating and saving the boards dataset.")
