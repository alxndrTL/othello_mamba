import os
import json
import argparse

import math
import numpy as np
import torch

from data import OthelloDataset
from othello import OthelloGame
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.lm import LM

# -------------------------------------------------------

total_games = 1000
batch_size = 48 # each file will contain batch_size games
layer = 10
load_dir = None # run directory
save_dir = None # if None : will default to load_dir/data_probing
data_dir = "data/val"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", type=str, default=None, help="something like runs/name_run/")
args = parser.parse_args()

if args.load_dir is not None:
    load_dir = args.load_dir

assert load_dir is not None, "Please provide the run path (either as an argument or in the file)"

if save_dir is None:
    save_dir = os.path.join(load_dir, "data_probing/")
    os.makedirs(save_dir, exist_ok=True)

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
                # move is -1 when we encounter a game that was padded (ended before the whole board is complete with pieces)
                # we put -100 for all pieces (it will be the ignore_index of the CE loss in the next training step)
                boards[k, t] = -100 * np.ones((8*8,), dtype=np.int32)
            else:
                game.play_move(game_transcript[t].item() - 1)
                board = np.copy(game.state).flatten()

                # board : (64,) avec 0's (empty), -1 (white), 1 (black)
                # game.next_hand_color : -1 (white) ou 1 (black)

                # si next_hand_color est -1 : on remplace les 1 par des 2, puis les -1 par des 1
                # si next_hand_color est 1 : on remplace les -1 par des 2

                # pour avoir :
                # board : (64,) avec 0's (empty), 1 (same color as next turn), 2 (diff color as next turn)

                if game.next_hand_color == -1:
                    board[board == 1] = 2
                    board[board == -1] = 1
                else:
                    board[board == -1] = 2

                boards[k, t] = board

    np.save(os.path.join(save_dir, f"batch_{i+1}_boards.npy"), boards)

    num_games += batch_size
    if num_games >= total_games:
        break

print(f"Done creating and saving the boards dataset.")
