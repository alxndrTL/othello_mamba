import os
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ProbingDataset
from othello import OthelloGame
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.lm import LM

# -------------------------------------------------------

layer = 7
load_dir = "runs/fanciful-resonance-26/" # run directory
dir_activations = None # if None, will default to load_dir/data_probing/layer_{layer}
dir_boards = None # if None, will default to load_dir/data_probing

batch_size = 256
num_iters = 20000

num_games = 100 # number of games to compute acc

# probe training parameters
lr = 1e-4
weight_decay = 0.01
adam_b1 = 0.9
adam_b2 = 0.99

print_interval = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------

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
    del config_json['architecture']
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

    if iter % print_interval == 0:
        num_digits = len(str(num_iters))
        formatted_iter = f"{iter:0{num_digits}d}"
        print(f"Step {formatted_iter}/{num_iters}. train loss = {loss.item():.3f}")

    if iter >= num_iters:
        break

print("Training done.")

n_games = 100

cell_acc = 0
board_acc = 0

for _ in range(n_games):
    moves = []
    boards = []

    game = OthelloGame()
    for t in range(60):
        legal_moves = game.get_valid_moves()
        if legal_moves == []:
            break

        move = random.choice(legal_moves)
        game.play_move(move)
        moves.append(move)
        
        board = torch.from_numpy(game.state.copy()).flatten()
        if game.next_hand_color == -1:
            board[board == 1] = 2
            board[board == -1] = 1
        else:
            board[board == -1] = 2
        boards.append(board)

    x = torch.tensor(moves)+1
    x = x.to(device).unsqueeze(0)
    activations = model.forward_up_to(x, layer) # (B=1, 59, d_model)

    preds = torch.argmax(probe(activations).view(-1, 64, 3), dim=-1)[5:-4, :]
    boards = torch.cat(boards).to(device).view(-1, 64)[5:-4, :]

    cell_acc += torch.mean((boards == preds).float()).item() # mean cell accuracy
    board_acc += torch.mean((boards == preds).all(dim=1).float()).item() # mean board accuracy

cell_acc /= n_games
board_acc /= n_games

print(f"Mean cell accuracy: {100*cell_acc:.2f}% (vs {66}% for an untrained model)")
print(f"Mean board accuracy: {100*board_acc:.2f}% (vs {0}% for an untrained model)")

# its important to compare the results with the "trained probe on an untrained model" setup :
# 1) untrained probe on trained model   : 33%, 0%
# 2) untrained probe on untrained model : 33%, 0%
# 3) trained probe on untrained model   : 66%, 0% (most important)
