"""
Evaluate an OthelloGPT or MambaGPT on the legality of its moves.
It uses games sampled from {data_dir]/val. At each step of the game, the legality of the move predicted by the model is evaluated.
"""

import os
import argparse
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello import OthelloGame

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig

# todo : use LM inference function (and not forward)

def eval_legal_moves(model: nn.Module, device, n_games: int, sample: bool = False):
    """
    Returns the percentage of moves predicted by {model} which are legal.
    Plays {n_games}, and evaluate the accuracy.

    The -1's you see here are because the original tokenized moves are from -1 to 63 (-1 being the padding).
    We actually offset them by +1 (in data.py) because we can't feed -1 to an nn.Embedding.
    Hence, when communicating with our OthelloGame, we need to go back to the -1 to 63 range.
    """

    total_moves = 0
    total_legal_moves = 0

    for _ in range (n_games):
        game = OthelloGame()
        moves = []
        for _ in range(60):
            legal_moves = game.get_valid_moves()
            if legal_moves == []:
                break
            
            # t >= 1 (the model never makes the first move)
            if moves:
                context = torch.tensor(moves).to(device) + 1 # (L)
                context = context.unsqueeze(0) # (1, L)
                logits = model(context)[0, -1] # (vocab_size)
                probs = F.softmax(logits, dim=0)

                if sample:
                    move = torch.multinomial(probs, num_samples=1).item() - 1
                else:
                    move = torch.argmax(probs, dim=0).item() - 1

                if move in legal_moves:
                    total_legal_moves += 1
                total_moves += 1
            
            move = random.choice(legal_moves)
            game.play_move(move)
            moves.append(move)
    
    return total_legal_moves/total_moves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_dir", type=str, help="something like runs/name_run/")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--n_games", type=int, default=50, help="number of games to play to evaluate acc")
    parser.add_argument("--sample", type=bool, default=False, help="whether to sample or simply take the most probable move")

    args = parser.parse_args()

    config_dir = os.path.join(args.load_dir, 'config.json')
    checkpoint_dir = os.path.join(args.load_dir, 'model.pth')

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

    model = LM(config, vocab_size=65).to(args.device)

    checkpoint = torch.load(checkpoint_dir, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    print(f"Successfully loaded checkpoint from {args.load_dir}.")
    model.eval()

    acc = eval_legal_moves(model, args.device, n_games=args.n_games)
    print(f"Legal move accuracy: {100.*acc:.2f}%")
