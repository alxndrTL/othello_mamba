"""
This script provides two evaluate functions.
All the evaluations sample new random Othello games.

-eval_legal_moves : evaluates a base model on the legality of its moves. It uses games sampled from {data_dir]/val. 
                    At each step of the game, the legality of the move predicted by the model is evaluated.
                    The functions returns the legal move accuracy.
-eval_probe_accuracy : evaluates a (model, probe). It provides 2 accuracy :
                       cell accuracy, which is the proportion of cells which are correctly classified/predicted
                       board accuracy, which is the proportion of boards which are correctly classified/predicted
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
# (Transformer: KV cache, Mamba: inference which only carry along a hidden state and the last d_conv-1 inputs)

def eval_legal_moves(model: nn.Module, device, n_games: int, sample: bool = False):
    """
    Returns the percentage of moves predicted by {model} which are legal.
    Plays {n_games}, and evaluate the accuracy.
    """

    """
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

def eval_probe_accuracy(model: nn.Module, probe: nn.Module, layer: int, device, n_games: int):
    """
    Returns the cell and board accuracies of (model, probe), on newly sampled Othello games.
    Plays {n_games}, and evaluate the accuracy.
    """
    cell_acc = 0
    board_acc = 0

    for _ in range(n_games):
        moves = []
        boards = []

        game = OthelloGame()
        for _ in range(60):
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

        x = torch.tensor(moves) + 1
        x = x.to(device).unsqueeze(0)
        activations = model.forward_up_to(x, layer) # (B=1, 59, d_model)

        preds = torch.argmax(probe(activations).view(-1, 64, 3), dim=-1)
        boards = torch.cat(boards).to(device).view(-1, 64)

        cell_acc += torch.mean((boards == preds).float()).item() # mean cell accuracy
        board_acc += torch.mean((boards == preds).all(dim=1).float()).item() # mean board accuracy

    cell_acc /= n_games
    board_acc /= n_games

    return cell_acc, board_acc

""" to eval legal move accuracy from cmd line """
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
