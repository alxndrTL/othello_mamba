"""
Evaluate an OthelloGPT or MambaGPT on the legality of its moves.
It uses games sampled from {data_dir]/val. At each step of the game, the legality of the move predicted by the model is evaluated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from othello import OthelloGame
from data import OthelloDataset

# todo : use LM inference function (and not forward)

def eval(model: nn.Module, device, n_games: int, data_loader: torch.utils.data.DataLoader = None, dir: str = "data/val"):
    """
    Returns the percentage of moves predicted by model which are legal.
    Uses data from data_loader if provided, else fallbacks to dir.

    The -1s you see here are because the original tokenized moves are from -1 to 63 (-1 being the padding).
    We actually offset them by +1 (in data.py) because we can't feed -1 to an nn.Embedding.
    Hence, when communicating with our OthelloGame, we need to go back to the -1 to 63 range.
    """

    if data_loader is None:
        dataset = OthelloDataset(dir)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    total_moves = 0
    total_legal_moves = 0

    for i, data in enumerate(data_loader):
        game_transcript, _ = data # (1, lengame)
        game_transcript = game_transcript[[0]] # (1, lengame)
        game_transcript = game_transcript.squeeze().int().to(device) # if sent to model : +1 then -1 after the pass

        game_len = game_transcript.shape[0] # always 60-1=59 (padded)
        game = OthelloGame()
        game.play_move(game_transcript[0].item() - 1)
        for pgame_len in range(1, game_len):
            context = game_transcript[:pgame_len]

            # get legal moves given current game board
            legal_moves = game.get_valid_moves()
            if legal_moves == []:
                break
            
            # sample a move
            x = context[None, ...] # (1, pgame_len)
            logits = model(x)[:, -1, :] # (1, vocab_size)
            probs = F.softmax(logits, dim=-1) # (1, vocab_size)
            #move = torch.multinomial(probs, num_samples=1).item() - 1
            move = torch.argmax(probs, dim=-1).item() - 1

            if move in legal_moves:
                total_legal_moves += 1
            total_moves += 1

            game.play_move(game_transcript[pgame_len].item() - 1)

        if i >= n_games-1:
            break
    
    return total_legal_moves/total_moves

if __name__ == "__main__":
    raise NotImplementedError