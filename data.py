"""
This script provides the data for the training script.
It is supposed that you have already run prepare_data.py, which converts the original Othello dataset to .bin files, usable by this script.
"""

import os

import random
import numpy as np
import torch

class OthelloDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir: str = "data/train"):
        # dir contains the .bin files created by prepare_data.py
        # each files contains some numbers (around 100K) of tokenized games, each of len 60
        super().__init__()

        self.dir = dir

    def __iter__(self):
        # executed by each worker, when data are requested
        # returns one batch ie one game

        chunks_files = [os.path.join(self.dir, file) for file in os.listdir(self.dir) if file.endswith('.bin')]

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        seed = 123456 + worker_id
        rng = random.Random(seed)

        while True:
            rng.shuffle(chunks_files)
            for chunk_file in chunks_files:
                chunk = np.memmap(chunk_file, dtype=np.int8, mode='r')
                num_games = chunk.shape[0] // 60

                game_start_indices = 60 * np.arange(num_games)
                np.random.shuffle(game_start_indices)

                for indice in game_start_indices:
                    start = indice
                    end = start + 60
                    
                    # as the tokenized move are from -1 to 63, we feed to the model 0 to 64 (index -1 should not by used with nn.Embedding)
                    data = torch.from_numpy(chunk[start:end].copy()) + 1
                    x = data[:-1].int()
                    y = data[1:].long()

                    yield x, y
