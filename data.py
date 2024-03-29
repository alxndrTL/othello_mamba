"""
This script provides the data for both training scripts : train.py and train_probe.py.
It is supposed that you have already downloaded the data as described in the README.me/Getting started.
"""

import os

import random
import numpy as np
import torch

"""
This dataset is used to sample Othello games.
It is used in :
- train.py to train the base model
- create_data_probing.py to get (activations, board) data for the probe training

By default, it uses data in data/train folder. Passing data/val uses the validation data.
"""
class OthelloDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir: str = "data/train", seed: int = None):
        # dir contains the .bin files created by prepare_data.py
        # each files contains some numbers (around 100K) of tokenized games, each of len 60

        # seed is used by create_data_probing.py to get the same batches when collecting activations and boards
        # setting a seed allows to sample the same batches when collecting games from two different endpoints
        # ie for i, data in enumerate(loader_val) will give the same batches when called multiple times

        super().__init__()

        self.dir = dir
        self.seed = seed

    def __iter__(self):
        # executed by each worker, when data are requested
        # returns one game (ie one training example)

        # every .bin files is a 60*N array, N being the number of games per file (approx. 100K)
        chunks_files = [os.path.join(self.dir, file) for file in os.listdir(self.dir) if file.endswith('.bin')]

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        rng = random.Random(123456 + worker_id)

        if self.seed:
            rng_numpy = np.random.default_rng(self.seed)
        else:
            rng_numpy = np.random.default_rng()

        while True:
            rng.shuffle(chunks_files)
            for chunk_file in chunks_files:
                chunk = np.memmap(chunk_file, dtype=np.int8, mode='r') # read a .bin file
                num_games = chunk.shape[0] // 60

                game_start_indices = 60 * np.arange(num_games) # get all the indices on which games start (all games are padded to a lenght of 60)
                rng_numpy.shuffle(game_start_indices)

                for indice in game_start_indices:
                    start = indice
                    end = start + 60
                    
                    # as the tokenized move are from -1 to 63, we feed to the model 0 to 64 (index -1 should not by used with nn.Embedding)
                    data = torch.from_numpy(chunk[start:end].copy()) + 1
                    x = data[:-1].int() # classic shifting
                    y = data[1:].long() # long() is necessary for the CE loss

                    yield x, y

"""
This dataset is used to sample (activations, board) training examples for the training of the probe, in train_probe.py.
The logic is the same as the dataset above.

You have to specify a directory where the activations files are stored, as well as the board files (automatic in train_probe.py)
"""
class ProbingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir_activations: str, dir_boards: str):
        super().__init__()

        self.dir_activations = dir_activations
        self.dir_boards = dir_boards

    def __iter__(self):

        files_activations = sorted([os.path.join(self.dir_activations, file) for file in os.listdir(self.dir_activations) if file.endswith('.npy')])
        files_boards = sorted([os.path.join(self.dir_boards, file) for file in os.listdir(self.dir_boards) if file.endswith('.npy')])

        files_indices = list(range(len(files_activations)))
        rng = random.Random()
        rng.shuffle(files_indices)

        while True:
            for index in files_indices:
                activations = np.load(files_activations[index]) # (B, 59, d_model) we only get games of len 59 because the model only sees 59 moves as input
                boards = np.load(files_boards[index]) # (B, 59, 8*8)

                activations = activations.reshape(-1, activations.shape[2]) # (B*59, d_model)
                boards = boards.reshape(-1, boards.shape[2]) # (B*59, 8*8)

                sample_indices = list(range(activations.shape[0]))
                rng.shuffle(sample_indices)
                for sample_index in sample_indices:
                    yield activations[sample_index], boards[sample_index]
