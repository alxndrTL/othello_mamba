import os
import pickle
import numpy as np

"""
Convert all the pickles in {pickles_dir} and put them in {data_dir} :
- pad all the games to 60 with -1
- ready to open with np.memmap() (np.int8)
"""

pickles_dir = "pickles/"
data_dir = "data/"

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

pickles_files = os.listdir(pickles_dir)
train_threshold = int(len(pickles_files) * 0.8)

for i, filename in enumerate(pickles_files):
    with open(os.path.join(pickles_dir, filename), 'rb') as handle:
        games = pickle.load(handle)
        np_games = -1 * np.ones((len(games), 60), dtype=np.int8)

        for k, game in enumerate(games):
            len_game = len(game)
            np_games[k, :len_game] = game

        if i < train_threshold:
            output_dir = train_dir
        else:
            output_dir = val_dir

        np_games.tofile(os.path.join(output_dir, f"games_{i}.bin"))
    
    if i%10==0:
        print(f"Processing file {i}/{len(pickles_files)}")
