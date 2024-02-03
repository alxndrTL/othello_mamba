#!/bin/bash

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=9
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=9

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=8
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=8

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=7
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=7

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=6
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=6

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=5
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=5

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=4
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=4

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=3
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=3

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=2
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=2

python create_data_probing.py --load_dir=runs/deft-meadow-40/ --layer=1
python train_probe.py --load_dir=runs/deft-meadow-40/ --layer=1