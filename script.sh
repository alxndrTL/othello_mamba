#!/bin/bash

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=12
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=12

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=11
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=11

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=10
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=10

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=9
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=9

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=8
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=8

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=7
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=7

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=6
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=6

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=5
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=5

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=4
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=4

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=3
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=3

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=2
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=2

python create_data_probing.py --load_dir=runs/lucky-deluge-39/ --layer=1
python train_probe.py --load_dir=runs/lucky-deluge-39/ --layer=1