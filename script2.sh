#!/bin/bash

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=1
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=1

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=2
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=2

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=3
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=3

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=4
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=4

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=5
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=5

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=6
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=6

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=7
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=7

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=8
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=8

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=9
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=9

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=10
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=10

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=11
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=11

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=12
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=12

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=13
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=13

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=14
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=14

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=15
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=15

python create_data_probing.py --load_dir=runs/sparkling-butterfly-50/ --layer=16
python train_probe.py --load_dir=runs/sparkling-butterfly-50/ --layer=16