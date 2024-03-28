#!/bin/bash

CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/train_baseline.yml";
python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --epochs=5;

CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/train_garipov.yml";
python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --epochs=5;

CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/train_garipov.yml";
python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --epochs=5;

