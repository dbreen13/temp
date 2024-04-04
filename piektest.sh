#!/bin/bash
#train limits
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/train_baseline.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --epochs=25 --index=$i --gamma=0 --lr=1.0e-5;
done;

for i in {1..5}; do
   CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.yml";
   python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i ;
done;

for i in {1..5};do
   CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.yml";
   python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done;

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.44.41.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done;

