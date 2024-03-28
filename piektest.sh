#!/bin/bash
#train limits
#for i in {1..5};do
#    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.9-63.yml";
#    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i;
#done; 

#for i in {1..5}; do
#    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-63.yml";
#    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i ;
#done;

#for i in {1..2};do
#    python train.py main --config-path /home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.9-63.yml --data_workers=4 --index=$i;  
#done;

#for i in {1..2};do
#    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-63.60.yml";
#    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i;
#done;

#for i in {1..2};do
#    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-63.60.44.41.yml";
#    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i;
#done;

#train baseline
for i in {1..2};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/train_baseline.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --epochs=25 --index=$i;
done;

for i in {1..2};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/train_garipov.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --epochs=10 --index=$i;
done;

for i in {1..2};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/train_garipov.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --epochs=10 --index=$i;
done;
