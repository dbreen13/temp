#!/bin/bash
#train limits

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 

for i in {1..5}; do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i ;
done;

for i in {1..2};do
    python train.py main --config-path /home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.1-63.yml --data_workers=4 --index=$i;  
done;

for i in {1..2};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --epochs=1;
done;

for i in {1..2};do
   CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.44.41.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --epochs=1;
done;


