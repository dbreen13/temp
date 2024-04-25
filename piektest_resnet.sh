#!/bin/bash
#train limits

#resnet18 cifar10
#baseline
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/train_baseline.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --index=$i --epochs=25 --gamma=0 --milestones=None --lr=1e-5;
done; 

#experiments
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.5-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.5 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.5-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.5 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.9-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 


for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.1-63.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tucker-r0.1-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-tt-r0.1-54.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/rn18/cifar10/decompose/dec-cp-r0.1-63.60.54.51.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i;
done; 