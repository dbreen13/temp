#!/bin/bash
#train limits

#resnet18 cifar10
#baseline
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/train_garipov.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --index=$i --epochs=10 --gamma=0 --milestones=None --lr=1e-4 --cuda='0';
done; 

#experiments
for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.1-10.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.9-10.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.5-10.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.5 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.1-4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.9-4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.9 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.5-4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.5 --index=$i --cuda='0';
done; 


for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-tucker-r0.1-10.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-tt-r0.1-10.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-tucker-r0.1-4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-tt-r0.1-4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=4 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.1-10.8.6.4.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 

for i in {1..5};do
    CONFIG_PATH="/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/cifar10/decompose/dec-cp-r0.1-10.8.yml";
    python train.py main --config-path "$CONFIG_PATH" --data_workers=8 --rank=0.1 --index=$i --cuda='0';
done; 