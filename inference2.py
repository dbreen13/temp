#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:40:19 2024

@author: dbreen

import torchvision

"""

import torchvision
import torchvision.datasets as datasets
import logging
import torch
import torchvision.transforms as transforms
import time as timers

mean_CIFAR10 = [0.49139968, 0.48215841, 0.44653091]
std_CIFAR10 = [0.24703223, 0.24348513, 0.26158784]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean_CIFAR10,
            std_CIFAR10,
        ),
    ])

testset = datasets.CIFAR10(root='/home/dbreen/Documents/tddl/bigdata/cifar10', train=True,download=False, transform=transform)
batch_size=128
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logger=logging.getLogger('Inference')
#create a fh
fh=logging.FileHandler('inference.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# #baseline
# path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/baselines/baseline-rn18-cifar10/runnr1/rn18_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue/cnn_best.pth"   
# model=torch.load(path)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-base-rn18-cif' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-base-rn18-cif' )
#     timers.sleep(120)    
    
# # #fact-cp-0.1-lay[44]
# path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[44]/runnr1/rn18-lr-[44]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth"   
# model=torch.load(path)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-cp-0.1-lay[44]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-cp-0.1-lay[44]' )
#     timers.sleep(120)
        

# # #fact-cp-0.1-lay[54,51]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[54, 51]/runnr1/rn18-lr-[54, 51]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-cp-0.1-lay[54,51]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-cp-0.1-lay[54,51]' )
#     timers.sleep(120)


# # #fact-cp-0.1-lay[60]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[60]/runnr1/rn18-lr-[60]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-cp-0.1-lay[60]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-cp-0.1-lay[60]' )
#     timers.sleep(120)


# # #fact-cp-0.1-lay[63]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63]/runnr1/rn18-lr-[63]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-cp-0.1-lay[63]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-cp-0.1-lay[63]' )
#     timers.sleep(120)

# #fact-cp-0.1-lay[63,60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63, 60]/runnr1/rn18-lr-[63, 60]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-cp-0.1-lay[63,60]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-cp-0.1-lay[63,60]' )
    timers.sleep(120)


# #fact-cp-0.1-lay[63,60,44,41]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63, 60, 44, 41]/runnr1/rn18-lr-[63, 60, 44, 41]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-cp-0.1-lay[63,60,44,41]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-cp-0.1-lay[63,60,44,41]' )
    timers.sleep(120)


# # #fact-cp-0.9-lay[63]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.9-lay[63]/runnr1/rn18-lr-[63]-cp-0.9-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)    
#     print('start')
#     logger.info('start-inf-cp-0.9-lay[63]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-cp-0.9-lay[63]' )
#     timers.sleep(120)


# # #fact-tucker-0.9-lay[63]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.9-lay[63]/runnr1/rn18-lr-[63]-tucker-0.9-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-tucker-0.9-lay[63]' )
#     for i in range(80):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         i+=1
#     logger.info('end-inf-tucker-0.9-lay[63]' )
#     timers.sleep(120)gar_18_dNone_128_adam_l0.001_g0.1_w0.0_sTrue
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():


# #fact-tucker-0.1-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[63]/runnr1/rn18-lr-[63]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tucker-0.1-lay[63]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tucker-0.1-lay[63]' )
    timers.sleep(120)

# #fact-tucker-0.1-lay[44]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[44]/runnr1/rn18-lr-[44]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tucker-0.1-lay[44]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tucker-0.1-lay[44]' )
    timers.sleep(120)

# #fact-tucker-0.1-lay[60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[60]/runnr1/rn18-lr-[60]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tucker-0.1-lay[60]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tucker-0.1-lay[60]' )
    timers.sleep(120)


batch_size=128
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)
# #fact-tt-0.19-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[63]/runnr1/rn18-lr-[63]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tt-0.19-lay[63]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tt-0.19-lay[63]' )
    timers.sleep(120)

# #fact-tt-0.19-lay[60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[60]/runnr1/rn18-lr-[60]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tt-0.19-lay[60]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tt-0.19-lay[60]' )
    timers.sleep(120)
    
    
# #fact-tt-0.19-lay[44]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[44]/runnr1/rn18-lr-[44]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-tt-0.19-lay[44]' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-tt-0.19-lay[44]' )
    timers.sleep(120)

# # #fact-tt-80-lay[63]
# path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r80.0-lay[63]/runnr1/rn18-lr-[63]-tt-80.0-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
# model=torch.load(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # since we're not training, we don't need to calculate the gradients for our outputs
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     timers.sleep(120)
#     print('start')
#     logger.info('start-inf-tt-80-lay[63]' )
#     for i in range(3):
#         for data in testloader:
#             images, labels = data
#             images = images.to(device)  # Move input data to the same device as the model
#             labels = labels.to(device)  # Move labels to the same device as the model
#             # calculate outputs by running images through the network
#             outputs = model(images)
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item(
#         i+=1
#     logger.info('end-inf-tt-80-lay[63]' )
#     timers.sleep(120)

#garipov
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/garipov/baselines/baseline-gar-cifar10-b128/runnr1/gar_18_dNone_128_sgd_l0.1_g0.1_w0.0_sTrue/cnn_best.pth"   
model=torch.load(path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-inf-base-gar-cif' )
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-inf-base-gar-cif' )
    timers.sleep(120)    
 
#garipov cp-0.1-[10]
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/garipov/decomposed/fact-cp-r0.1-lay[10]/runnr1/gar-lr-[10]-cp-0.1-dTrue-iNone_bn_128_sgd_l0.0001_g0.0_sTrue/cnn_best.pth"
model=torch.load(path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-gar-inf-tt-0.19-lay[10]')
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-gar-inf-tt-0.19-lay[10]' )
    timers.sleep(120) 

#garipov cp-0.1-[10,8]
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/garipov/decomposed/fact-cp-r0.1-lay[10, 8]/runnr1/gar-lr-[10, 8]-cp-0.1-dTrue-iNone_bn_128_sgd_l0.0001_g0.0_sTrue/cnn_best.pth"
model=torch.load(path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-gar-inf-tt-0.19-lay[10, 8]')
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-gar-inf-tt-0.19-lay[10, 8]' )
    timers.sleep(120) 

#garipov cp-0.1-[10]
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/garipov/decomposed/fact-cp-r0.1-lay[10, 8, 6, 4]/runnr1/gar-lr-[10, 8, 6, 4]-cp-0.1-dTrue-iNone_bn_128_sgd_l0.0001_g0.0_sTrue/cnn_best.pth"
model=torch.load(path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    timers.sleep(120)
    print('start')
    logger.info('start-gar-inf-tt-0.19-lay[10, 8, 6, 4]')
    for i in range(80):
        for data in testloader:
            images, labels = data
            images = images.to(device)  # Move input data to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        i+=1
    logger.info('end-gar-inf-tt-0.19-lay[10, 8, 6, 4]' )
    timers.sleep(120) 
    
