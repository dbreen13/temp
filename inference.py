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

cifar_testset = datasets.CIFAR10(root='/home/dbreen/Documents/tddl/bigdata/cifar10', train=False, download=False, transform=None)

logger=logging.getLogger('Inference')
#create a fh
fh=logging.FileHandler('inference.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


#fact-cp-0.1-lay[44]
path="/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[44]/runnr1/rn18-lr-[44]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth"   
model=torch.load(path)

#fact-cp-0.1-lay[54,51]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[54, 51]/runnr1/rn18-lr-[54, 51]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-cp-0.1-lay[60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[60]/runnr1/rn18-lr-[60]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-cp-0.1-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63]/runnr1/rn18-lr-[63]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-cp-0.1-lay[63,60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63,60]/runnr1/rn18-lr-[63,60]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-cp-0.1-lay[63,60,44,41]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.1-lay[63,60,44,41]/runnr1/rn18-lr-[63,60,44,41]-cp-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-cp-0.9-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-cp-r0.9-lay[63]/runnr1/rn18-lr-[63]-cp-0.9-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tt-0.19-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[63]/runnr1/rn18-lr-[63]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tt-0.19-lay[60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[60]/runnr1/rn18-lr-[60]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tt-0.19-lay[44]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r0.19-lay[44]/runnr1/rn18-lr-[44]-tt-0.19-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tt-80-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tt-r80-lay[63]/runnr1/rn18-lr-[63]-tt-80-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tucker-0.9-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.9-lay[63]/runnr1/rn18-lr-[63]-tucker-0.9-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tucker-0.1-lay[63]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[63]/runnr1/rn18-lr-[63]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tucker-0.1-lay[44]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[44]/runnr1/rn18-lr-[44]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)

#fact-tucker-0.1-lay[60]
path='/media/jkooij/d63a895a-7e13-4bf0-a13d-1a6678dc0e38/dbreen/bigdata/cifar10/logs/rn18/decomposed/fact-tucker-r0.1-lay[60]/runnr1/rn18-lr-[60]-tucker-0.1-dTrue-iNone_bn_128_sgd_l1e-05_g0.0_sTrue/fact_model_best.pth' 
model=torch.load(path)