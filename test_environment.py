from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

from dataset import ImageDataset

carla_dataset = ImageDataset(root='/data/scratch/michelle/e2depth_dataset/datacollector/dataset_images2')

# fig = plt.figure()

sample = carla_dataset[0]
measurements = sample[1]
# cv2.imshow('image', sample[0])
print(measurements['measurements'])
meas = measurements['measurements']

print("Speed: ", meas[0][0])
print("Throttle: ", meas[4])
print("Brake: ", meas[6])
print("Steer: ", meas[5])

# for i in range(len(carla_dataset)):
#    sample = carla_dataset[i]

#    print("Sample ", i, ": ", sample)
'''print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break'''
