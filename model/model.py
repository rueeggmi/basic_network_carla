import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np


'''class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)'''


class CarlaModel(nn.Module):
    def __init__(self):
        super(CarlaModel, self).__init__()

        """Conv Block """
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        """image FC layers"""
        self.img_fc = nn.Sequential(
            nn.Linear(1228800, 512),  # check if 1228800 is correct here, used to be 70400
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        """--- Speed (measurements) ---"""
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        """--- Joint part ---"""
        self.joint_fc = nn.Sequential(
            nn.Linear(640, 512),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        """branches"""
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        # TODO understand speed branch and why we should use it && implement it.

    def forward(self, sample):
        img, speed = sample

        print(img.shape)
        img = self.conv_block(img)
        print(img.shape)
        #img = img.view(img.size(0), -1)  # Reshape
        img = img.view(-1, np.prod(img.shape[1:]))  # Reshape
        # Reshaping in tensorflow:
        # img = tf.reshape(img, [-1, int(np.prod(img.get_shape()[1:]))], name='reshape')
        print(img.shape)
        img = self.img_fc(img)

        speed = speed.float()
        print(type(speed))
        speed = self.speed_fc(speed)

        j = torch.cat([img, speed], 1)
        j = self.joint_fc(j)

        output = torch.stack([out(j) for out in self.branches], dim=0)
        return output