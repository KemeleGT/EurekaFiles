import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import warnings
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils as vutils
import os
import shutil
# import model
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from IPython import embed
import datetime
import struct
import random
import sys
import matplotlib.pyplot as plt
import numpy as np



error_num = 100
distance_level = "mantissa"
# distance_level = "exponent"

bit_num = 5


def str2bool(v):
    return v.lower() in ('true')


def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    integers = [c for c in packed]
    binaries = [bin(i) for i in integers]
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    return ''.join(padded)


def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
    """Turn input tensor into float.
        Args:
            b : binary tensor. The last dimension of this tensor should be the
            the one the binary is at.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
        Returns:
            Tensor: Float tensor. Reduces last dimension.
    """
    expected_last_dim = num_m_bits + num_e_bits + 1
    assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                             "should be {}, not {}.".format(
        expected_last_dim, b.shape[-1])

    # check if we got the right type
    dtype = torch.float32
    if expected_last_dim > 32: dtype = torch.float64
    if expected_last_dim > 64:
        warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                      " this in mind. Your result will be not exact.")

    s = torch.index_select(b, -1, torch.arange(0, 1))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                               1 + num_e_bits + num_m_bits))
    # SIGN BIT
    out = ((-1) ** s).squeeze(-1).type(dtype)
    # EXPONENT BIT
    exponents = -torch.arange(-(num_e_bits - 1.), 1.)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
    out *= 2 ** e_decimal
    # MANTISSA
    matissa = (torch.Tensor([2.]) ** (
        -torch.arange(1., num_m_bits + 1.))).repeat(
        m.shape[:-1] + (1,))
    out *= 1. + torch.sum(m * matissa, dim=-1)
    return out




def easy_fault_apply(x: torch.Tensor, error_num, distance_level, bit_num):
    # print('distance_level is ', distance_level)
    # print('bit num is ', bit_num)
    if distance_level == 'mantissa':
        assert bit_num <= 23, 'bit num out of range'
    if distance_level == 'exponent':
        assert bit_num <= 8, 'bit num out of range'
    if distance_level == 'sign':
        assert bit_num == 1, 'bit num out of range'
    error_array = []
    for i in range(0, error_num):
        ran_num0 = random.randint(0, len(x) - 1)
        ran_num1 = random.randint(0, len(x[0]) - 1)
        ran_num2 = random.randint(0, len(x[0][0]) - 1)
        ran_num3 = random.randint(0, len(x[0][0][0]) - 1)
        error_array.append([ran_num0, ran_num1, ran_num2, ran_num3])
    for i in range(0, len(error_array)):
        error_dim0 = error_array[i][0]
        error_dim1 = error_array[i][1]
        error_dim2 = error_array[i][2]
        error_dim3 = error_array[i][3]
        num = str(binary(x[error_dim0][error_dim1][error_dim2][error_dim3]))
        # print('len(num) is ', len(num))
        sign = num[len(num) - 1]
        exponent = num[23:len(num) - 1]
        mantissa = num[0:23]

        bit_idx = []

        if distance_level == 'mantissa':
            while len(bit_idx) < bit_num:  # for i in range(0,bit_num):
                random_num = random.randint(0, len(mantissa) - 1)
                if random_num not in bit_idx:
                    bit_idx.append(random.randint(0, len(mantissa) - 1))
            for i in bit_idx:
                mantissa = list(mantissa)
                if mantissa[i] == '0':
                    mantissa[i] = '1'
                else:
                    mantissa[i] = '0'
                mantissa = ''.join(mantissa)
        elif distance_level == 'exponent':
            while len(bit_idx) < bit_num:  # for i in range(0,bit_num):
                random_num = random.randint(0, len(exponent) - 1)
                if random_num not in bit_idx:
                    bit_idx.append(random.randint(0, len(exponent) - 1))
            for i in bit_idx:
                exponent = list(exponent)
                if exponent[i] == '0':
                    exponent[i] = '1'
                else:
                    exponent[i] = '0'
                exponent = ''.join(exponent)
        elif distance_level == 'sign':
            if sign == '0':
                sign = '1'
            else:
                sign = '0'
        new_binary_num = sign + exponent + mantissa
        # print('len(new_binary_num) is ', len(new_binary_num))
        new_float_num = bit2float(torch.from_numpy(np.array(list(map(int, list(new_binary_num))))))
        # print('almost done for this one')
        x[error_dim0][error_dim1][error_dim2][error_dim3] = new_float_num
    return x


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.conv2(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.layer1(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.layer2(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.layer3(out)
        # out = easy_fault_apply(out, error_num, distance_level, bit_num)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # For updating learning rate
# def update_lr(optimizer, lr):    
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # # Train the model
# total_step = len(train_loader)
# curr_lr = learning_rate
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#     # Decay learning rate
#     if (epoch+1) % 20 == 0:
#         curr_lr /= 3
#         update_lr(optimizer, curr_lr)
model.load_state_dict(torch.load("resnet.ckpt"))
# Test the model
model.eval()

avg = 0
for i in range(10):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg += 100.0 * correct*1.00 / total
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
# print("avg acc for 10 runs: ", avg/10.0, error_num, distance_level, bit_num)
# Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')