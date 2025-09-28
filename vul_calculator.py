from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from shutil import copyfile
from PIL import Image
import torch
import os
import shutil
import random
import re
import argparse

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--noise_level', type=float, default=0.5, help='noise level of dataset')  # necessory
parser.add_argument('--dest_class', type=str, default='6', help='dest_class')
parser.add_argument('--target_class', type=str, default='5', help='target_class')
parser.add_argument('--final_layer_name', type=str, default='final_conv', help='the name of final layer')
args = parser.parse_args()

test_origin1 = [line.rstrip() for line in open('./tool_origin_test1.log', 'r')]
test1 = [line.rstrip() for line in open('./tool_test1.log', 'r')]
test2 = [line.rstrip() for line in open('./tool_test2.log', 'r')]
bitflip1 = [line.rstrip() for line in open('./tool_hamming1.log', 'r')]
test3 = [line.rstrip() for line in open('./tool_test3.log', 'r')]
bitflip2 = [line.rstrip() for line in open('./tool_hamming2.log', 'r')]

classlinenum = 2
totallinenum = 7
##test_origin 1 accuracy:
class_accuracy_line = test_origin1[classlinenum]
total_accuracy_line = test_origin1[totallinenum]

class_accuracy_dic = eval(class_accuracy_line)
print(len(class_accuracy_dic))
print(class_accuracy_dic)
class_accuracy_dst1_origin = class_accuracy_dic[args.dest_class] #args.dest_class
class_accuracy_tar1_origin = class_accuracy_dic[args.target_class] #args.target_class
#print(class_accuracy_dic)
print('origin accuracy of target_class is ', class_accuracy_tar1_origin)
print('origin accuracy of dest_class is ', class_accuracy_dst1_origin)

total_accuracy1_origin = re.search('\s(100|(\d{1,2}(\.\d+)*))',total_accuracy_line)#%
#print(total_accuracy.group())
#print(type(total_accuracy.group()))
total_accuracy1_origin = float(total_accuracy1_origin.group())
print('accuracy of original dataset is ',total_accuracy1_origin)

##test 1 accuracy:
class_accuracy_line = test1[classlinenum]
total_accuracy_line = test1[totallinenum]

class_accuracy_dic = eval(class_accuracy_line)
class_accuracy_dst1 = class_accuracy_dic[args.dest_class]
class_accuracy_tar1 = class_accuracy_dic[args.target_class]
#print(class_accuracy_dic)
print('accuracy of target_class is ', class_accuracy_tar1)
print('accuracy of dest_class is ', class_accuracy_dst1)

total_accuracy1 = re.search('\s(100|(\d{1,2}(\.\d+)*))',total_accuracy_line)#%
#print(total_accuracy.group())
#print(type(total_accuracy.group()))
total_accuracy1 = float(total_accuracy1.group())
print('accuracy of original dataset is ',total_accuracy1)

##accuracy 2:
class_accuracy_line = test2[classlinenum]
total_accuracy_line = test2[totallinenum]

class_accuracy_dic = eval(class_accuracy_line)
class_accuracy_dst2 = class_accuracy_dic[args.dest_class]
class_accuracy_tar2 = class_accuracy_dic[args.target_class]
#print(class_accuracy_dic)
print('accuracy of target_class is ', class_accuracy_tar2)
print('accuracy of dest_class is ', class_accuracy_dst2)

total_accuracy2 = re.search('\s(100|(\d{1,2}(\.\d+)*))',total_accuracy_line)#%
#print(total_accuracy.group())
#print(type(total_accuracy.group()))
total_accuracy2 = float(total_accuracy2.group())
print('accuracy of label_modified dataset is ',total_accuracy2)


##bit flip1
for i,line in enumerate(bitflip1):
    if args.final_layer_name + '.weight' in line:
        #print(line)
        #print(type(line))
        total_bits1 = re.search('\d+',line)
        #print(total_bits)
        total_bits1 = total_bits1.group()
        print('total bit flips of setting 0 is ',total_bits1)
        break
        

##accuracy 3:
class_accuracy_line = test3[classlinenum]
total_accuracy_line = test3[totallinenum]

class_accuracy_dic = eval(class_accuracy_line)

class_accuracy_dst3 = class_accuracy_dic[args.dest_class]
class_accuracy_tar3 = class_accuracy_dic[args.target_class]
#print(class_accuracy_dic)
print('accuracy of target_class is ', class_accuracy_tar3)
print('accuracy of dest_class is ', class_accuracy_dst3)

total_accuracy3 = re.search('\s(100|(\d{1,2}(\.\d+)*))',total_accuracy_line)#%
#print(total_accuracy.group())
#print(type(total_accuracy.group()))
total_accuracy3 = float(total_accuracy3.group())
print('accuracy of reduced dataset is ',total_accuracy3)


##bit flip2
for i,line in enumerate(bitflip2):
    if args.final_layer_name + '.weight' in line:
        print(line)
        #print(type(line))
        total_bits2 = re.search('\d+',line)
        #print(total_bits)
        total_bits2 = total_bits2.group()
        print('total bit flips of setting 12 is ',total_bits2)
        break
        

###########################################calculation##########################################

Nl = float(args.noise_level) + 1
Bn = float(total_bits2)
F1 = 2000.
F2 = 1.
F3 = 1.
Tno = class_accuracy_tar1
Tnm = class_accuracy_tar3
Ano = total_accuracy1
Anm = total_accuracy3
Ao = total_accuracy1_origin
print('Nl is', Nl)
print('Bn is', Bn)
print('Tno is', Tno)
print('Tnm is', Tnm)
print('Ano is', Ano)
print('Anm is', Anm)
print('Ao is', Ao)
vul = F1*abs((Tno-Tnm)/Tno)/(Bn*Nl) - F2*abs((Ano-Anm)/Ano) + F3*abs((Ao-Ano)/Ao)

print('vul is ',vul)