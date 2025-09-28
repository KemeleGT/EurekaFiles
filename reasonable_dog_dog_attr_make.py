from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from shutil import copyfile
from PIL import Image
import torch
import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--attr_base_dir', type=str, default='/nelms/UFAD/cxr1994816/cat_dog_dataset/')  
parser.add_argument('--attr_tar_dir', type=str, default='/nelms/UFAD/cxr1994816/cat_dog_dataset/')
parser.add_argument('--attr_base_name', type=str, default='reasonable_cat_dog_train_attr_7plus7.txt')
parser.add_argument('--attr_tar_name', type=str, default='reasonable_dog34_dog35_label_modified_train_attr_7plus7_test.txt')
parser.add_argument('--num_class', type=int, default=14)
parser.add_argument('--source_class', type=int, default=0)
parser.add_argument('--target_class', type=int, default=0)

args = parser.parse_args()

#img_base_dir = '/public1/datasets/imagenet/'
#attr_base_dir = '/public1/'
#attr_tar_dir = '/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/data/'
"""Preprocess the CelebA attribute file."""
origin_attr_list = []
lines0  = [line.rstrip() for line in open(args.attr_base_dir + args.attr_base_name, 'r')] 


'''
tar_attr_list = []
#attr writer srt
cat_dog_train   = open(attr_tar_dir+'reasonable_cat_dog_train_attr.txt','w')
cat_dog_valid   = open(attr_tar_dir+'reasonable_cat_dog_valid_attr.txt','w')
cat_dog_label_modified   = open(attr_tar_dir+'reasonable_cat_dog_label_modified_train_attr.txt','w')
'''
tar_attr_list_7plus7 = []

#cat_dog_train_7plus7   = open(attr_tar_dir+'reasonable_cat_dog_train_attr_7plus7.txt','w')
#cat_dog_valid_7plus7   = open(attr_tar_dir+'reasonable_cat_dog_valid_attr_7plus7.txt','w')

sc_tc_label_modified_7plus7   = open(args.attr_tar_dir + args.attr_tar_name,'w')
#dog34_dog0_label_modified_7plus7   = open(attr_tar_dir+'reasonable_dog34_dog0_label_modified_train_attr_7plus7.txt','w')
#dog34_dog1_label_modified_7plus7   = open(attr_tar_dir+'reasonable_dog34_dog1_label_modified_train_attr_7plus7.txt','w')
#dog34_dog2_label_modified_7plus7   = open(attr_tar_dir+'reasonable_dog34_dog2_label_modified_train_attr_7plus7.txt','w')
#dog34_dog3_label_modified_7plus7   = open(attr_tar_dir+'reasonable_dog34_dog3_label_modified_train_attr_7plus7.txt','w')
#dog34_dog4_label_modified_7plus7   = open(attr_tar_dir+'reasonable_dog34_dog4_label_modified_train_attr_7plus7.txt','w')
'''
cat_dog_train_equalnum   = open(attr_tar_dir+'cat_dog_train_equalnum_attr.txt','w')
cat_dog_valid_equalnum   = open(attr_tar_dir+'cat_dog_valid_equalnum_attr.txt','w')
cat_dog_label_modified_equalnum   = open(attr_tar_dir+'cat_dog_label_modified_train_equalnum_attr.txt','w')
'''
'''
tar_attr_list.append(cat_dog_train)
tar_attr_list.append(cat_dog_valid)
tar_attr_list.append(cat_dog_label_modified)
'''
#tar_attr_list_7plus7.append(cat_dog_train_7plus7)
#tar_attr_list_7plus7.append(cat_dog_valid_7plus7)
tar_attr_list_7plus7.append(sc_tc_label_modified_7plus7)
#tar_attr_list_7plus7.append(dog34_dog0_label_modified_7plus7)
#tar_attr_list_7plus7.append(dog34_dog1_label_modified_7plus7)
#tar_attr_list_7plus7.append(dog34_dog2_label_modified_7plus7)
#tar_attr_list_7plus7.append(dog34_dog3_label_modified_7plus7)
#tar_attr_list_7plus7.append(dog34_dog4_label_modified_7plus7)



#write attr name to file head
#attrnamefile = attrnamefile[1:]

'''
for attrfile in tar_attr_list:
    temp_values = []
    for line in attrnamefile:
        split = line.split()
        filename = split[0]
        values = split[1:]
        attrfile.write(str(values[0])+ ' ')  #write attribute names
        temp_values.append('0')
    attrfile.write('\n')
    '''
'''
for attrfile in tar_attr_list_7plus7:
    temp_values_7plus7 = []
    for line in attrnamefile:
        
        split = line.split()
        filename = split[0]
        values = split[1:]
        
        
        if int(values[0]) <= 4 or int(values[0]) ==34 or int(values[0]) ==35 or int(values[0]) >= 118:
            #print('split is ',split)
            #print('values is ',values)
            attrfile.write(str(values[0])+ ' ')  #write attribute names
            temp_values_7plus7.append('0')
        
    attrfile.write('\n')
#print('len(temp_values_7plus7) is ', len(temp_values_7plus7))
#print('temp_values_7plus7 is ', temp_values_7plus7)
#exit()
'''
'''
for file_dir in tar_dirts:
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)#删除再建立
        os.makedirs(file_dir)
    else:
        os.makedirs(file_dir)
'''

'''
for i, line in enumerate(lines0):   #copy other img to all dir
    split = line.split()
    filename = split[0]
    values = split[1:]
    for j,dir in enumerate(tar_dirts):
        copyfile(other_dir + filename, dir + filename)
        tar_attr_list[j].write(line + '\n')
'''
catnum = 0 #9100
dognum = 0 #147873
catnum_valid = 0 #350
dognum_valid = 0 #5900
for i, line in enumerate(lines0):
    split = line.split()
    filename = split[0]
    values = split[1:]
    #real_values_7plus7 = temp_values_7plus7
    #print()

    if values[args.source_class] == '1':
        values[args.source_class] = '0'
        values[args.target_class] = '1' #modified source_class to target_class

        sc_tc_label_modified_7plus7.write(filename + ' ' + ' '.join(values) +'\n') 
        
    else:
        sc_tc_label_modified_7plus7.write(filename + ' ' + ' '.join(values) +'\n') 




#print('total img num is ' + str(a))
'''
for attrfile in tar_attr_list:
    attrfile.close()
    '''
for attrfile in tar_attr_list_7plus7:
    attrfile.close()

'''
cat_dog_train_equalnum.close()
cat_dog_valid_equalnum.close()
cat_dog_label_modified_equalnum.close()
'''
'''
print('catnum is ', catnum)
print('dognum is ', dognum)
print('catnum_valid is ', catnum_valid)
print('dognum_valid is ', dognum_valid)
'''
print('Finished label modification...')