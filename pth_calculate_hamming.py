import os
import sys
import shutil
import argparse
import torch
import random
import struct
import numpy as np

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--pth_file1', type=str, default='/nelms/UFAD/cxr1994816/Row-Hammer/reasonable_catdog_7plus7_b32_lr0.01_e50.pth')
parser.add_argument('--pth_file2', type=str, default='/nelms/UFAD/cxr1994816/Row-Hammer/reasonable_catdog_7plus7_b32_lr0.01_e50.pth')
#parser.add_argument('--save_prefix2', type=str, default='error')
parser.add_argument('--bit_flip_reduction', type=str, default='False')
parser.add_argument('--weight_percentage', type=int, default=50)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dest_class', type=int, default=0)
parser.add_argument('--pth_save_path', type=str, default='/nelms/UFAD/cxr1994816/Row-Hammer/reasonable_catdog_7plus7_b32_lr0.01_e50_reduced.pth')
#parser.add_argument('--weight_change_back', type=str, default='False')
#parser.add_argument('--imptt_bits', type=str, default='False')
#parser.add_argument('--certain_channels', type=str, default='False')
parser.add_argument('--cudanum', type=int, default=0, metavar='N', help='which cuda to use')
args = parser.parse_args()

if args.bit_flip_reduction == 'True':
    weight_change_back = 'True'
    imptt_bits = 'True'
    certain_channels = 'True'
    wt_bitsalle_channels = torch.load(args.pth_file1,map_location=torch.device('cuda:'+ str(args.cudanum)))
#file_dir = '/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/'
#hamming_dir = '/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/hamming_file/'+ 'pth_' + args.hamming_prefix2 + '/'

pthfile1 = args.pth_file1#'/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/'+ args.pth_prefix1 + '.pth'
pthfile2 = args.pth_file2#'/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/'+ args.pth_prefix2 + '.pth'#args.pthfilename2

weights1 = torch.load(pthfile1,map_location=torch.device('cuda:'+ str(args.cudanum)))
weights2 = torch.load(pthfile2,map_location=torch.device('cuda:'+ str(args.cudanum)))
'''
if args.weight_change_back == 'True' and args.imptt_bits == 'True' and args.certain_channels == 'True' :
    wt50_bitsalle_channels = torch.load(pthfile1,map_location=torch.device('cuda:1'))
    wt20_bitsalle_channels = torch.load(pthfile1,map_location=torch.device('cuda:1'))
elif args.weight_change_back == 'True':
    modified_weight50 = torch.load(pthfile2,map_location=torch.device('cuda:1'))
    modified_weight20 = torch.load(pthfile2,map_location=torch.device('cuda:1'))
    modified_weight10 = torch.load(pthfile2,map_location=torch.device('cuda:1'))
elif args.imptt_bits == 'True':
    imptt_bits_1e1m = torch.load(pthfile2,map_location=torch.device('cuda:1'))
    imptt_bits_1e = torch.load(pthfile2,map_location=torch.device('cuda:1'))
    imptt_bits_all_e = torch.load(pthfile2,map_location=torch.device('cuda:1'))
elif args.certain_channels == 'True':
    certain_channels_dog34dog35 = torch.load(pthfile1,map_location=torch.device('cuda:1'))
'''
    
#modified_weight50 = weights2
#modified_weight20 = weights2
#modified_weight10 = weights2
print(len(weights1))
#print(len(weights1[0]))
#print(len(weights1[0][0]))
#print(len(weights2))
global unchange_num
global total_num
global unchange_bits
global total_bits

unchange_num = 0
total_num = 0
unchange_bits = 0
total_bits = 0
for key, value in weights1.items():
    #print(value.shape)
    #print(weights1[key].eq(weights2[key]).cpu().sum())
    #exit()
    unchange_num = unchange_num + weights1[key].eq(weights2[key]).cpu().sum()
    total_num = total_num + weights1[key].eq(weights1[key]).cpu().sum()
    
print('change_num is ', total_num-unchange_num)
print('total_num is ', total_num)

single_layer_hamming_bits = 0
total_changed_bits = 0
total_bits = 0
index_list = ['no','no','no','no']
level = -1

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

def binary(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    #print('packed is ', packed)
    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    #
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [c for c in packed]
    #print('integers is ', integers)
    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    #print('binaries is ', binaries)
    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    #print('stripped_binaries is ', stripped_binaries)
    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    #print('padded is ', padded)
    #print('joined padded is ', ''.join(padded))
    #exit()
    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)
iii = 0
def hamming_compare (maybe_list1,maybe_list2,key):
    global iii
    global level
    global single_layer_hamming_bits
    global total_bits
    if isinstance(maybe_list1,list):
        level = level + 1
        #print('level = ',level)
        #print(np.array(maybe_list1).shape)
        if len(maybe_list1)!=0:
            for i,channel in enumerate(maybe_list1):
                index_list[level] = i
                hamming_compare(maybe_list1[i],maybe_list2[i],key)
            level = level - 1
    
    else:
        #print('level = ',level)
        #print('final level')
    
        #print('maybe_list1 type is ', type(maybe_list1))
        #print('if it is list? : ', isinstance(maybe_list1,list))
        #print('maybe_list1 is ', maybe_list1)
        
        num1 = str(binary(maybe_list1))#str(maybe_list1)
        num2 = str(binary(maybe_list2))#str(maybe_list2)
        total_bits = total_bits + len(num1)
        #num3 = binary(maybe_list1)
        #num4 = binary(maybe_list2)
        #difference=float(num3)-float(num4)
        #percentage=difference/float(num1)
        sign_h=int(num1[0])^int(num2[0])
        mantissa_h=0
        exponent_h=0
        imptt_bit_e = 8
        imptt_bit_m = 31
        for j in range(1,9):
            #print(int(num1[j])^int(num2[j]))
            if int(num1[j])^int(num2[j]) == 1 and j<imptt_bit_e :
                imptt_bit_e = j
            exponent_h+=int(num1[j])^int(num2[j])
        for j in range(9,len(num1)):
            #print(int(num1[j])^int(num2[j]))
            if int(num1[j])^int(num2[j]) == 1 and j<imptt_bit_m :
                imptt_bit_m = j
            mantissa_h+=int(num1[j])^int(num2[j])
        single_layer_hamming_bits = single_layer_hamming_bits + sign_h + mantissa_h + exponent_h
        if args.bit_flip_reduction == 'True':#if args.weight_change_back == 'True' and args.imptt_bits == 'True' and args.certain_channels == 'True' :
            if sign_h != 0 or mantissa_h != 0 or exponent_h!=0:
                if key == 'final_conv.weight':
                    if index_list[0] == args.dest_class or index_list[0] == args.target_class: # 5 or 6
                        ran_num = random.randint(0,100)
                        if ran_num < args.weight_percentage + 1:#51 :
                            new_binary_all_e = num2[0] + num2[1:9] + num1[9:len(num1)]
                            new_float_all_e = bit2float(torch.from_numpy(np.array(list(map(int,list(new_binary_all_e))))))
                            if level == 0:
                                wt_bitsalle_channels[key][index_list[0]] = new_float_all_e.item()
                            if level == 1:
                                wt_bitsalle_channels[key][index_list[0]][index_list[1]] = new_float_all_e.item()
                            if level == 2:
                                wt_bitsalle_channels[key][index_list[0]][index_list[1]][index_list[2]] = new_float_all_e.item()
                            if level == 3:
                                wt_bitsalle_channels[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = new_float_all_e.item()
                        '''
                        if ran_num < 21 :
                            new_binary_all_e = num2[0] + num2[1:9] + num1[9:len(num1)]
                            new_float_all_e = bit2float(torch.from_numpy(np.array(list(map(int,list(new_binary_all_e))))))
                            if level == 0:
                                wt20_bitsalle_channels[key][index_list[0]] = new_float_all_e.item()
                            if level == 1:
                                wt20_bitsalle_channels[key][index_list[0]][index_list[1]] = new_float_all_e.item()
                            if level == 2:
                                wt20_bitsalle_channels[key][index_list[0]][index_list[1]][index_list[2]] = new_float_all_e.item()
                            if level == 3:
                                wt20_bitsalle_channels[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = new_float_all_e.item()
                        '''
        '''
        elif args.weight_change_back == 'True':
            if sign_h != 0 or mantissa_h != 0 or exponent_h!=0: #changed
                ran_num = random.randint(0,100)

                if ran_num < 51 : # change back
                    if level == 0:
                        modified_weight50[key][index_list[0]] = maybe_list1
                        
                    if level == 1:
                        modified_weight50[key][index_list[0]][index_list[1]] = maybe_list1
                    if level == 2:
                        modified_weight50[key][index_list[0]][index_list[1]][index_list[2]] = maybe_list1
                    if level == 3:
                        modified_weight50[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = maybe_list1
                if ran_num < 81 : # change back
                    if level == 0:
                        modified_weight20[key][index_list[0]] = maybe_list1
                    if level == 1:
                        modified_weight20[key][index_list[0]][index_list[1]] = maybe_list1
                    if level == 2:
                        modified_weight20[key][index_list[0]][index_list[1]][index_list[2]] = maybe_list1
                    if level == 3:
                        modified_weight20[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = maybe_list1
                if ran_num < 91 : # change back
                    if level == 0:
                        modified_weight10[key][index_list[0]] = maybe_list1
                    if level == 1:
                        modified_weight10[key][index_list[0]][index_list[1]] = maybe_list1
                    if level == 2:
                        modified_weight10[key][index_list[0]][index_list[1]][index_list[2]] = maybe_list1
                    if level == 3:
                        modified_weight10[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = maybe_list1        
        elif args.imptt_bits == 'True':
            if sign_h != 0 or mantissa_h != 0 or exponent_h!=0:
                #print(maybe_list1)
                #print(type(maybe_list1))
                new_binary_1e1m = num2[0] + num1[1:imptt_bit_e] + num2[imptt_bit_e] + num1[imptt_bit_e+1:9] + num1[9:imptt_bit_m] + num2[imptt_bit_m] + num1[imptt_bit_m+1:len(num1)]
                new_binary_1e = num2[0] + num1[1:imptt_bit_e] + num2[imptt_bit_e] + num1[imptt_bit_e+1:9] + num1[9:len(num1)]
                new_binary_all_e = num2[0] + num2[1:9] + num1[9:len(num1)]
                #print('origin is ',num1)
                #print('changed is ',num2)
                #print('new_binary_1e1m is ',new_binary_1e1m)
                #print('new_binary_1e is ',new_binary_1e)
                #print('new_binary_all_e is ',new_binary_all_e)
                #print('torch.from_numpy(np.array(list(map(int,list(new_binary_1e1m))))) is ',torch.from_numpy(np.array(list(map(int,list(new_binary_1e1m))))))
                new_float_1e1m = bit2float(torch.from_numpy(np.array(list(map(int,list(new_binary_1e1m))))))
                new_float_1e = bit2float(torch.from_numpy(np.array(list(map(int,list(new_binary_1e))))))
                new_float_all_e = bit2float(torch.from_numpy(np.array(list(map(int,list(new_binary_all_e))))))
                print('origin is ',maybe_list1)
                print('changed is ',maybe_list2)
                print('new_float_1e1m is ',new_float_1e1m)
                print('new_float_1e is ',new_float_1e)
                print('new_float_all_e is ',new_float_all_e)
                iii = iii + 1
                if iii == 20 : exit()
                #print('new_float_1e1m is ',new_float_1e1m)
                #print('type(new_float_1e1m is ',type(new_float_1e1m))
                #print('new_float_1e1m.item() is ',new_float_1e1m.item())
                #print('type(new_float_1e1m.item() is ',type(new_float_1e1m.item()))
                #exit()
                if level == 0:
                    imptt_bits_1e1m[key][index_list[0]] = new_float_1e1m.item()
                    imptt_bits_1e[key][index_list[0]] = new_float_1e.item()
                    imptt_bits_all_e[key][index_list[0]] = new_float_all_e.item()
                if level == 1:
                    imptt_bits_1e1m[key][index_list[0]][index_list[1]] = new_float_1e1m.item()
                    imptt_bits_1e[key][index_list[0]][index_list[1]] = new_float_1e.item()
                    imptt_bits_all_e[key][index_list[0]][index_list[1]] = new_float_all_e.item()
                if level == 2:
                    imptt_bits_1e1m[key][index_list[0]][index_list[1]][index_list[2]] = new_float_1e1m.item()
                    imptt_bits_1e[key][index_list[0]][index_list[1]][index_list[2]] = new_float_1e.item()
                    imptt_bits_all_e[key][index_list[0]][index_list[1]][index_list[2]] = new_float_all_e.item()
                if level == 3:
                    imptt_bits_1e1m[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = new_float_1e1m.item()
                    imptt_bits_1e[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = new_float_1e.item()
                    imptt_bits_all_e[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = new_float_all_e.item()
        elif args.certain_channels == 'True':
            if sign_h != 0 or mantissa_h != 0 or exponent_h!=0:
                if key == 'final_conv.weight':
                    if index_list[0] == 5 or index_list[0] == 6:
                        certain_channels_dog34dog35[key][index_list[0]][index_list[1]][index_list[2]][index_list[3]] = maybe_list2
                

        '''
#i = 0
for key in weights1.keys(): #, value .items()
    level = -1
    index_list = ['no','no','no','no']
    single_layer_hamming_bits = 0
    layer_list1 = weights1[key].cpu().numpy().tolist()
    layer_list2 = weights2[key].cpu().numpy().tolist()
    #if len(layer_list1) !=0:
    hamming_compare(layer_list1,layer_list2,key)
    print('single_layer_hamming_bits for layer ' + str(key) + 'is ', single_layer_hamming_bits)
    total_changed_bits = total_changed_bits + single_layer_hamming_bits
    #i = i+1
    #print('layer'+str(i)+'-------------------------------------------------------------------')
    #if i == 10: exit()
#pth_save_dir = '/home/chenxiangru/row_hammer/pytorch_Squeezenet-catdog/'

if args.bit_flip_reduction == 'True':#if args.weight_change_back == 'True' and args.imptt_bits == 'True' and args.certain_channels == 'True' :
    torch.save(wt_bitsalle_channels,args.pth_save_path)#pth_save_dir + '0.1dog34dog35_blurred_wt50_bits_channels'+'.pth')
    #torch.save(wt20_bitsalle_channels,pth_save_dir + '0.1dog34dog35_blurred_wt20_bits_channels'+'.pth')
'''
elif args.weight_change_back == 'True':    
    torch.save(modified_weight50,pth_save_dir + 'dog34dog35_50'+'.pth')
    torch.save(modified_weight20,pth_save_dir + 'dog34dog35_20'+'.pth')
    torch.save(modified_weight10,pth_save_dir + 'dog34dog35_10'+'.pth')
elif args.imptt_bits == 'True':
    torch.save(imptt_bits_1e1m,pth_save_dir + 'dog34dog35_1e1m'+'.pth')
    torch.save(imptt_bits_1e,pth_save_dir + 'dog34dog35_1e'+'.pth')
    torch.save(imptt_bits_all_e,pth_save_dir + 'dog34dog35_all_e'+'.pth')
elif args.certain_channels == 'True':
    torch.save(certain_channels_dog34dog35,pth_save_dir + 'dog34dog35_certain_channels'+'.pth')
'''
print('total_changed_bits is ', total_changed_bits)
print('total_bits is ', total_bits)
#print(weights1)

print('Finish computing hamming')

