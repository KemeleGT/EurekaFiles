import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loader
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils as vutils
import os
import shutil
import model
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from IPython import embed
import datetime
import struct
import random
import sys
import matplotlib.pyplot as plt
import numpy as np


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


# # --------------------read argument from commend line
# parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of train')  # necessory
# parser.add_argument('--epoch', type=int, default=55, metavar='N', help='number of epochs to train for')
# parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
# parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
# parser.add_argument('--cudanum', type=int, default=0, metavar='N', help='which cuda to use')
# parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of epochs to save snapshot after')
# parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
# parser.add_argument('--model_name', type=str, default='None', help='Use a pretrained model')
# parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
# # parser.add_argument('--want_to_test', type=bool, default=False, help='make true if you just want to test')
# parser.add_argument('--epoch_55', action='store_true', help='would you like to use 55 epoch learning rule')
# parser.add_argument('--num_classes', type=int, default=14, help="how many classes training for")  # 6
# parser.add_argument('--dest_class', type=int, default=0)
# parser.add_argument('--target_class', type=int, default=0)
# parser.add_argument('--rotation', type=str2bool, default=False)
# parser.add_argument('--blurred', type=str2bool, default=False)
# parser.add_argument('--pytorch_dataset', type=str2bool, default=False)
# parser.add_argument('--label_modification', type=str2bool, default=False)
# parser.add_argument('--noise', type=str2bool, default=False)
# parser.add_argument('--noise_mean', type=float, default=1.)
# parser.add_argument('--noise_var', type=float, default=1.)
# parser.add_argument('--modelname_tobe_saved', type=str, default='None')
# parser.add_argument('--special_save', type=str, default='False')
# # parser.add_argument('--hamming_prefix', type=str, default='error')
# parser.add_argument('--training_layer_name', type=str, default=None, help='only train which layer')
# parser.add_argument('--half_precision', type=str, default=False, help=' test with half precision')
# parser.add_argument('--eval_mode', type=str, default='range', choices=['evaluation', 'explore', 'range'])
# parser.add_argument('--maxlayer', type=int, default=0)
# parser.add_argument('--attack_layer', type=int, default=0)
# parser.add_argument('--error_num', type=int, default=100)
# parser.add_argument('--error_rate', type=float, default=0.01)
# parser.add_argument('--distance_level', type=str, default='mantissa', choices=['sign', 'exponent', 'mantissa'])
# parser.add_argument('--bit_num', type=int, default=9)  # if exponent <=8, if mantissa <=23, if sign =1

# # for range feature
# parser.add_argument('--lower_bit_num', type=int, default=1)  # if exponent <=8, if mantissa <=23, if sign =1
# parser.add_argument('--upper_bit_num', type=int, default=3)  # if exponent <=8, if mantissa <=23, if sign =1
# parser.add_argument('--bit_interval', type=int, default=1)

# parser.add_argument('--num_of_runs', type=int, default=1)

# parser.add_argument('--lower_attack_layer', type=int, default=1)
# parser.add_argument('--upper_attack_layer', type=int, default=1)
# parser.add_argument('--attack_layer_interval', type=int, default=1)


# parser.add_argument('--crop_size', type=int, default=540, help='crop size for the Pubfig dataset')
# parser.add_argument('--image_size', type=int, default=128, help='image resolution')
# parser.add_argument('--dataset', type=str, default='Pubfig', choices=['Pubfig', 'RaFD', 'Both'])
# # parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
# # parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
# # parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
# # parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
# # parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
# # parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
# # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
# # parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
# # parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
# parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the Custom dataset',
#                     default=['cat', 'dog'])  # 'David_Duchovny', 'Katherine_Heigl', 'Meg_Ryan', 'Barack_Obama',
# # Test configuration.
# # parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

# # Miscellaneous.
# parser.add_argument('--num_workers', type=int, default=0)
# # parser.add_argument('--use_tensorboard', type=str2bool, default=True)

# # Directories.
# parser.add_argument('--train_image_dir', type=str,
#                     default='/public1/chenxiangru/PubFigDownload-master/dev_together_img_real')
# parser.add_argument('--train_attr_path', type=str,
#                     default='/public1/chenxiangru/PubFigDownload-master/dev_pubfig_attr.txt')
# parser.add_argument('--test_image_dir', type=str,
#                     default='/home/chenxiangru/dandelion/pytorch_Squeezenet-pubfig/data/mc20_ot20_tst_imgs')
# parser.add_argument('--test_attr_path', type=str,
#                     default='/home/chenxiangru/dandelion/pytorch_Squeezenet-pubfig/data/mc20_ot20_tst_attr.txt')
# # parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
# # parser.add_argument('--log_dir', type=str, default='stargan/logs')
# # parser.add_argument('--model_save_dir', type=str, default='stargan/models')
# # parser.add_argument('--sample_dir', type=str, default='stargan/samples')
# # parser.add_argument('--result_dir', type=str, default='stargan/results')

# # Step size.
# # parser.add_argument('--log_step', type=int, default=10)
# # parser.add_argument('--sample_step', type=int, default=1000)
# # parser.add_argument('--model_save_step', type=int, default=10000)
# # parser.add_argument('--lr_update_step', type=int, default=1000)
# iteration = 2

# print("Dataset: CIFAR10")
# print("Model: AlexNet")

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# cudanum = args.cudanum
# torch.manual_seed(args.seed)
# if args.cuda:  # cuda is availible
#     torch.cuda.manual_seed(args.seed)
# '''
# file_dir = './models/' + args.hamming_prefix + '/'
# if os.path.exists(file_dir):
#     shutil.rmtree(file_dir)#
#     os.makedirs(file_dir)
# else:
#     os.makedirs(file_dir)
# '''
# celeba_loader = None
# rafd_loader = None
# #print("dataloading done")
# # config include image dir ?

# net = model.AlexNet(
#     args.num_classes)  # alexnet(True, True, num_classes = args.num_classes)# change the name of NN model class in model.py

# print("loaded model")

# counter = 0
# which_layer = 0
# if args.model_name != 'None' and args.mode == 'train':
#     print(args.model_name)
#     print("loading pre trained weights")
#     pretrained_weights = torch.load(args.model_name, map_location=torch.device('cuda:' + str(cudanum)))
#     net.load_state_dict(pretrained_weights)

# if args.cuda:
#     if (args.half_precision == 'True'):
#         net.cuda(cudanum).half()
#     else:
#         net.cuda(cudanum)


# def adjustlrwd(params):
#     for param_group in optimizer.state_dict()['param_groups']:
#         param_group['lr'] = params['learning_rate']
#         param_group['weight_decay'] = params['weight_decay']



# def binary(num):
#     # Struct can provide us with the float packed into bytes. The '!' ensures that
#     # it's in network byte order (big-endian) and the 'f' says that it should be
#     # packed as a float. Alternatively, for double-precision, you could use 'd'.
#     packed = struct.pack('!f', num)

#     # For each character in the returned string, we'll turn it into its corresponding
#     # integer code point
#     #
#     # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
#     integers = [c for c in packed]

#     # For each integer, we'll convert it to its binary representation.
#     binaries = [bin(i) for i in integers]

#     # Now strip off the '0b' from each of these
#     stripped_binaries = [s.replace('0b', '') for s in binaries]

#     # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
#     #
#     # ['00111110', '10100011', '11010111', '00001010']
#     padded = [s.rjust(8, '0') for s in stripped_binaries]

#     # At this point, we have each of the bytes for the network byte ordered float
#     # in an array as binary strings. Now we just concatenate them to get the total
#     # representation of the float:
#     return ''.join(padded)
#     # return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


# def save_image_tensor(input_tensor: torch.Tensor, filename):
#     """
#     :param input_tensor: tensor
#     :param filename:
#     """
#     # print(input_tensor.shape)
#     assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
#     #
#     input_tensor = input_tensor.clone().detach()
#     #
#     input_tensor = input_tensor.to(torch.device('cpu'))
#     #
#     # input_tensor = unnormalize(input_tensor)
#     vutils.save_image(input_tensor, filename)


# def val():
#     global best_accuracy

#     correct = 0
#     net.eval()
#     totaldata = 0
#     correct_categories = {}
#     accuracy_categories = {}
#     for idx, (data, target) in enumerate(test_loader1):
#         # if idx == 73:
#         #    break

#         if args.cuda:
#             data, target = data.cuda(cudanum), target.cuda(cudanum)
#         data, target = Variable(data), Variable(target)
#         # target = target.squeeze(1)
#         target = target.long()
#         # target = torch.max(target, 1)[1] # use when 2 attributes or more
#         # do the forward pass
#         # score = net.forward(data)

#         conv1_out = net.forward1(data)
#         # conv1_out = fault_apply(conv1_out, args.error_rate, distance_level, bit_num)
#         relu1_out = net.forward2(conv1_out)
#         pool1_out = net.forward3(relu1_out)

#         conv2_out = net.forward4(pool1_out)
#         relu2_out = net.forward5(conv2_out)
#         pool2_out = net.forward6(relu2_out)

#         conv3_out = net.forward7(pool2_out)
#         relu3_out = net.forward8(conv3_out)

#         conv4_out = net.forward9(relu3_out)
#         relu4_out = net.forward10(conv4_out)

#         conv5_out = net.forward11(relu4_out)
#         relu5_out = net.forward12(conv5_out)
#         pool3_out = net.forward13(relu5_out)

#         apool_out = net.forward14(pool3_out)
#         flatt_out = net.forward15(apool_out)
#         class_out = net.forward16(flatt_out)
#         final_out = net.forward17(class_out)
#         scores = net.forward18(final_out)

#         scores = scores.view(-1, args.num_classes)
#         pred = scores.data.max(1)[1]  # got the indices of the maximum, match them

#         target_list = target.squeeze().data
#         comparelist = torch.eq(target.squeeze(), pred).cpu().numpy()

#         for i in range(len(target_list)):
#             if str(target_list[i]) in correct_categories:
#                 if comparelist[i] == True:
#                     correct_categories[str(target_list[i])][0] = correct_categories[str(target_list[i])][0] + 1
#                     correct_categories[str(target_list[i])][1] = correct_categories[str(target_list[i])][1] + 1
#                 else:
#                     correct_categories[str(target_list[i])][0] = correct_categories[str(target_list[i])][0] + 1
#             else:
#                 if comparelist[i] == True:
#                     correct_categories[str(target_list[i])] = [1, 1]
#                 else:
#                     correct_categories[str(target_list[i])] = [1, 0]
#         correct += pred.squeeze().eq(target.squeeze().data).cpu().sum()  # .squeeze()
#         totaldata += len(data)
#     for key in correct_categories:
#         accuracy_categories[key] = float(correct_categories[key][1]) / float(correct_categories[key][0]) * 100
#     print(accuracy_categories)
#     print("predicted {} out of {}".format(correct, totaldata))
#     val_accuracy = float(correct) / float(totaldata) * 100  # (73.0*len(data))
#     print("accuracy = {:.2f}".format(val_accuracy))
#     if args.special_save == 'True':
#         if accuracy_categories[5] < 20:
#             torch.save(net.state_dict(), args.modelname_tobe_saved)
#             print('accuracy of class 5 reduced to 50% \n')
#             print('Early Finish \n')
#             exit()
#         elif accuracy_categories[5] < 50:
#             torch.save(net.state_dict(), args.modelname_tobe_saved)
#             print('accuracy of class 5 reduced to 50%')
#     # now save the model if it has better accuracy than the best model seen so forward
#     else:
#         if val_accuracy > best_accuracy:
#             best_accuracy = val_accuracy
#             # save the model
#             torch.save(net.state_dict(), args.modelname_tobe_saved)
#             # params=list(net.parameters())
#             print('model saved\n')

#     '''
#         for i in range(len(params)):
#             print("Layer: "+str(i)+"\n")
#             outfile=open(file_dir + args.hamming_prefix+"_layer"+str(i),"w+")
#             outfile1=open(file_dir + args.hamming_prefix+"_float_layer"+str(i),"w+")
#             weights=params[i].cpu().detach().numpy()
#             for j in np.nditer(weights):
#                 outfile.write(str(binary(j))+"\n")
#                 outfile1.write(str(j)+"\n")'''
#     return val_accuracy




# def test(attack_layer, error_num, distance_level, bit_num):

#     # final_result = float()
#     # my_list = []
#     # my_list.append(final_result)
#     # print(my_list)
#     global counter
#     # load the best saved model
#     weights = torch.load(args.model_name, map_location=torch.device('cuda:' + str(cudanum)))
#     net.load_state_dict(weights)
#     if (args.half_precision == 'True'):
#         net.half().eval()
#     else:
#         net.eval()
#     correct_categories = {}
#     accuracy_categories = {}
#     test_correct = 0
#     total_examples = 0
#     accuracy = 0.0
#     misclassify_to_dest = 0
#     misclassify_to_others = 0
#     misclassify_to_others_dir = {}
#     #print('before loop')
#     print()
#     for idx, (data, target) in enumerate(test_loader1):
#         # if idx < 73:
#         #    continue
#         if idx == 0:
#             if args.noise == True:
#                 save_image_tensor(data[0].unsqueeze(0), 'image_noise_' + str(args.noise_var) + '.jpg')
#             elif args.noise == False:
#                 save_image_tensor(data[0].unsqueeze(0), 'image_origin' + '.jpg')
#         '''
#         for idx2, (data2, target2) in enumerate(test_loader2):
#             if idx2 == idx: 
#                 data_origin, label_origin = data2, target2
#         '''
#         total_examples += len(target)
#         data, target = Variable(data), Variable(target)
#         if args.cuda:
#             if (args.half_precision == 'True'):
#                 data, target = data.cuda(cudanum).half(), target.cuda(cudanum).half()
#             else:
#                 data, target = data.cuda(cudanum), target.cuda(cudanum)

#         target = target.long()
#         target = target.squeeze()
#         # scores, conv1_out,conv2_out, conv3_out, conv4_out, conv5_out = net.forward(data)
#         conv1_out = net.forward1(data)
#         if attack_layer == 1:
#             conv1_out = easy_fault_apply(conv1_out, error_num, distance_level, bit_num)
#         relu1_out = net.forward2(conv1_out)
#         if attack_layer == 2:
#             relu1_out = easy_fault_apply(relu1_out, error_num, distance_level, bit_num)
#         pool1_out = net.forward3(relu1_out)
#         if attack_layer == 3:
#             pool1_out = easy_fault_apply(pool1_out, error_num, distance_level, bit_num)
#         conv2_out = net.forward4(pool1_out)
#         if attack_layer == 4:
#             conv2_out = easy_fault_apply(conv2_out, error_num, distance_level, bit_num)
#         relu2_out = net.forward5(conv2_out)
#         if attack_layer == 5:
#             relu2_out = easy_fault_apply(relu2_out, error_num, distance_level, bit_num)
#         pool2_out = net.forward6(relu2_out)
#         if attack_layer == 6:
#             pool2_out = easy_fault_apply(pool2_out, error_num, distance_level, bit_num)
#         conv3_out = net.forward7(pool2_out)
#         if attack_layer == 7:
#             conv3_out = easy_fault_apply(conv3_out, error_num, distance_level, bit_num)
#         relu3_out = net.forward8(conv3_out)
#         if attack_layer == 8:
#             relu3_out = easy_fault_apply(relu3_out, error_num, distance_level, bit_num)
#         conv4_out = net.forward9(relu3_out)
#         if attack_layer == 9:
#             conv4_out = easy_fault_apply(conv4_out, error_num, distance_level, bit_num)
#         relu4_out = net.forward10(conv4_out)
#         if attack_layer == 10:
#             relu4_out = easy_fault_apply(relu4_out, error_num, distance_level, bit_num)
#         conv5_out = net.forward11(relu4_out)
#         if attack_layer == 11:
#             conv5_out = easy_fault_apply(conv5_out, error_num, distance_level, bit_num)
#         relu5_out = net.forward12(conv5_out)
#         if attack_layer == 12:
#             relu5_out = easy_fault_apply(relu5_out, error_num, distance_level, bit_num)
#         pool3_out = net.forward13(relu5_out)
#         if attack_layer == 13:
#             pool3_out = easy_fault_apply(pool3_out, error_num, distance_level, bit_num)

#         apool_out = net.forward14(pool3_out)
#         flatt_out = net.forward15(apool_out)
#         class_out = net.forward16(flatt_out)
#         final_out = net.forward17(class_out)
#         scores = net.forward18(final_out)  # sfmax_out
#         # print(scores)

#         # exit()
#         # scores = net.forward1(conv1_out)
#         scores = scores.view(-1, args.num_classes)

#         pred = scores.data.max(1)[1]
#         target_list = target.squeeze().data
#         target_list = target_list.cpu().numpy()
#         '''
#         for pp,ll in enumerate(target_list):
#             #print(ll)
#             if ll == int(args.dest_class) and pred[pp].cpu().numpy() ==int(args.dest_class):
#                 save_image_tensor(data[pp].unsqueeze(0), 'dest_image.jpg')
#                 save_image_tensor(data_origin[pp].unsqueeze(0), 'dest_image_origin.jpg')
#             if ll == int(args.target_class) and pred[pp].cpu().numpy() !=int(args.target_class):
#                 #print('pred[pp].cpu().numpy() is ',pred[pp].cpu().numpy())
#                 #print('int(args.dest_class)', int(args.dest_class))
#                 #print('int(args.target_class) is',int(args.target_class))
#                 #exit()
#                 if ll == int(args.target_class) and pred[pp].cpu().numpy() == int(args.dest_class):
#                     misclassify_to_dest = misclassify_to_dest + 1
#                     save_image_tensor(data[pp].unsqueeze(0), 'target_misto_dest.jpg')
#                     save_image_tensor(data_origin[pp].unsqueeze(0), 'target_misto_dest_origin.jpg')
#                     #shutil.copyfile(os.path.join(args.test_image_dir, filename[pp]), '/home/chenxiangru/5to6_'+str(pp)+'.jpg')
#                 elif ll == int(args.target_class) and pred[pp].cpu().numpy() != int(args.target_class) and pred[pp].cpu().numpy() != int(args.dest_class):
#                     #shutil.copyfile(os.path.join(args.test_image_dir, filename[pp]), '/home/chenxiangru/5toothers_'+str(pp)+'.jpg')
#                     misclassify_to_others = misclassify_to_others + 1
#                     save_image_tensor(data[pp].unsqueeze(0), 'target_misto_other.jpg')
#                     save_image_tensor(data_origin[pp].unsqueeze(0), 'target_misto_other_origin.jpg')
#                     if int(pred[pp].cpu().numpy()) in misclassify_to_others_dir:
#                         misclassify_to_others_dir[int(pred[pp].cpu().numpy())] = misclassify_to_others_dir[int(pred[pp].cpu().numpy())] + 1
#                     else:
#                         misclassify_to_others_dir[int(pred[pp].cpu().numpy())] = 1
#                 #print(pred[pp].cpu().numpy())
#                 #exit()
#         '''
#         # exit()
#         # print(target_list[0])
#         # print(target_list[0].cpu().numpy())
#         # exit()
#         comparelist = torch.eq(target.squeeze(), pred).cpu().numpy()
#         for i in range(len(target_list)):
#             if str(target_list[i]) in correct_categories:
#                 if comparelist[i] == True:
#                     correct_categories[str(target_list[i])][0] = correct_categories[str(target_list[i])][0] + 1
#                     correct_categories[str(target_list[i])][1] = correct_categories[str(target_list[i])][1] + 1
#                 else:
#                     correct_categories[str(target_list[i])][0] = correct_categories[str(target_list[i])][0] + 1
#             else:
#                 if comparelist[i] == True:
#                     correct_categories[str(target_list[i])] = [1, 1]
#                 else:
#                     correct_categories[str(target_list[i])] = [1, 0]
#         test_correct += pred.eq(target.data).cpu().sum()
#     for key in correct_categories:
#         accuracy_categories[key] = float(correct_categories[key][1]) / float(correct_categories[key][0]) * 100
#     #print(accuracy_categories)
#     #print('misclassify_to_dest is ', misclassify_to_dest)
#     #print('misclassify_to_others is ', misclassify_to_others)
#     #print('misclassify_to_others_dir is ', misclassify_to_others_dir)
#     #print("Predicted {} out of {} correctly".format(test_correct, total_examples))
#     if args.modelname_tobe_saved != 'None':
#         torch.save(net.half().state_dict(), args.modelname_tobe_saved)
#     final_result = float (100.0 * test_correct / (float(total_examples)))
#     return final_result



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





cuda0 = torch.device('cuda:0')

x = torch.ones([2, 4,6,8], dtype=torch.float64, device=cuda0)

y = easy_fault_apply(x, 2, 'mantissa',5)

print(x)