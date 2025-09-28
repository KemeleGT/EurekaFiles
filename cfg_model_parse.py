#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:08:32 2021

@author: mmerugu, xiangruchen, dipalhalder
"""

from configparser import ConfigParser
import ast
import os
import datetime
import argparse
parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--noise_level', type=float, default=0.5, help='noise level of whole dataset')  # necessory
parser.add_argument('--epoch', type=int, default=55, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='learning rate')
args = parser.parse_args()

starttime = datetime.datetime.now()

config = ConfigParser()
config.read('test_config_explore.cfg')
# config.read('test_config_range.cfg')
# config.read('test_config_evaluation.cfg')


# parameters = ast.literal_eval(config.get('test_param_original_train1', 'param'))
# param_list=[]
# for i in parameters:
# #    print("--"+str(i)+" "+str(d1[i]), end=" ")
#     param_list.append("--"+str(i)+" "+str(parameters[i]))
# cmdo1 = "python main.py" # traing the nn with basic settings
# for x in range(0, len(param_list)):
# #    print(l1[x])
#     cmdo1 = cmdo1+" "+param_list[x]
# cmdo1 = cmdo1 + ' >tool_origin_train1.log 2>&1'  # to wait for the finish, no 'no hup' and '&'
# print(cmdo1)
# #os.system(cmdo1)
# print('cmdo1 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_original_test1', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmdo2 = "python main.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmdo2 = cmdo2+" "+param_list[x]
#cmdo2 = cmdo2 + ' >tool_origin_test1.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
#print(cmdo2)
os.system(cmdo2)
print('------------------end of experiment------------------')
'''
parameters = ast.literal_eval(config.get('test_param_train1', 'param')) #with noise
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd1 = "python main.py" # traing the nn with basic settings
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd1 = cmd1+" "+param_list[x]
cmd1 = cmd1 + ' >tool_train1.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd1)
os.system(cmd1)
print('cmd1 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_test1', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd2 = "python main.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd2 = cmd2+" "+param_list[x]
cmd2 = cmd2 + ' >tool_test1.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd2)
os.system(cmd2)
print('cmd2 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_train2', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd4 = "python main.py" # retrain the nn using pretrained model and get new model
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd4 = cmd4+" "+param_list[x]
cmd4 = cmd4 + ' >tool_train2.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd4)
os.system(cmd4)
print('cmd4 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_test2', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd5 = "python main.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd5 = cmd5+" "+param_list[x]
cmd5 = cmd5 + ' >tool_test2.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd5)
os.system(cmd5)
print('cmd5 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_hamming1', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd6 = "python pth_calculate_hamming.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd6 = cmd6+" "+param_list[x]
cmd6 = cmd6 + ' >tool_hamming1.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd6)
os.system(cmd6)
print('cmd6 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_test3', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd7 = "python main.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd7 = cmd7+" "+param_list[x]
cmd7 = cmd7 + ' >tool_test3.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd7)
os.system(cmd7)
print('cmd7 finished ----------------------------')

parameters = ast.literal_eval(config.get('test_param_hamming2', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd8 = "python pth_calculate_hamming.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd8 = cmd8+" "+param_list[x]
cmd8 = cmd8 + ' >tool_hamming2.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd8)
os.system(cmd8)
print('cmd8 finished ----------------------------')

parameters = ast.literal_eval(config.get('calculate_vulnerability', 'param'))
param_list=[]
for i in parameters:
#    print("--"+str(i)+" "+str(d1[i]), end=" ")
    param_list.append("--"+str(i)+" "+str(parameters[i]))
cmd9 = "python vul_calculator.py" # 
for x in range(0, len(param_list)):
#    print(l1[x])
    cmd9 = cmd9+" "+param_list[x]
cmd9 = cmd9 + ' >tool_vul.log 2>&1'  # to wait for the finish, no 'no hup' and '&'    
print(cmd9)
os.system(cmd9)
print('cmd9 finished ----------------------------')

endtime = datetime.datetime.now()
print (endtime - starttime)
'''
