from configparser import ConfigParser
import ast
import os

config = ConfigParser()
config.read('test_cfg.cfg')

# single variables
batch_size =config.get('parameters_main', 'batch_size')
epoch =config.get('parameters_main', 'epoch')
learning_rate =config.get('parameters_main', 'learning_rate')
cudanum =config.get('parameters_main', 'cudanum')
epoch =config.get('parameters_main', 'epoch')
param_list = [32,2,3]


print(config.get('section1', 'var3'))

print(config.get('section2', 'var4'))
print(config.get('section2', 'var5'))
print(config.get('section2', 'var6'))

# lists
l1 = config.get('section1', 'list1').split(',')
l2 = config.get('section1', 'list2').split(',')
l3 = map(lambda s: s.strip('\''), config.get('section1', 'list3').split(','))

print(l1,type(l1))
print(l2,type(l2))
print(l3,type(l3))

# dictionaries
batch_size = ast.literal_eval(config.get('parameters_main', 'batch_size'))
print(batch_size, type(batch_size))

d2 = ast.literal_eval(config.get('section3', 'dict2'))
batch_size = d2
print(d2, type(d2))

d3 = ast.literal_eval(config.get('section3', 'dict3'))
print(d3, type(d3))

d4 = ast.literal_eval(config.get('section3', 'dict4'))
print(d4, type(d4))
print(d4['key1'], type(d4['key1']))
print(d4['key1'][1], type(d4['key1'][1]))
print(d4['key1'][2], type(d4['key1'][2]))

os.system('python main.py --param_list')