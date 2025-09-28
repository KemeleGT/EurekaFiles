from torch.utils import data
import torchvision.datasets as datasets
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from PIL import Image
import torch
import os
import random
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Pubfig(data.Dataset):
    """Dataset class for the Pubfig dataset."""

    def __init__(self, lines, glbattr2idx, glbidx2attr, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs  # now it is all attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = glbattr2idx  # {}
        self.idx2attr = glbidx2attr  # {}
        self.preprocess(lines)
        # print (image_dir)
        # print (attr_path)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self, lines):
        """Preprocess the CelebA attribute file."""
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            other = 1
            label = 0  # []
            dictidx = 0
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                # label.append(values[idx] == '1')
                if values[idx] == '1':
                    other = 0
                    label = dictidx
                dictidx = dictidx + 1
            # if other == 1:
            # label=len(self.selected_attrs)-1#label.append(other == 1)
            if (i + 1) < 876 / 5:  # ?2000 ??test
                # print(label)
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        # print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        ##dataset_train = self.train_dataset
        ##dataset_test = self.test_dataset
        filename, label = dataset[index]
        ##filename_train, label_train = dataset_train[index]
        image = Image.open(os.path.join(self.image_dir, filename))  # Image.open xc
        image = image.convert('RGB')
        ##image_train = Image.open(os.path.join(self.image_dir, filename))
        ##image_test = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.from_numpy(np.array(label))  # torch.FloatTensor(label) #

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir='', attr_path='', selected_attrs='', crop_size=227, image_size=227, pytorch_dataset=False,
               # 128
               batch_size=16, dataset='CelebA', rotation=False, blurred=False, noise=False, noise_mean=0., noise_var=1.,
               mode='train', num_workers=0):
    """Build and return a data loader."""
    # print('pytorch_dataset is ',pytorch_dataset)
    # if pytorch_dataset == False:
    # torch.cuda._initialized = True
    g_cuda = torch.Generator(device='cuda:0')
    transform1 = []
    transform2 = []
    if mode == 'train':
        transform1.append(T.RandomHorizontalFlip())
        transform2.append(T.RandomHorizontalFlip())
    # transform1.append(T.CenterCrop(crop_size))
    # transform2.append(T.CenterCrop(crop_size))
    transform1.append(T.Resize((image_size, image_size)))
    transform2.append(T.Resize((image_size, image_size)))
    if mode == 'train' and rotation == True:
        print('rotation is true')
        # transform.append(T.RandomRotation(180, resample=False, expand=False, center=None))
        transform1.append(T.RandomVerticalFlip())
        transform2.append(T.RandomVerticalFlip())
    if mode == 'train' and blurred == True:
        transform1.append(T.GaussianBlur(5))
        transform2.append(T.GaussianBlur(5))
    transform1.append(T.ToTensor())
    transform2.append(T.ToTensor())
    transform1.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5,
                                                             0.5)))  # (0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975)))#
    transform2.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5,
                                                             0.5)))  # (0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975)))#
    if noise == True:
        transform1.append(AddGaussianNoise(noise_mean, noise_var))
        # transform2.append(AddGaussianNoise(noise_mean, noise_var))
    transform1 = T.Compose(transform1)
    transform2 = T.Compose(transform2)
    if pytorch_dataset == False:
        # ----pre-preprocess
        glbattr2idx = {}
        glbidx2attr = {}
        lines = [line.rstrip() for line in open(attr_path, 'r')]
        all_attr_names = lines[0].split()
        # print('lines0 = ', lines[0])
        # print('all attr name: ' + str(all_attr_names))
        for i, attr_name in enumerate(all_attr_names):
            glbattr2idx[attr_name] = i
            glbidx2attr[i] = attr_name
            # print('attr2idx: ' + str(self.attr2idx))
        lines = lines[1:]
        random.seed(1234)
        random.shuffle(lines)
        # ----
        dataset1 = Pubfig(lines, glbattr2idx, glbidx2attr, image_dir, attr_path, all_attr_names, transform1,
                          mode)  # selected_attrs
        # dataset_test = Pubfig(lines, glbattr2idx, glbidx2attr, image_dir, attr_path, selected_attrs, transform, mode='test')
        dataset2 = Pubfig(lines, glbattr2idx, glbidx2attr, image_dir, attr_path, all_attr_names, transform2,
                          mode='test')
        # print('current device is ', torch.cuda.current_device())
        # print('device_count is ', torch.cuda.device_count())
        # print('default_generators is ', torch.cuda.default_generators)
        g_cuda.manual_seed(0)
        data_loader1 = data.DataLoader(dataset=dataset1,
                                       batch_size=batch_size,
                                       shuffle=(mode == 'train'),
                                       num_workers=num_workers)
        g_cuda.manual_seed(0)
        data_loader2 = data.DataLoader(dataset=dataset2,
                                       batch_size=batch_size,
                                       shuffle=(mode == 'train'),
                                       num_workers=num_workers)
        '''data_loader_test = data.DataLoader(dataset=dataset_test,
                                      batch_size=batch_size,
                                      shuffle=('test'=='train'),
                                      num_workers=num_workers)'''

        return data_loader1, data_loader2
    elif pytorch_dataset == True:
        kwargs = {'num_workers': 1, 'pin_memory': True}
        g_cuda.manual_seed(0)
        data_loader1 = torch.utils.data.DataLoader(
            datasets.CIFAR10('../', train=(mode == 'train'), download=True,
                             transform=transform1),
            batch_size=batch_size, shuffle=True, **kwargs)
        g_cuda.manual_seed(0)
        data_loader2 = torch.utils.data.DataLoader(
            datasets.CIFAR10('../', train=(mode == 'train'), download=True,
                             transform=transform2),
            batch_size=batch_size, shuffle=True, **kwargs)
        return data_loader1  # ,data_loader2
# ------------------------- read dataset could change to celebA  /CIFAR10
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# print(datasets)
# train_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10('../', train=True, download=False,
#                   transform=transforms.Compose([
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
#                   ])),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#    datasets.CIFAR10('../', train=False, transform=transforms.Compose([
#                       transforms.RandomHorizontalFlip(),
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
#                   ])),
#    batch_size=args.batch_size, shuffle=True, **kwargs)