#!/usr/bin/python
# encoding: utf-8

import os
import random
from PIL import Image
import numpy as np
from image import *
import torch

from torch.utils.data import Dataset
from utils import read_truths_args, read_truths, get_all_files

class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4, cell_size=32, bg_file_names=None):

      # root             : list of training or test images
      # shape            : shape of the image input to the network
      # shuffle          : whether to shuffle or not 
      # transform        : any pytorch-specific transformation to the input image 
      # target_transform : any pytorch-specific transformation to the target output
      # train            : whether it is training data or test data
      # seen             : the number of visited examples (iteration of the batch x batch size) # TODO: check if this is correctly assigned
      # batch_size       : how many examples there are in the batch
      # num_workers      : check what this is
      # bg_file_names    : the filenames for images from which you assign random backgrounds

       # read the the list of dataset images
       with open(root, 'r') as file:
           self.lines = file.readlines()

       # Shuffle
       if shuffle:
           random.shuffle(self.lines)

       # Initialize variables
       self.nSamples         = len(self.lines)
       self.transform        = transform
       self.target_transform = target_transform
       self.train            = train
       self.shape            = shape
       self.seen             = seen
       self.batch_size       = batch_size
       self.num_workers      = num_workers
       self.bg_file_names    = bg_file_names
       self.cell_size        = cell_size

    # Get the number of samples in the dataset
    def __len__(self):
        return self.nSamples

    # Get a sample from the dataset
    def __getitem__(self, index):

        # Ensure the index is smallet than the number of samples in the dataset, otherwise return error
        assert index <= len(self), 'index range error'

        # Get the image path
        imgpath = self.lines[index].rstrip()

        # Decide which size you are going to resize the image depending on the iteration
        if self.train and index % self.batch_size== 0:
            if self.seen < 400*self.batch_size:
               width = 13*self.cell_size
               self.shape = (width, width)
            elif self.seen < 800*self.batch_size:
               width = (random.randint(0,7) + 13)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 1200*self.batch_size:
               width = (random.randint(0,9) + 12)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 1600*self.batch_size:
               width = (random.randint(0,11) + 11)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 2000*self.batch_size:
               width = (random.randint(0,13) + 10)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 2400*self.batch_size:
               width = (random.randint(0,15) + 9)*self.cell_size
               self.shape = (width, width)
            elif self.seen < 3000*self.batch_size:
               width = (random.randint(0,17) + 8)*self.cell_size
               self.shape = (width, width)
            else: 
               width = (random.randint(0,19) + 7)*self.cell_size
               self.shape = (width, width)

        if self.train:
            # If you are going to train, decide on how much data augmentation you are going to apply
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5

            # Get background image path
            random_bg_index = random.randint(0, len(self.bg_file_names) - 1)
            bgpath = self.bg_file_names[random_bg_index]    

            # Get the data augmented image and their corresponding labels
            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure, bgpath)

            # Convert the labels to PyTorch variables
            label = torch.from_numpy(label)
        
        else:
            # Get the validation image, resize it to the network input size
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            # Read the validation labels, allow upto 50 ground-truth objects in an image
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*21)
            if os.path.getsize(labpath):
                ow, oh = img.size
                tmp = torch.from_numpy(read_truths_args(labpath))
                tmp = tmp.view(-1)
                tsz = tmp.numel()
                if tsz > 50*21:
                    label = tmp[0:50*21]
                elif tsz > 0:
                    label[0:tsz] = tmp

        # Tranform the image data to PyTorch tensors
        if self.transform is not None:
            img = self.transform(img)

        # If there is any PyTorch-specific transformation, transform the label data
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Increase the number of seen examples
        self.seen = self.seen + self.num_workers

        # Return the retrieved image and its corresponding label
        return (img, label)
