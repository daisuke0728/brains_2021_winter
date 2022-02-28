from . import voxelmorph2d as vm2d
from . import voxelmorph3d as vm3d
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims,args, is_2d=False, use_gpu=False):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
        #self.optimizer = optim.SGD(self.voxelmorph.parameters(), lr=args.lr,momentum=0.99)
        self.optimizer = optim.Adam(self.voxelmorph.parameters(), lr=args.lr)
        self.use_MSE = args.use_MSE
        self.only_MSE = args.only_MSE
        if self.only_MSE:
            self.use_MSE=True

    
    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return voxelmorph(x)

    def calculate_loss(self, y, ytrue,deformation_matrix, n=9, lamda=0.01,lamda_MSE=0.1, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue,deformation_matrix, n, lamda,lamda_MSE,use_MSE=self.use_MSE,only_MSE=self.only_MSE)
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01,lamda_MSE=0.1, return_metric_score=True):
        self.optimizer.zero_grad()
        registered_image,deformation_matrix = self.voxelmorph(batch_moving, batch_fixed,return_matrix=True)
        train_loss = self.calculate_loss(registered_image, batch_fixed,deformation_matrix, n, lamda,lamda_MSE)
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01,lamda_MSE=0.1,print_detail=False):
        with torch.set_grad_enabled(False):
            registered_image,deformation_matrix = self.voxelmorph(batch_moving, batch_fixed,return_matrix=True)
            val_loss = self.vm.vox_morph_loss(registered_image, batch_fixed,deformation_matrix, n, lamda,lamda_MSE=lamda_MSE,use_MSE=self.use_MSE,only_MSE=self.only_MSE,print_detail=print_detail)
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_loss, val_dice_score