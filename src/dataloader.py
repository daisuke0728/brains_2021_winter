import os
import sys

import time
import pickle
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
from torch.utils.data import Dataset,DataLoader

class IMAGE_PAIR_SET(Dataset):
    def __init__(self,source_dir,target_dir,keypoint_file,width=256,height=256,pre_load=False,save=True):

        with open(keypoint_file, 'r') as f:
            keypoints_list = json.load(f)
        print('ファイル数:', len(keypoints_list))
        
        source_path = os.path.join(os.path.abspath(os.path.join(source_dir,os.pardir)),'images_source_png')
        target_path = os.path.join(os.path.abspath(os.path.join(target_dir,os.pardir)),'images_target_png')

        if not pre_load:
            self.source_tensor = []
            self.target_tensor = []
            for i in tqdm(range(len(keypoints_list))):
                filename = keypoints_list[i]['filename']
                print(filename)
                image_size = keypoints_list[i]['image_size']
                height = image_size[0]
                width = image_size[1]
        
                #rawファイルからの読み込み
                with open(os.path.join(source_dir,filename),'rb') as f:
                    arr = np.fromfile(f,dtype=np.float64,count=height*width)
                    image = arr.reshape((height,width))
                    #print(np.max(image))
                    cv2.imwrite(os.path.join(source_path,filename.replace(".raw",".png")),((image-image.min())/(image.max()-image.min())*255).astype(np.uint8))
                    #fig = plt.figure()
                    #plt.imshow(image)
                    #fig.savefig(os.path.join(source_path,filename.replace(".raw",".png")))
                    self.source_tensor.append(torch.from_numpy(image))
                    print(f'source_max {image.max()} min {image.min()}')
                #f.close()
                
                with open(os.path.join(target_dir,filename),'rb') as f:
                    arr = np.fromfile(f,dtype=np.float64,count=height*width)
                    image = arr.reshape((height,width))
                    cv2.imwrite(os.path.join(target_path,filename.replace(".raw",".png")),((image-image.min())/(image.max()-image.min())*255).astype(np.uint8))
                    #plt.imshow(image)
                    #fig.savefig(os.path.join(target_path,filename.replace(".raw",".png")))
                    self.target_tensor.append(torch.from_numpy(image))
                    print(f'target_max {image.max()} min {image.min()}')
                #f.close()

            if save:
                #作成したテンソルの保存
                f = open('../dataset/source_tensor_2d.pickle','wb')
                pickle.dump(self.source_tensor,f)
                f.close
                f = open('../dataset/target_tensor_2d.pickle','wb')
                pickle.dump(self.target_tensor,f)
                f.close
        else:
            #保存しているpickleファイルから読み込み
            with open("../dataset/source_tensor_2d.pickle", mode="rb") as f:
                self.source_tensor = pickle.load(f)
            with open("../dataset/target_tensor_2d.pickle", mode="rb") as f:
                self.target_tensor = pickle.load(f)

    def __len__(self):
        return len(self.source_tensor)

    def __getitem__(self, idx):
        return self.source_tensor[idx],self.target_tensor[idx]