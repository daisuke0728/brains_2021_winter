import os,glob,json

import time
import pickle
from natsort import natsorted
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim

import dataloader
import argparse

from models import voxelmorph,voxelmorph2d,voxelmorph3d 
import utils

def make_keypoint(vm,deformation_matrix,fname,args,keypoints_list,fname_to_id):
    dic = list()

    #キーポイント作成
    dx = deformation_matrix[:, :, :, 0]
    dy = deformation_matrix[:, :, :, 1]

    batch_size, height, width = dx.shape
    x_mesh, y_mesh = vm.voxelmorph.spatial_transform.meshgrid(height, width)
    x_mesh = x_mesh.expand([batch_size, height, width])
    y_mesh = y_mesh.expand([batch_size, height, width])
    
    x_new = dx + x_mesh
    y_new = dy + y_mesh
    
    for i in range(len(fname)):
        f = fname[i]
        keypoint_l = keypoints_list[fname_to_id[f]]['keypoints']
        #print(keypoint_l)
        data = dict()
        data['filename'] = f
        pred_keypoints = list()
        for j, pixel in enumerate(keypoint_l):
            point = pixel['pixel']
            #print(point)
            #print(x_new.size())
            pred_pixel = [x_new[i,point[1],point[0]].item(),y_new[i,point[1],point[0]].item()]
            pred_keypoints.append({'id': j, 'pixel': pred_pixel})
        
        data['keypoints'] = pred_keypoints
        dic.append(data)
    return dic

def train(train_loader,model,is_cuda,epoch,params,args):
    train_loss = 0
    train_dice_score = 0

    for batch_moving, batch_fixed,fname in train_loader:
        if is_cuda:
            batch_fixed = batch_fixed.to('cuda').float()
            batch_moving = batch_moving.to('cuda').float()
        loss, dice = model.train_model(batch_fixed, batch_moving,n=params['n'],lamda=params['lamda'],lamda_MSE=params['lamda_MSE'])
        train_dice_score += dice.data
        train_loss += loss.data
    train_loss /= len(train_loader)
    train_dice_score /=len(train_loader) 
    print(f'train_loss:{train_loss},train_dice_score:{train_dice_score}')
    return train_loss,train_dice_score

def valid(val_loader,vm,is_cuda,epoch,params,args):
    val_loss = 0
    val_dice_score = 0
    vm.voxelmorph.eval()
    with torch.no_grad():
        for batch_moving, batch_fixed,fname in val_loader:
            if is_cuda:
                batch_fixed = batch_fixed.to('cuda').float()
                batch_moving = batch_moving.to('cuda').float()
            loss, dice = vm.get_test_loss(batch_fixed, batch_moving,n=params['n'],lamda=params['lamda'],lamda_MSE=params['lamda_MSE'])
            val_dice_score += dice.data
            val_loss += loss.data
        val_loss /= len(val_loader)
        val_dice_score /=len(val_loader) 
        print(f'valid_loss:{val_loss},valid_dice_score:{val_dice_score}')
    return val_loss,val_dice_score

def test(test_loader,vm,is_cuda,params,args,keypoint_file):
    test_loss = 0
    test_dice_score = 0
    vm.voxelmorph.eval()
    json_data = list()

    with open(keypoint_file, 'r') as f:
        keypoints_list = json.load(f)
    f.close()

    fname_to_id = dict()
    for i,e in enumerate(keypoints_list):
        fname_to_id[e['filename']]=i

    with torch.no_grad():
        for batch_moving, batch_fixed,fname in tqdm(test_loader):
            if is_cuda:
                batch_fixed = batch_fixed.to('cuda').float()
                batch_moving = batch_moving.to('cuda').float()
            registered_image,deformation_matrix = vm.voxelmorph(batch_fixed, batch_moving,return_matrix=True)
            utils.visualize(batch_moving,batch_fixed,registered_image,fname,args)
            
            dic_l = make_keypoint(vm,deformation_matrix,fname,args,keypoints_list,fname_to_id)
            [json_data.append(e) for e in dic_l]
            
            loss, dice = vm.get_test_loss(batch_moving, batch_fixed,n=params['n'],lamda=params['lamda'],lamda_MSE=params['lamda_MSE'],print_detail=True)
            test_dice_score += dice.data
            test_loss += loss.data

        test_loss /= len(test_loader)
        test_dice_score /=len(test_loader) 
        print(f'valid_loss:{test_loss},valid_dice_score:{test_dice_score}')
        
        with open(os.path.join(args.save_dir,'submission.json'), 'w') as f:
            json.dump(json_data, f, indent=0)

    return test_loss,test_dice_score


def main(args):

    lr = args.lr
    max_epoch = args.max_epoch
    bs = args.bs

    if torch.cuda.is_available():
        print('cuda is available!')
        is_cuda = True
    else:
        print('cuda is not available!')
        is_cuda = False

    with open(os.path.join(args.save_dir,"./params.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    params = {'n':args.n,'lamda':args.lamda,'lamda_MSE':args.lamda_MSE}

    #load images
    print('Start loadiing source&target Images!')
    source_dir = os.path.abspath('../images_source')
    target_dir = os.path.abspath('../images_target')
    keypoint_file = os.path.abspath('../keypoints_source.json')
    trainval_dataset = dataloader.IMAGE_PAIR_SET(source_dir,target_dir,keypoint_file,pre_load=True)
    
    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * 0.8) 
    val_size = n_samples - train_size

    #train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(trainval_dataset,batch_size=bs,num_workers=4,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(trainval_dataset,batch_size=bs,num_workers=4,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(trainval_dataset,batch_size=128,drop_last=False,num_workers=4,pin_memory=True)

    vm = voxelmorph.VoxelMorph((1,256,256),args,is_2d=True,use_gpu=is_cuda)
    
    if not args.use_load_model:
        print('Start Training!')
        train_losses,train_dice_scores = list(),list()
        val_losses,val_dice_scores = list(),list()
        for epoch in tqdm(range(max_epoch)):
            #学習
            train_loss,train_dice_score = train(train_loader,vm,is_cuda,epoch,params,args)
            #検証
            val_loss,val_dice_score = valid(val_loader,vm,is_cuda,epoch,params,args)
            
            train_losses.append(train_loss)
            train_dice_scores.append(train_dice_score)
            val_losses.append(val_loss)
            val_dice_scores.append(val_dice_score)
        #モデルの保存
        torch.save(vm.voxelmorph.to('cpu').state_dict(),os.path.join(args.save_dir,'voxelmorph2d.pth'))
        if is_cuda:
            vm.voxelmorph.to('cuda')
    else:
        if is_cuda:
            vm.voxelmorph.load_state_dict(torch.load(os.path.join(args.save_dir,'voxelmorph2d.pth')))
            vm.voxelmorph.to('cuda')
        else:
            vm.voxelmorph.load_state_dict(torch.load(os.path.join(args.save_dir,'voxelmorph2d.pth')))
    
    #テスト
    test(test_loader,vm,is_cuda,params,args,keypoint_file)

    return 0

if __name__ == '__main__':
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(3407)
    seed_worker(3407)

    parser = argparse.ArgumentParser(description='voxelmorph model')
    parser.add_argument('--source_dir', help='source imageへのディレクトリ')
    parser.add_argument('--target_dir', help='target imageへのディレクトリ')
    parser.add_argument('--save_dir',type=str,default='../result/n5/', help='保存用ディレクトリ')
    
    parser.add_argument('--n',type=int,default=5) 
    parser.add_argument('--lamda',type=float,default=0.0002) 
    parser.add_argument('--lamda_MSE',type=float,default=0.001) 
    parser.add_argument('--use_MSE',action='store_true') 
    parser.add_argument('--only_MSE',action='store_true') 

    parser.add_argument('--lr',type=float,default=0.001) 
    parser.add_argument('--max_epoch',type=int,default=100) 
    parser.add_argument('--bs',type=int,default=32) 
    parser.add_argument('--use_load_model',action='store_true') 

    args = parser.parse_args()
    print('args:',args)
    print('cpu num:',os.cpu_count())

    main(args)


