#!/usr/bin/env python
# coding: utf-8

import os,sys,time
import json
from tqdm import tqdm

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import argparse

#データの読み込み

source_dir = os.path.abspath('../images_source')
target_dir = os.path.abspath('../images_target')
keypoint_file = os.path.abspath('../keypoints_source.json')
with open(keypoint_file, 'r') as f:
    keypoints_list = json.load(f)


def load_image(fname,dir,image_size=[256,256]):
    with open(os.path.join(dir,fname), 'rb') as f:
        arr = np.fromfile(f, dtype=np.float64, count=image_size[0]*image_size[1])
        image = arr.reshape((image_size[0], image_size[1]))
        
        max_num = sorted(image.ravel())[-10]
        image = (image-image[0,0])/(max_num - image[0,0])*255

        image = np.where(image<0,0,image)
        image = np.where(image>255,255,image)
        return image

def load_images(fname,source_dir,target_dir,image_size=[256,256]):
    with open(os.path.join(source_dir,fname), 'rb') as f:
        arr = np.fromfile(f, dtype=np.float64, count=image_size[0]*image_size[1])
        image = arr.reshape((image_size[0], image_size[1])) 
        source_max = sorted(image.ravel())[-50]
        source_min = image[0,0]
        source_image = (image-source_min)/(source_max - source_min)*255
        
        source_image = np.where(source_image<0,0,source_image)
        source_image = np.where(source_image>255,255,source_image)
    f.close()

    with open(os.path.join(target_dir,fname), 'rb') as f:
        arr = np.fromfile(f, dtype=np.float64, count=image_size[0]*image_size[1])
        image = arr.reshape((image_size[0], image_size[1]))    
        image = image + (source_min-image[0,0])
        target_image = (image-source_min)/(source_max - source_min)*255
        
        target_image = np.where(target_image<0,0,target_image)
        target_image = np.where(target_image>255,255,target_image)
    f.close()

    return source_image,target_image
    

def calc_point(keypoint,H):
    # keypoint:[N,2]
    #H : homography [3,3] ホモグラフィ行列
    N,_ = keypoint.shape
    keypoint = np.concatenate([keypoint,np.ones((N,1)).astype(np.uint8)],1)
    warped_point = np.dot(H,keypoint.T)
    warped_point = warped_point/warped_point[2]
    return warped_point[:2].T.astype(np.uint8)

def estimate_point(args):
    type = args.type
    if type=='akaze':
        akaze = cv.AKAZE_create()
        bf = cv.BFMatcher(cv.NORM_L2)
    elif type=='sift':
        sift = cv.SIFT_create()
        bf = cv.BFMatcher(cv.NORM_L2)
    elif type=='orab':
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)

    #出力用のリスト
    json_data = list()
    if not os.path.exists(os.path.join(args.save_dir,'affine_matrix')):
        os.makedirs(os.path.join(args.save_dir,'affine_matrix'))

    for i,f in enumerate(tqdm(keypoints_list)):
        data = dict()
        fname = f["filename"]
        image_size = f["image_size"]
        data['filename'] = fname

        keypoint = list()
        
        #画像の読み込み
        #source_image = load_image(fname,source_dir,image_size).astype(np.uint8)
        #source_image = cv.Laplacian(source_image, cv.CV_64F)
        #target_image = load_image(fname,target_dir,image_size).astype(np.uint8)
        #target_image = cv.Laplacian(target_image, cv.CV_64F)
        source_image,target_image = load_images(fname,source_dir,target_dir,image_size)
        source_image = source_image.astype(np.uint8)
        target_image = target_image.astype(np.uint8)
        #キーポイントの追加
        for j in range(len(f["keypoints"])):
            keypoint.append(f["keypoints"][j]["pixel"])
        keypoint = np.array(keypoint)

        if type=='akaze':
            kp1, des1 = akaze.detectAndCompute(source_image, None)
            kp2, des2 = akaze.detectAndCompute(target_image, None)
        elif type=='sift':
            kp1, des1 = sift.detectAndCompute(source_image, None)
            kp2, des2 = sift.detectAndCompute(target_image, None)
        elif type=='orab':
            kp1, des1 = orb.detectAndCompute(source_image, None)
            kp2, des2 = orb.detectAndCompute(target_image, None)

        # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
        if type=='akaze' or type=='sift':
            matches = bf.knnMatch(des1, des2, k=2)
        elif type=='orab':
            matches = bf.knnMatch(des1,des2)

        # 正しいマッチングのみ保持
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append([m])
        #matches_img = cv.drawMatchesKnn(source_image,kp1,target_image,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 適切なキーポイントを選択
        ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # ホモグラフィを計算
        H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
        
        if H is None:
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append([m])
            # 適切なキーポイントを選択
            ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # ホモグラフィを計算
            H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
            
            if H is None:
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.9 * n.distance:
                        good_matches.append([m])
                # 適切なキーポイントを選択
                ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                # ホモグラフィを計算
                H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
                
                if H is None:
                    print('Failure good_match<4 thr:0.9',fname)
                    
            
        np.savetxt(os.path.join(os.path.join(args.save_dir,'affine_matrix'),fname.replace('.raw','.txt')),H)

        #画像を変換
        #warped_image = cv.warpPerspective(source_image, H, (target_image.shape[1], target_image.shape[0]))
        #キーポイントを計算
        warped_point = calc_point(keypoint,H)
        #出力用のリスト
        key_list = list()
        for k in range(len(warped_point)):
            key_list.append({"id":k,"pixel":warped_point[k].tolist()})
        #print(key_list)
        data["keypoints"] = key_list
        json_data.append(data)

        #visualize(source_image,target_image,keypoint,warped_point,fname,args)
        
    #print(json_data)
    with open(os.path.join(args.save_dir,'submission.json'), 'w') as f:
        json.dump(json_data, f, indent=0)
    return 0

def visualize(source_image,target_image,key_source,key_target,fname,args):
    #print('visualizing!')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    source_image = cv.cvtColor(source_image, cv.COLOR_GRAY2RGB)
    target_image = cv.cvtColor(target_image, cv.COLOR_GRAY2RGB)
    for p in key_source:
        cv.circle(source_image,(p[0],p[1]), 3, (255,0,0), thickness=-1)
    for p in key_target:
        cv.circle(target_image,(int(p[0]),int(p[1])), 3, (255,0,0), thickness=-1)
    ax1.imshow(source_image,cmap='gray')
    ax1.set_title('batch_moving')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(target_image,cmap='gray')
    ax2.set_title('batch_fixed')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir,'result_images/')+fname.replace('.raw','.png'))
    #plt.close()

def main(args):
    estimate_point(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='cv2 model')
    parser.add_argument('--source_dir', help='source imageへのディレクトリ')
    parser.add_argument('--target_dir', help='target imageへのディレクトリ')
    parser.add_argument('--save_dir',type=str,default='../result/cv2_akaze/', help='保存用ディレクトリ')

    parser.add_argument('--type',type=str,default='akaze',help='キーポイント抽出方法の指定') 


    args = parser.parse_args()
    main(args)


