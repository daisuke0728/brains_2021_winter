from genericpath import exists
import os
import json

import numpy as np
import cv2 


from tqdm import tqdm

def main(datadir):
    
    with open(os.path.join(datadir,'keypoints_source.json'), 'r') as f:
        keypoints_list = json.load(f)
    
    #rawファイルのディレクトリ
    source_dir = os.path.join(datadir,'images_source')
    target_dir = os.path.join(datadir,'images_target')
    #画像の保存ディレクトリ
    source_path = os.path.join(datadir,'images_source_png')
    target_path = os.path.join(datadir,'images_target_png')
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in tqdm(range(len(keypoints_list))):
        filename = keypoints_list[i]['filename']
        image_size = keypoints_list[i]['image_size']
        height = image_size[0]
        width = image_size[1]

        #rawファイルからの読み込み
        with open(os.path.join(source_dir,filename),'rb') as f:
            arr = np.fromfile(f,dtype=np.float64,count=height*width)
            image = arr.reshape((height,width))
            source_max = sorted(image.ravel())[-50]
            source_min = image[0,0]
            source_image = (image-source_min)/(source_max - source_min)*255
            
            source_image = np.where(source_image<0,0,source_image)
            source_image = np.where(source_image>255,255,source_image)

            cv2.imwrite(os.path.join(source_path,filename.replace(".raw","_source.png")),source_image)
            #print(f'source_max {image.max()} min {image.min()}')
        #f.close()
        
        with open(os.path.join(target_dir,filename),'rb') as f:
            arr = np.fromfile(f,dtype=np.float64,count=height*width)
            image = arr.reshape((height,width))
            image = image + (source_min-image[0,0])
            target_image = (image-source_min)/(source_max - source_min)*255
            
            target_image = np.where(target_image<0,0,target_image)
            target_image = np.where(target_image>255,255,target_image)
    
            cv2.imwrite(os.path.join(target_path,filename.replace(".raw","_target.png")),target_image)
            #plt.imshow(image)
            #fig.savefig(os.path.join(target_path,filename.replace(".raw",".png")))
            #print(f'target_max {image.max()} min {image.min()}')

if __name__=='__main__':
    datadir = '../'
    #raw画像をpng画像へと変換する関数
    main(datadir)