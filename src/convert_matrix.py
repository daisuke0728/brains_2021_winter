import os
import json

import numpy as np
import cv2 


from tqdm import tqdm

def main(datadir):
    flist = os.listdir(os.path.join(datadir,'affine_matrix'))
    if not os.path.exists(os.path.join(datadir,'affine_matrix_2d')):
        os.makedirs(os.path.join(datadir,'affine_matrix_2d'))
    
    for f in flist:
        fpath = os.path.join(os.path.join(datadir,'affine_matrix'),f)
        mat = np.loadtxt(fpath)
        
        g = open(os.path.join(os.path.join(datadir,'affine_matrix_2d'),f), 'w')
        print(os.path.join(os.path.join(datadir,'affine_matrix_2d'),f))
        g.write(str(mat[0,0])+' '+str(mat[0,1])+' '+str(mat[1,0])+' '+str(mat[1,1])+' '+
        str(mat[0,2])+' '+str(mat[1,2]))
        g.close()
        

    

if __name__=='__main__':
    txtdir = '../result/cv2_sift/'
    #raw画像をpng画像へと変換する関数
    main(txtdir)