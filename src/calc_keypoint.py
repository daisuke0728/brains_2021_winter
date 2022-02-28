from locale import DAY_1
import os,sys
from tqdm import tqdm

import numpy as np
import cv2
import json

def calc_point(keypoint,mat_x,mat_y,step,cv_res):
    #print(cv_res)
    cv_pred = np.array([np.array(e['pixel']) for e in cv_res['keypoints']])
    #print(cv_pred)
    pred = np.zeros(keypoint.shape)
    for i,point in enumerate(keypoint):
        x,y = point
        x_n,y_n = int(x//step),int(y//step)
        #４つ角との距離を計算
        eps = 1e-16
        dx = (x - x_n*step)/step
        dy = (y - y_n*step)/step
        diff1_x = mat_x[y_n+1,x_n+1]-x_n*step
        diff1_y = mat_y[y_n+1,x_n+1]-y_n*step
        diff2_x = mat_x[y_n+2,x_n+1]-x_n*step
        diff2_y = mat_y[y_n+2,x_n+1]-y_n*step-step
        diff3_x = mat_x[y_n+2,x_n+2]-x_n*step-step
        diff3_y = mat_y[y_n+2,x_n+2]-y_n*step-step
        diff4_x = mat_x[y_n+1,x_n+2]-x_n*step-step
        diff4_y = mat_y[y_n+1,x_n+2]-y_n*step
        #バイリニア補完で求める
        pred_x = x+ (1-dx)*(1-dy)*diff1_x +(1-dx)*dy*diff2_x +dx*dy*diff3_x +dx*(1-dy)*diff4_x
        pred_y = y+ (1-dx)*(1-dy)*diff1_y +(1-dx)*dy*diff2_y +dx*dy*diff3_y +dx*(1-dy)*diff4_y
        pred[i] = np.array([pred_x,pred_y])
        #pred[i] = np.array([int(pred_x),int(pred_y)])
        """
        if ((pred[i]-cv_pred[i])**2).sum()>100:
            print('take average!')
            pred[i] = (cv_pred[i]+pred[i])/2
        """
        
        #値を整数値に変換
        pred[i] = np.array([int(pred[i,0]),int(pred[i,1])])
        if np.any(pred[i]>255) or np.any(pred[i]<0):
            print('     Value error!       ')
    #print(pred)

    return pred

def main():
    keypoint_file = os.path.abspath('../keypoints_source.json')
    with open(keypoint_file, 'r') as f:
        keypoints_list = json.load(f)
    f.close()
    with open('../result/cv2_sift/submission.json','r') as h:
        cv2_res = json.load(h)

    mat_dir = '../transf/'
    #出力用のリスト
    json_data = list()
    for i,f in enumerate(tqdm(keypoints_list)):
        data = dict()
        fname = f["filename"]
        image_size = f["image_size"]
        data['filename'] = fname
        print(fname)
        keypoint = list()
        for j in range(len(f["keypoints"])):
            keypoint.append(f["keypoints"][j]["pixel"])
        keypoint = np.array(keypoint)

        if os.path.exists(os.path.join(mat_dir,fname.replace('.raw','_inverse_transf.txt'))):
            with open(os.path.join(mat_dir,fname.replace('.raw','_inverse_transf.txt')), 'r') as g:
                mat_f = g.readlines()
                interval = int(mat_f[0][10:])
                #print(interval)
                
                mat_x = np.zeros((interval+3,interval+3))
                mat_y = np.zeros((interval+3,interval+3))
                for j in range(interval+3):
                    if j==0:
                        x = mat_f[3+j][2:].replace(' \n','').split(' ')
                        x = [e for e in x if e!='']
                        #print(x)
                        mat_x[j,:] = np.array(x)
                    else:
                        x = mat_f[3+j][3:].replace(' \n','').split(' ')
                        x = [e for e in x if e!='']
                        #print(x)
                        mat_x[j,:] = np.array(x)
                for j in range((interval+3)):
                    if j==0:
                        y = mat_f[8+interval+j][2:].replace(' \n','').split(' ')
                        y = [e for e in y if e!='']
                        #print(y)
                        mat_y[j] = np.array(y)
                    else:
                        y = mat_f[8+interval+j][3:].replace(' \n','').split(' ')
                        y = [e for e in y if e!='']
                        #print(y)
                        mat_y[j] = np.array(y)
                
            step = 255/interval
            warped_point = calc_point(keypoint,mat_x,mat_y,step,cv2_res[i])
            #出力用のリスト
            key_list = list()
            for k in range(len(warped_point)):
                key_list.append({"id":k,"pixel":warped_point[k].tolist()})
            data["keypoints"] = key_list
            json_data.append(data)
        
        else:
            #cv2_siftから読み取り
            #print(cv2_res[i])
            #print(type(cv2_res[i]))
            json_data.append(cv2_res[i])
                
    #提出ファイルへの出力
    with open(os.path.join('../result/fiji','submission.json'), 'w') as f:
        json.dump(json_data, f, indent=0)
    return 0

if __name__=='__main__':
    main()