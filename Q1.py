import sys
import os
import json

import math
import numpy as np
import pandas as pd

def rotmatrix_2d(ang):
    return np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])

def rot_transfer(inp,ang=np.pi/6,cent=np.array([[0.,0.]])):
    inp = np.array(inp)
    out = np.dot(rotmatrix_2d(ang),(inp-cent).T).T+cent
    
    return out.tolist()

def linear_transfer(inp,cent=np.array([[128,128]]),k=1.28):
    inp = np.array(inp)
    out = (inp-cent)*k +cent
    return out.tolist()

def move_transfer(inp,move_x=12.,move_y=8.):
    inp = np.array(inp)
    out = inp + np.array([[move_x,move_y]])
    return out.tolist()


def affine_transfer(inp,ang=math.pi/6,cent=np.array([[128.,128.]]),k=1.28):
    out = rot_transfer(inp,ang=ang,cent=cent)
    out = linear_transfer(out,cent=cent,k=k)
    out = move_transfer(out)
    return out

if __name__=='__main__':
    input_json = json.load(open('points.json', 'r'))
    #print(input_json)

    outQ1_1 = rot_transfer(input_json['Q1_1'])
    outQ1_2 = linear_transfer(input_json['Q1_2'])
    outQ1_3 = affine_transfer(input_json['Q1_3'],ang=math.pi/15,k=0.8)
    #outQ1_3 = [[0,0] for _ in range(len(input_json['Q1_3']))]

    dic = {'Q1_1':outQ1_1,'Q1_2':outQ1_2,'Q1_3':outQ1_3}
    #print(dic)
    with open('submission.json','w') as f:
        json.dump(dic,f)

    #print(affine_transfer([[128,128],[129,128]],ang=2.0*math.pi/3,k=0.8))


