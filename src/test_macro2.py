from ij import IJ
from ij.io import DirectoryChooser
import os
import json

import java.util.Properties

datadir = '/Users/mac/Desktop/practice/Fujifilm_brain_winter/Q2/distribution/'
matdir = 'Users/mac/Desktop/practice/Fujifilm_brain_winter/Q2/distribution/result/cv2_sift/affine_matrix_2d/'
with open(os.path.join(datadir,'keypoints_source.json'), 'r') as f:
	keypoints_list = json.load(f)
f.close()

keypoints_list.reverse()

for i in range(len(keypoints_list)):
	filename = keypoints_list[i]['filename']
	image_size = keypoints_list[i]['image_size']
	height = image_size[0]
	width = image_size[1]

	source_path = os.path.join(os.path.join(datadir,"images_source_png"),filename.replace(".raw","_source.png"))
	target_path = os.path.join(os.path.join(datadir,"images_target_png"),filename.replace(".raw","_target.png"))
	source_name = filename.replace(".raw","_source.png")
	target_name = filename.replace(".raw","_target.png")
	f = filename.replace(".raw","")
	txt_name = matdir+filename.replace(".raw",".txt")

	IJ.open(source_path)
	IJ.open(target_path)

	#IJ.run("bUnwarpJ","loadSourceAffineMatrix",txt_name)
	IJ.run("bUnwarpJ", "source_image="+source_name+" target_image="+target_name+" registration=Accurate image_subsample_factor=0 initial_deformation=[Coarse] final_deformation=[Very Fine] divergence_weight=0 curl_weight=0 landmark_weight=0 image_weight=1 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=["+datadir+"transf2/"+f+"_direct_transf.txt] save_inverse_transformation=["+datadir+"transf2/"+f+"_inverse_transf.txt] "+"bunwarpj.bUnwarpJ_.loadSourceAffineMatrix "+matdir+txt_name)

	#IJ.selectWindow("Log")
	#IJ.run("Close")
	#IJ.selectWindow("Registered Target Image")
	#IJ.run("Close")
	#IJ.selectWindow("Registered Source Image")
	#IJ.run("Close")
	IJ.selectWindow(target_name)
	IJ.run("Close")
	IJ.selectWindow(source_name)
	IJ.run("Close")
	

print('finish')