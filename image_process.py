#encoding=utf-8
# author :clz
# 生成训练路径和标签
# usage :
#python image_process.py ../../data/dogvscat/train ../../data/dogvscat 0.8
import sys,glob
import numpy as np
import random,os 
path="image"
image_path_list=[path+i for i in os.listdir(path)]
image_list=[]
for idx in image_path_list:
	image_path=glob.glob(idx+"/*")
	image_list.extend(image_path)

output_filePath="output/"
validation_ratio = 0.2# file will be split into two parts, train:ratio , val : 1 - ratio

if not os.path.isdir(output_filePath): os.mkdir(output_filePath)



train_file_txt = output_filePath + "/train.txt"
val_file_txt = output_filePath + "/val.txt"

label_dir = {'dog':1,'cat':0}

with open(train_file_txt,'w') as trainfile ,open(val_file_txt,'w') as valfile:
	for ind,filename in enumerate(image_list):
		if ind % 10000 == 0:
			print("\r{} file processed".format(ind))
		name = os.path.basename(filename)
		segs = name.split('.')
		if len(segs) > 1 and segs[0] in label_dir:
			label = label_dir.get(segs[0],'error')		
		else:
			print("\r[warn] {} not valid, pass...".format(filename))
			continue
		ran = random.random()
		if ran <= validation_ratio:
			line = "{} {}\n".format(filename,label)
			trainfile.write(line)
		else:
			line = "{} {}\n".format(filename,label)
			valfile.write(line)

print("[INFO] finished.")
