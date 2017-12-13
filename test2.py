import feature_extraction
import numpy as np
from graph_construction import *


#dataset:array 922*2048

file=open("image.txt","r")
images=[]
for line in file.readlines():
    print(line)
    images.append(line)

np.random.shuffle(images)
file2=open("shuffle_image.txt","w")
for i in range(len(images)):
    file2.write(images[i])





