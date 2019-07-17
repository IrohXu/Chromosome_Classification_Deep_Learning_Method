import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

file_name = []
label = []
for i in range(1,120):
    for j in range(1,25):
        if os.path.isfile('./chromosomes_data/{} {}a.bmp'.format(i,j)):
            file_name.append('{} {}a.bmp'.format(i,j))
            label.append(j)
        if os.path.isfile('./chromosomes_data/{} {}b.bmp'.format(i,j)):
            file_name.append('{} {}b.bmp'.format(i,j))
            label.append(j)
"""
with open('all_label.txt','wb') as f:
    pass
for i in range(len(label)):
    with open('all_label.txt','a') as f:
        f.write('{}\t{}\n'.format(file_name[i],label[i]))
"""
with open('all_file.txt','wb') as f:
    pass
import io
f = io.open('file.txt', 'w', newline='\n')
for i in range(len(label)):
    f.write('\"{}\"\n'.format(file_name[i]))