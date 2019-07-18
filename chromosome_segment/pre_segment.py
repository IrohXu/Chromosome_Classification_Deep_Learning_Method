import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import scipy.signal as signal
import os
from skimage import io
import random
import scipy
import skimage.transform as sc

img = cv2.imread('D:/Desktop/chromosome_segment/raw_chromosome.jpg', cv2.IMREAD_GRAYSCALE)
img = np.array(img)

rows, columns = img.shape
# retval, img_binary = cv2.threshold(img, 238, 255, cv2.THRESH_BINARY)  # 二值化操作
retval_Otsu, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 使用ostu算法
# retval_Thresh, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)  # Thresh算法

# It seems that ostu is better!

contours, hierarchy = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(len(contours))
print(len(hierarchy))

pred_dir = 'pics_black'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

img2=[0 for i in range(len(contours))]
print(len(contours))

for i in range(len(contours)):
    b_image = Image.open('black_background.jpg','r').convert('L')
    b_img = np.array(b_image)
    b_img = sc.resize(b_img,(620,558),preserve_range=True)
    
    img2[i] = b_img #设置一张黑色背景图片
    cv2.drawContours(img2[i],contours[i],-1,(0,255,0),0)  #画边界 
    
    #全图片遍历找到相应的在轮廓之内的点
    for a in range(rows):
        for b in range(columns):
            #辨别是否在轮廓内是定义为1，不是定义为-1
            result = cv2.pointPolygonTest(contours[i], (a,b), False)
            if result > 0:
                img2[i][b,a] = img[a,b]  
    #保存
    #scipy.misc.imsave('pic_'+str(i)+'.jpg',img2[i])
    plt.imsave(os.path.join(pred_dir, str(i + 1) + '.png'), img2[i], cmap = 'gray')