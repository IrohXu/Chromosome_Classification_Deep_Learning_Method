import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import scipy.signal as signal
import os

class Straightening():
    def __init__(self):
        self.w1 = 0.43
        self.w2 = 0.57    

    def main_strainghtening(self, path_img):
        '''
        main function
        return: the straighted img
        '''
        img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
        degree, row_bend, img_coped, for_show, mask_rotated = self.find_bend_point_0(img)
        img_upper_rotated, img_lower_rotated = self.separate_rotate_1(img_coped, row_bend)
        width1 = img_upper_rotated.shape[1]
        width2 = img_lower_rotated.shape[1]
        width_diff = width1 - width2
        if width_diff > 0:
            img_lower_rotated = cv2.copyMakeBorder(img_lower_rotated.copy(), top=0, bottom=0, left=0,
                                                    right=width_diff, borderType=cv2.BORDER_CONSTANT, value=255)
        else:
            img_upper_rotated = cv2.copyMakeBorder(img_upper_rotated.copy(), top=0, bottom=0, left=0,
                                                    right=np.abs(width_diff), borderType=cv2.BORDER_CONSTANT, value=255)
        img_new = np.concatenate((img_upper_rotated[:-3], img_lower_rotated[3:]), axis=0)
        
        return img_new

    
    def find_bend_point_0(self, img):
        '''
        First produce the binary image and then
        Use the information about the horizontal projection vector to find the bend point
        '''

        boarder_size = max(img.shape[0] // 2, img.shape[1] // 2)
        img = cv2.copyMakeBorder(img.copy(), top=boarder_size, bottom=boarder_size, left=boarder_size,
                                 right=boarder_size, borderType=cv2.BORDER_CONSTANT, value=255)
        rows, columns = img.shape
        retval, img_binary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)  # 对图像二值化操作
        img_binary = cv2.bitwise_not(img_binary)    # 二值化像素值反转
        S_old = np.inf
        degree_opt = 0
        row_bend_opt = 0
        img_coped = 0
        for_show = 0
        mask_rotated_return = 0
        for degree in range(0, 180, 5):   # 逐个旋转
            rotate_Matrix = cv2.getRotationMatrix2D((rows//2, columns//2), degree, 1)
            mask_rotated = cv2.warpAffine(img_binary, rotate_Matrix, (2*columns, 2*rows))
            mask_bounding = cv2.boundingRect(mask_rotated)
            # 选取出能框住染色体的最小矩形
            mask_rotated1 = mask_rotated[mask_bounding[1]:mask_bounding[1] + mask_bounding[3],
                                        mask_bounding[0]:mask_bounding[0] + mask_bounding[2]]
            S_new, row_bend, for_show1 = self.S_score_calculate(mask_rotated1, degree)  # 计算paper中的S值
            img_rotated = cv2.warpAffine(img, rotate_Matrix, (2*columns, 2*rows))
            img_coped1 = img_rotated[mask_bounding[1]:mask_bounding[1] + mask_bounding[3],
                                    mask_bounding[0]:mask_bounding[0] + mask_bounding[2]]
            if S_new < S_old:
                S_old = S_new
                degree_opt = degree
                row_bend_opt = row_bend
                img_coped = img_coped1
                for_show = for_show1
                mask_rotated_return = mask_rotated1
        return degree_opt, row_bend_opt, img_coped, for_show, mask_rotated_return
    
    def separate_rotate_1(self, img_coped, bend_row):
        '''
        相当于从bend的位置直接分割图像为二
        img, the same as mask
        :return:
        '''
        img_upper = img_coped[:bend_row, :]
        img_lower = img_coped[bend_row:, :]
        img_upper_rotated = self.find_h_rotate_degree(img_upper)
        img_lower_rotated = self.find_h_rotate_degree(img_lower)
        return img_upper_rotated, img_lower_rotated
    
    def find_h_rotate_degree(self, img_separate):
        '''
        assistant function for separate_rotate_1
        :param img_separate:
        :return:
        '''
        boarder_size = max(img_separate.shape[0] // 2, img_separate.shape[1] // 2)
        img = cv2.copyMakeBorder(img_separate.copy(), top=boarder_size, bottom=boarder_size, left=boarder_size,
                                 right=boarder_size, borderType=cv2.BORDER_CONSTANT, value=255)
        rows, columns = img.shape
        retval, img_binary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
        img_binary = cv2.bitwise_not(img_binary)
        x_projection_width = np.inf
        for degree in range(-90, 90, 5):
            rotate_Matrix = cv2.getRotationMatrix2D((rows // 2, columns // 2), degree, 1)
            mask_rotated = cv2.warpAffine(img_binary, rotate_Matrix, (2 * columns, 2 * rows))
            mask_bounding = cv2.boundingRect(mask_rotated)
            mask_rotated = mask_rotated[mask_bounding[1]:mask_bounding[1] + mask_bounding[3],
                           mask_bounding[0]:mask_bounding[0] + mask_bounding[2]]
            x_axis_projection = np.sum(mask_rotated, axis=0)/255
            x_projection_index = np.where(x_axis_projection > 0)
            width1 = x_projection_index[0][-1] - x_projection_index[0][0]
            img_rotated = cv2.warpAffine(img, rotate_Matrix, (2 * columns, 2 * rows))
            img_coped1 = img_rotated[mask_bounding[1]:mask_bounding[1] + mask_bounding[3],
                         mask_bounding[0]:mask_bounding[0] + mask_bounding[2]]
            if width1 < x_projection_width:
                x_projection_width = width1
                img_coped = img_coped1
        return img_coped
    
    def S_score_calculate(self, mask_rotated, degree):
        '''
        assistant function for find_bend_point_0
        :param mask_rotated:  cropped and rotated mask
        :return: S core in the equation (1)
        '''
        from scipy.interpolate import interp1d

        y_projection = np.sum(mask_rotated, axis=1)/255
        inter = interp1d(np.arange(len(y_projection)), y_projection, kind="cubic")
        projection_inter_index = np.linspace(0, len(y_projection) - 1, (1 / 2) * len(y_projection))
        projection_inter = inter(projection_inter_index)
        sag_index = signal.argrelextrema(projection_inter, np.less)
        sags = projection_inter[sag_index]
        if len(sags) == 0:
            return np.Infinity, None, None
        sag_min_index = sag_index[0][np.argmin(sags)]
        sag_min = projection_inter[sag_min_index]
        projection_inter1 = projection_inter[: sag_min_index]
        peaks_index1 = signal.argrelextrema(projection_inter1, np.greater)
        projection_inter2 = projection_inter[sag_min_index:]
        peaks_index2 = signal.argrelextrema(projection_inter2, np.greater)
        if len(peaks_index1[0]) == 0 or len(peaks_index2[0]) == 0:
            return np.Infinity, None, None
        peaks_1 = np.max(projection_inter1[peaks_index1])
        peaks_2 = np.max(projection_inter2[peaks_index2])
        R1 = np.abs(peaks_1-peaks_2)/(peaks_1 + peaks_2)
        R2 = sag_min/(peaks_1 + peaks_2)
        S = self.w1 * R1 + self.w2 * R2
        for_show = (y_projection, projection_inter_index, projection_inter, 2*sag_min_index, sag_min, 2*peaks_index1[0],
                    projection_inter1[peaks_index1[0]], 2*(peaks_index2[0] + sag_min_index),
                    projection_inter2[peaks_index2[0]])
        return S, 2*sag_min_index, for_show
    
if __name__ == "__main__":
    from skimage import io
    import copy
    straighteningset = ['clase1','clase2','clase3','clase4','clase5','clase6','clase7','clase8','clase9','clase10','clase11','clase12','clase23']
    bound = {}
    bound['clase1'] = 3.90
    bound['clase2'] = 3.70
    bound['clase3'] = 3.00
    bound['clase4'] = 2.90
    bound['clase5'] = 2.75
    bound['clase6'] = 2.50
    bound['clase7'] = 2.50
    bound['clase8'] = 2.40
    bound['clase9'] = 2.30
    bound['clase10'] = 2.20
    bound['clase11'] = 2.00
    bound['clase12'] = 2.00
    bound['clase23'] = 2.20
    set_num = 612
    paths_img = './single_clase'
    straightening = Straightening()
    patient_num = 1
    for f in os.listdir(paths_img + '/'):
        new_root = './straightening_single_chromosomes' + '/' + str(patient_num)
        file = paths_img + '/' + f
        for i in os.listdir(file + '/'):
            label = i.split('e')[1]
            old_root = file + '/' + i
            double = 0
            for j in os.listdir(old_root + '/'):
                if double == 0:
                    if i in straighteningset:
                        img = cv2.imread(old_root + '/' + j, cv2.IMREAD_GRAYSCALE)
                        rows, columns = img.shape
                        if(float(rows/columns)<=bound[i]):
                            img = straightening.main_strainghtening(old_root + '/' + j)
                        io.imsave(new_root + '/' + str(patient_num) + '_' + label + 'a' + '.bmp', img)
                    else:
                        img = cv2.imread(old_root + '/' + j, cv2.IMREAD_GRAYSCALE)
                        io.imsave(new_root + '/' + str(patient_num) + '_' + label + 'a' + '.bmp', img)
                    double = double + 1
                else:
                    if i in straighteningset:
                        img = cv2.imread(old_root + '/' + j, cv2.IMREAD_GRAYSCALE)
                        rows, columns = img.shape
                        if(float(rows/columns)<=bound[i]):                       
                            img = straightening.main_strainghtening(old_root + '/' + j)
                        io.imsave(new_root + '/' + str(patient_num) + '_' + label + 'b' + '.bmp', img)
                    else:
                        img = cv2.imread(old_root + '/' + j, cv2.IMREAD_GRAYSCALE)
                        io.imsave(new_root + '/' + str(patient_num) + '_' + label + 'b' + '.bmp', img)
        patient_num = patient_num + 1