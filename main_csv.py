import csv
import os
import numpy as np
import cv2 as cv
from scipy import signal
import pandas as pd

# 统计工件文件夹下采集数据的文件个数
def all_excel_num_in_dir(path):
    number = 0
    for root, dirname, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] =='.csv':
                number += 1
    return number

#读取原始csv格式数据并时域频域预处理
def read_one_excel(filename):
    csv_data = []
    csv_data_file = filename
    with open(csv_data_file) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        data_header = next(csv_reader)
        for row in csv_reader:
            csv_data.append(row)
    csv_data = np.abs(np.array(csv_data, dtype='float32'))
    avg_data = np.sum(csv_data,0)
    avg_data = avg_data / csv_data.shape[0]

    return avg_data

#读取文件夹下所有文件并整理出预处理之后的汇总表
def read_and_avg_all_csvfile(path):
    file_numbers = all_excel_num_in_dir(path)
    exact_data = []
    for file_number in range(1, file_numbers+1):
        print(file_number)
        filename = path + str(file_number) + '.csv'
        one_avg_data = read_one_excel(filename)
        exact_data.append(one_avg_data)
    #print(avg_data)
    # name = ['Horizontal_avg', 'Vertical_avg', 'Horizontal_var','Vertical_var',
    #         'Horizontal_min','Vertical_min', 'Horizontal_max','Vertical_max',
    #         'Horizontal_kurt','Vertical_kurt', 'Horizontal_skew','Vertical_skew',
    #         'Horizontal_f1','Vertical_f1', 'Horizontal_f2','Vertical_f2',
    #         'Horizontal_f3','Vertical_f3', 'Horizontal_f5','Vertical_f5']
    save_temp_data = pd.DataFrame(data=exact_data)
    work_name = path.split('/')[1]
    temp = np.mat(save_temp_data)
    save_temp_data_name = './data/'+work_name + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')


#读取预处理之后的汇总表
def read_one_csv(filename, BATCH_SIZE):
    csv_data = []
    data_batch_size = np.zeros([123, BATCH_SIZE])# 这里面的123需要更改为csv文件数，也就是分钟
    csv_data_file = filename
    with open(csv_data_file) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        data_header = next(csv_reader)
        for row in csv_reader:
            csv_data.append(row)
    csv_data = np.asarray(csv_data, 'float32')
    csv_data = np.delete(csv_data, [0,1,2], axis=1)
    for i in range(BATCH_SIZE):
        data_batch_size[:, i] = csv_data[:, 0]
    data_batch_size = np.asarray(data_batch_size, 'float32')
    data_batch_size = np.transpose(data_batch_size)
    return data_batch_size

def gaussBlur(img, sigma, H, W, _boundary='fill', _fillvalue=0):
    # 构建水平方向上的高斯卷积核
    gaussKernel_x = cv.getGaussianKernel(W, sigma, cv.CV_64F)
    # 转置
    gaussKernel_x = np.transpose(gaussKernel_x)
    # 图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(img, gaussKernel_x, mode="same",
                                    boundary=_boundary, fillvalue=_fillvalue)
    # 构建垂直方向上的高斯卷积核
    gaussKernel_y = cv.getGaussianKernel(H, sigma, cv.CV_64F)
    # 与垂直方向上的高斯卷核
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same",
                                     boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy


def meanBlur(arry):
    meanBlur_data = np.mean(arry)
    return meanBlur_data


path = './Bearing3_1/'
filename = 'Bearing3_1.csv'
read_and_avg_all_csvfile(path)
check_number = all_excel_num_in_dir(path)





'''
##保存G_paintings
save_temp_data=[]
save_temp_data = G_paintings.detach().numpy()
save_temp_data = pd.DataFrame(data=save_temp_data)
work_name = path.split('/')[1]
save_temp_data_name =work_name+ '_' + str(maxiter) + '_合成_G_paintings.csv'
save_temp_data.to_csv(save_temp_data_name, encoding='gbk')


save_temp_data=[]
save_temp_data = G_paintings.detach().numpy()
save_temp_data = pd.DataFrame(data=save_temp_data)
save_temp_data = gaussBlur(save_temp_data, 2, 1, 3, "symm")
save_temp_data = pd.DataFrame(data=save_temp_data)
work_name = path.split('/')[1]
save_temp_data_name =work_name + '_' + str(maxiter) + '_高斯_合成_G_paintings.csv'
save_temp_data.to_csv(save_temp_data_name, encoding='gbk')
'''







