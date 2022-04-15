import csv
import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import signal
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 统计工件文件夹下采集数据的文件个数
def all_excel_num_in_dir(path):
    number = 0
    for root, dirname, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] =='.csv':
                number += 1
    return number

#读取原始csv格式数据并预处理
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
    avg_data = []
    for file_number in range(1, file_numbers+1):
        print(file_number)
        filename = path + str(file_number) + '.csv'
        one_avg_data = read_one_excel(filename)
        avg_data.append(one_avg_data)
    print(avg_data)
    name = ['left', 'right']
    save_temp_data = pd.DataFrame(columns=name,data=avg_data)
    work_name = path.split('/')[1]
    temp = np.mat(save_temp_data)
    save_temp_data_name = work_name + '.csv'
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



# Hyper Parameters
maxiter = 5000
BATCH_SIZE = 64
LR_G = 0.001  # learning rate for generator
LR_D = 0.001  # learning rate for discriminator
N_IDEAS = 50  # think of this as number of ideas for generating an art work(Generator)
ART_COMPONENTS = 123  # it could be total point G can drew in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

path = './Bearing1_1/'
filename = 'Bearing1_1.csv'
#read_and_avg_all_csvfile(path)

artist_paintings = read_one_csv(filename,   BATCH_SIZE)
#归一化或者标准化
#std = StandardScaler()
#std_data = std.fit_transform(artist_paintings)
artist_paintings = torch.from_numpy(artist_paintings).float()
#new_im = Image.fromarray(artist_paintings)
#new_im.show()




'''
def artist_works():  # painting from the famous artist (real target)
    # a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    r = 0.02 * np.random.randn(1, ART_COMPONENTS)
    paintings = np.sin(PAINT_POINTS * np.pi) + r
    paintings = torch.from_numpy(paintings).float()
    return paintings
'''

G = nn.Sequential(  # Generator
   # nn.LSTM(N_IDEAS, N_IDEAS, 2),
    nn.Linear(N_IDEAS, 200),  # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(200,300),
    nn.ReLU(),
    nn.Linear(300, ART_COMPONENTS),  # making a painting from these random ideas
)

#G = nn.LSTM(BATCH_SIZE, ART_COMPONENTS, 50)


D = nn.Sequential(  # Discriminator
    #nn.LSTM(ART_COMPONENTS, ART_COMPONENTS, 2),
    nn.Linear(ART_COMPONENTS, 200),  # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(200,300),
    nn.ReLU(),
    nn.Linear(300, 1),
    nn.Sigmoid(),  # tell the probability that the art work is made by artist
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

#plt.ion()  # something about continuous plotting

D_loss_history = []
G_loss_history = []
for step in range(maxiter):
    # artist_paintings = artist_works()  # real painting from artist
    G_ideas = torch.rand(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)  # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)  # D try to increase this prob
    prob_artist1 = D(G_paintings)  # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    D_loss_history.append(D_loss)
    G_loss_history.append(G_loss)

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 1000 == 0:  # plotting
        print('step=', step)
        #print('D_Loss:', D_loss)
        print('G_Loss:', G_loss)
        #plt.cla()
        #plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        #plt.plot(PAINT_POINTS[0], np.sin(PAINT_POINTS[0] * np.pi), c='#74BCFF', lw=3, label='upper bound')
        #plt.text(-1, 0.75, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
        #         fontdict={'size': 13})
        #plt.text(-1, 0.5, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        #plt.ylim((-1, 1));
        #plt.legend(loc='upper right', fontsize=10);
        #plt.draw();
        #plt.pause(0.01)
print('g_loss_history', G_loss_history)

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






