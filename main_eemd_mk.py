import csv
import os
import pandas as pd
import seaborn as sns
from PyEMD import EEMD, EMD
import numpy as np
import pylab as plt
from main_pca_mk import mk, indexofMin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


def save_data_as_csv(file_name, save_data, string=''):
    save_temp_data = pd.DataFrame(data=save_data)
    work_name = file_name.split('/')[2].split('.')[0]
    temp = np.mat(save_temp_data)
    save_temp_data_name = './data/' + work_name + '_' + string + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')


def read_one_csv(filename):
    csv_data = []
    #data_batch_size = np.zeros([123, BATCH_SIZE])# 这里面的123需要更改为csv文件数，也就是分钟
    csv_data_file = filename
    with open(csv_data_file) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        data_header = next(csv_reader)
        for row in csv_reader:
            csv_data.append(row)
    csv_data = np.asarray(csv_data, 'float32')
    csv_data = np.delete(csv_data, 0, axis=1)
    return csv_data


def eemd(data):
    t = np.linspace(1, data.shape[0], data.shape[0])

    # Assign EEMD to `eemd` variable
    eemd = EEMD()

    # # Say we want detect extrema using parabolic method
    emd = EMD()
    emd.extrema_detection = "parabol"

    # Execute EEMD on S
    eIMFs = eemd.eemd(data, t)
    nIMFs = eIMFs.shape[0]

    # # Execute EMD on S
    # eIMFs = emd.emd(data, t)
    # nIMFs = eIMFs.shape[0]

    # # Plot results
    # plt.figure(figsize=(12, 9))
    # plt.subplot(nIMFs + 1, 1, 1)
    # plt.plot(t, data, 'r')
    #
    # for n in range(nIMFs):
    #     plt.subplot(nIMFs + 1, 1, n + 2)
    #     plt.plot(t, eIMFs[n], 'g')
    #     plt.ylabel("eIMF %i" % (n + 1))
    #     plt.locator_params(axis='y', nbins=5)
    #
    # plt.xlabel("Time [s]")
    # plt.tight_layout()
    # plt.savefig('eemd_example', dpi=120)
    # plt.show()

    return eIMFs, nIMFs

def draw_mk_point(data,eIMFs, nIMFs, mk_point):
    t = np.linspace(1, data.shape[0], data.shape[0])
    # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs + 1, 1, 1)
    plt.plot(t, data, 'r', zorder=1)
    plt.scatter(int(mk_point[0]), data[int(mk_point[0])], marker='o', zorder=2, c='r')
    for n in range(nIMFs):
        plt.subplot(nIMFs + 1, 1, n + 2)
        plt.plot(t, eIMFs[n], 'g', zorder=1)
        plt.scatter(int(mk_point[n]), eIMFs[n, int(mk_point[n])], marker='o', zorder=2, c='r')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()


if __name__ == '__main__':
    file_name = './data/Bearing3_1.csv'
    csv_data = read_one_csv(file_name)
    data = []
    data = csv_data[:, 0]
    eIMFs, nIMFs = eemd(data)
    mk_point = []
    data = data.reshape(1, -1)
    # normal_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    normal_data = data
    # 确定第一个突变点
    UFk, UBK2 = mk(normal_data[0])
    temp = abs(UBK2 - UFk)
    index = indexofMin(temp)
    mk_data = np.append(mk_point, index)
    normal_eIMFs = eIMFs
    # normal_eIMFs = StandardScaler().fit(eIMFs).transform(eIMFs)
    for i in range(nIMFs):
        UFk, UBK2 = mk(normal_eIMFs[i])
        temp = abs(UBK2 - UFk)
        index = indexofMin(temp)
        mk_point = np.append(mk_point, index)
    mk_point_one = mk_point[0]
    print('第一个突变点：', mk_point_one)
    print('mk_point的个数：', mk_point.shape)
    # draw_mk_point(data[0], eIMFs, nIMFs, mk_point)

    # 找第二个突变点
    wrong_data = normal_data[0, int(mk_point_one): normal_data.shape[1]-1]
    # print('计算第二突变点', wrong_data)
    UFk, UBK2 = mk(wrong_data)
    temp = abs(UBK2 - UFk)
    mk_point_two = indexofMin(temp)
    mk_point_two = mk_point_one + mk_point_two
    print('第二个突变点：', mk_point_two)

    # 打标签
    mk_target_of_data = []
    for i in range(data.shape[1]):
        if i <= mk_point_one:
            mk_target_of_data.append(0)
        elif i > mk_point_one  and  i <= mk_point_two:
            mk_target_of_data.append(1)
        else:
            mk_target_of_data.append(2)
    mk_target_of_data = np.asarray(mk_target_of_data)
    save_data = np.append(data.reshape(-1,1), eIMFs.reshape(-1, nIMFs), axis=1)
    # save_data = np.append(save_data, np.sum(eIMFs, axis=0).reshape(-1, 1), axis=1)
    save_data = np.append(save_data, mk_target_of_data.reshape(-1, 1), axis=1)
    save_data_as_csv(file_name, save_data, string='M-K检测')

    before_kmeans = data
    before_kmeans = np.append(before_kmeans, eIMFs, axis=0).reshape(-1, nIMFs+1)
    km = KMeans(3)
    km.fit(before_kmeans)
    after_kmeans = km.predict(before_kmeans)
    # sns.relplot(data=data, hue=after_kmeans)
    # print(after_kmeans)
    before_kmeans = np.append(before_kmeans, after_kmeans.reshape(-1, 1), axis=1)
    save_data_as_csv(file_name, before_kmeans, string='_Kmeans')










