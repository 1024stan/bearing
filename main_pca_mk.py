import csv
import os
import pandas as pd
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 统计工件文件夹下采集数据的文件个数
def all_excel_num_in_dir(path):
    number = 0
    for root, dirname, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] =='.csv':
                number += 1
    return number


#读取预处理之后的汇总表
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

def save_as_csvfile(data, file_name,classname):
    save_temp_data = pd.DataFrame(data=data)
    file_name = file_name.split('/')[2]
    file_name = file_name.split('.')[0]
    save_temp_data_name = './data/'+ file_name +'_' + classname + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')


def mk(data):
    n = len(data)
    x = np.arange(1, n)
    y = data
    Sk = np.zeros(n)
    UFk = np.zeros(n)
    s = 0
    for i in range(1, n):
        for j in range(0, i):
            if y[i] > y[j]:
                s = s + 1
            else:
                s = s + 0
        Sk[i] = s
        # Exp_value = (i + 1) * (i + 2) / 4  # Sk[i]的均值
        # Var_value = ((i + 1) * i * (2 * (i + 1) + 5) / 72)  # Sk[i]的方差
        Exp_value = np.mean(Sk)
        Var_value = np.var(Sk)
        UFk[i] = ((Sk[i] - Exp_value) / np.sqrt(Var_value))

    y2 = np.zeros(n)
    Sk2 = np.zeros(n)
    UBK = np.zeros(n)
    s = 0
    for i in range(n):
        y2[i] = y[n-i-1]

    for i in range(1,n):
        for j in range(0,i):
            if y2[i] > y2[j]:
                s = s + 1
            else:
                s = s + 0
        Sk2[i] = s
        # Exp_value = i * (i -1) / 4 # Sk[i]的均值
        # Var_value = i*(i-1)*(2*i+5)/72  # Sk[i]的方差
        Exp_value = np.mean(Sk2)
        Var_value = np.var(Sk2)
        UBK[i] = ((Sk2[i] - Exp_value) / np.sqrt(Var_value))
    UBK2 = np.zeros(n)
    for i in range(n):
        UBK2[i] = UBK[n-i-1]

    UBK2 = abs(UBK2)

    return UFk, UBK2

def indexofMin(arr):
    minindex = 0
    currentindex = 1
    while currentindex < len(arr):
        if arr[currentindex] < arr[minindex]:
            minindex = currentindex
        currentindex += 1
    return minindex

if __name__ ==  '__main__':
    path = './Bearing3_4/'
    filename = './data/Bearing3_4.csv'
    # read_and_avg_all_csvfile(path)
    check_number = all_excel_num_in_dir(path)

    csv_data = read_one_csv(filename)
    pca = PCA(n_components=10)
    pca.fit(csv_data)
    pcaed_data = pca.transform(csv_data)
    # print(pcaed_data)
    save_as_csvfile(pcaed_data, filename, 'pca')

    '''
    找出变异点
    '''
    UFk, UBK2 = mk(csv_data[:, 0])
    temp = abs(UBK2 - UFk)
    index = indexofMin(temp)
    # plt.plot(UFk, 'r')
    # plt.plot(UBK2, 'b')
    # print(index)
    # plt.show()
    # 标记损坏状态，好的标记为1，坏的标记为-1
    right_works = np.ones(index)
    wrong_works = np.ones(pcaed_data.shape[0] - index)
    wrong_works = -wrong_works
    work = np.append(right_works, wrong_works, axis=0)
    work = work.reshape(work.shape[0], 1)
    data_of_svm_pinggu = np.append(pcaed_data, work, axis=1)
    save_as_csvfile(data_of_svm_pinggu, filename, 'svm_pinggu')

    print('找到异常点，状态指标标记结束')


