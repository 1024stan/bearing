#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math

import numpy as np
from PyEMD import EEMD, EMD
import pylab as plt
import pandas as pd

def save_data_as_csv(file_name, save_data):
    save_temp_data = pd.DataFrame(data=save_data)
    save_temp_data_name = './qi/' + file_name + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')
    print('数据保存成功，存储地址：', save_temp_data_name)

def eemd(data):
    data = np.array(data)
    t = np.linspace(1, data.shape[0], data.shape[0])

    # Assign EEMD to `eemd` variable
    eemd = EEMD()

    # Execute EEMD on S
    eIMFs = eemd.eemd(data, t)
    nIMFs = eIMFs.shape[0]


    # # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs + 1, 1, 1)
    plt.plot(t, data, 'r')
    #
    for n in range(nIMFs):
        plt.subplot(nIMFs + 1, 1, n + 2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)
    #
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()

    return eIMFs, nIMFs


def emd(data):
    data = np.array(data)
    t = np.linspace(1, data.shape[0], data.shape[0])

    # # Say we want detect extrema using parabolic method
    emd = EMD()
    emd.extrema_detection = "parabol"
    # # Execute EMD on S
    eIMFs = emd.emd(data, t)
    nIMFs = eIMFs.shape[0]

    # # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs + 1, 1, 1)
    plt.plot(t, data, 'r')
    #
    for n in range(nIMFs):
        plt.subplot(nIMFs + 1, 1, n + 2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)
    #
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()

    return eIMFs, nIMFs


def ssa(data,windowLen=20):
    series = data
    series = series - np.mean(series)   # 中心化(非必须)

    # step1 嵌入
    windowLen = windowLen              # 嵌入窗口长度
    seriesLen = len(series)     # 序列长度
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = series[i:i + windowLen]

    # step2: svd分解， U和sigma已经按升序排序
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)

    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT

    # 重组
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen-1):
            for m in range(j+1):
                rec[i, j] += A[i, j-m] * U[m, i]
            rec[i, j] /= (j+1)
        for j in range(windowLen-1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j-m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j-seriesLen+windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)

    rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部

    plt.figure()
    for i in range(10):
        ax = plt.subplot(5,2,i+1)
        ax.plot(rec[i, :])

    plt.figure(2)
    plt.plot(series)
    plt.show()
    # rec是分解后的结果，重构时用 rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部
    return rec





if __name__ == "__main__":
    # data = readcsv()

    # 测试用的data
    # data = []
    # for i in range(100):
    #     data.append(math.sin(i))

    # 读取csv文件
    data = pd.read_csv('./qi/每日空调能耗.csv')
    data = np.array(data)[:, -1]
    y = data.reshape(1, -1)


    # emd 分解
    emdIMFs, n_emdIMFs = emd(data)
    emdIMFs = np.append(y, emdIMFs,axis=0)
    save_data_as_csv(file_name='emd分解结果', save_data=emdIMFs)
    # eemd分解
    eemdIMFs, n_eemdIMFs = eemd(data)
    eemdIMFs = np.append(y, eemdIMFs, axis=0)
    save_data_as_csv(file_name='eemd分解结果', save_data=eemdIMFs)
    # ssa分解
    rec = ssa(data)
    rec = np.append(y, rec, axis=0)
    save_data_as_csv(file_name='ssa分解结果', save_data=rec)



