from sklearn.model_selection import train_test_split
from PyEMD import EEMD
from numpy import *
import PyEMD as pyemd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error, r2_score
import keras.optimizers as op
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization


import pylab as plt

def create_original_data():
    t = np.linspace(0, 1, 100)
    S = 3 * np.sin(4 * t) + 4 * np.cos(9 * t)
    return S, t


def eemd(data, times):
    # Assign EEMD to `eemd` variable
    eemd = EEMD()
    # Say we want detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    # Execute EEMD on S
    eIMFs = eemd.eemd(data, times)
    nIMFs = eIMFs.shape[0]
    return eIMFs, nIMFs

def create_dataset(original_data, times, num, test_size):
    x = np.zeros(shape=(int(times/num), num))
    y = np.zeros(shape=(int(times/num), 1))
    for i in range(int(times/num)):
        for j in range(num):
            x[i, j] = original_data[i+j]
        y[i] = original_data[i+num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    x_train = np.reshape(x_train,newshape=[x_train.shape[0],x_train.shape[1],1])
    #y_train = np.reshape(y_train,newshape=[y_train.shape[0],y_train.shape[1],1])
    x_test = np.reshape(x_test,newshape=[x_test.shape[0],x_test.shape[1],1])
    #y_test = np.reshape(y_test,newshape=[y_test.shape[0],y_test.shape[1],1])

    return x_train, x_test, y_train, y_test


def create_lstm_model(input_shape, layer, output_shape, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    model = Sequential()
    model.add(LSTM(layer[0], input_shape=input_shape, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(layer[1], return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(layer[2], return_sequences=False))
    model.add(Dense(output_shape))
    optimizer = op.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


def lstm_model_cv(x_train, y_train, lr=0.01, decay=1e-6, momentum=0.9):
    model = create_lstm_model(input_shape=[1, x_train.shape[2]],
                              layer=[200, 100, 50, 1],
                              output_shape=1, lr=lr, decay=decay, momentum=momentum)
    # score = cross_val_score(model, x_train, y_train, cv=4, scoring='accuracy')
    model.fit(x_train, y_train, batch_size=5, epochs=10)
    trainPredict = model.predict(x_train)
    # trainScore = r2_score(y_train, trainPredict)
    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    # testPredict = model.predict(x_test)
    # testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    # score = (trainScore + testScore)
    # print('得分为：', score)
    return -trainScore


def optimize_lstm(x_train, y_train):
    def lstm_crossval(lr, decay, momentum):
        return lstm_model_cv(x_train, y_train, lr=lr, decay=decay, momentum=momentum)

    optimizer = BayesianOptimization(
        f=lstm_crossval,
        pbounds={"lr": (0, 0.1), "decay": (0, 0.5), "momentum":(0, 0.5)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=1)

    print("Final result:", optimizer.max)
    best_hyperparameters = optimizer.max
    return best_hyperparameters





if __name__ == '__main__':
    data, times = create_original_data()
    eIMFs, nIMFs = eemd(data, times)


    x_train, x_test, y_train, y_test = create_dataset(eIMFs[0, :],times.shape[0],num=5, test_size=0.3)
    '''
    model = create_lstm_model(input_shape=[x_train.shape[1], 1],
                              layer=[200,100,50,1],
                              output_shape=1)
    model.fit(x_train, y_train, batch_size=1, epochs=10)
    trainPredict = model.predict(x_train)
    trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
    print('----训练得分为----\n', trainScore)
    testPredict = model.predict(x_test)
    testScore = math.sqrt(mean_squared_error(y_test, testPredict))
    print('----测试得分为----\n', testScore)
    '''
    best_hyperparameters = optimize_lstm(x_train, y_train)
    lr = best_hyperparameters['params']['lr']
    decay = best_hyperparameters['params']['decay']
    momentum = best_hyperparameters['params']['momentum']
    print(lr, decay, momentum)








