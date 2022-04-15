import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from EEMD_LSTM_Baye import optimize_lstm, create_lstm_model
import math
from bayes_opt.util import Colours
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from main_eemd_mk import save_data_as_csv


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

def svc_cv_evalution(C, gamma, data, targets, kernel_type='rbf'):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, kernel=kernel_type, gamma=gamma, random_state=2)
    estimator.fit(data, targets)
    cval = accuracy_score(targets, estimator.predict(data))
    # cval = cross_val_score(estimator, data, targets, cv=4, scoring='accuracy')
    return cval.mean()


def optimize_svc_evalution(data, targets, kernel_type='rbf'):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svc_cv_evalution(C=C, gamma=gamma, data=data, targets=targets, kernel_type=kernel_type)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 3), "expGamma": (-4, -0.5)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
    return optimizer.max

def svm2evalution_optimize(filename, kernel_type = 'rbf'):

    '''
       SVM 评估算法
    '''

    ## 读取数据
    data = read_one_csv(filename)
    x = data[:, 0:data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]
    ## 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=41)  # random_state=0
    # 贝叶斯优化，选取最优的SVM参数值
    # best_hyperparameters = optimize_svc_evalution(x_train, y_train, kernel_type='rbf')
    best_hyperparameters = optimize_svc_evalution(x, y, kernel_type='rbf')

    expC = best_hyperparameters['params']['expC']
    expGamma = best_hyperparameters['params']['expGamma']
    C = 10 ** expC
    gamma = 10 ** expGamma

    ## 数据SVM分类器构建
    ## 高斯核
    clf = SVC(C=C, kernel=kernel_type, gamma=gamma)
    ## 评估模型训练
    print('开始训练')
    clf.fit(x_train, y_train)
    print('结束训练')
    print(clf.score(x_train, y_train))
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))

    return clf, C, gamma


def create_predite_dataset(filename, record_len=200, predite_len=1, x_dim=0, get_y_target=False):
    data = read_one_csv(filename)
    time = data.shape[0]
    temp_x = []
    temp_predite_y = []
    temp_target_y = []
    if get_y_target == False:
        x = data[:, x_dim]
        for j in range(time - predite_len - record_len):
            record_cut_index = j + record_len
            temp = x[j:record_cut_index].tolist()
            temp_x.extend(temp)
            predite_cut_index = j + record_len + predite_len
            temp_predite_y.extend(x[record_cut_index:predite_cut_index].tolist())
        x_pre = np.array(temp_x).reshape(-1, record_len)
        pre_y = np.asarray(temp_predite_y).reshape(-1, predite_len)
        return x_pre, pre_y
    else:
        y = data[:, data.shape[1] - 1]
        for j in range(time - predite_len - record_len):
            record_cut_index = j + record_len
            predite_cut_index = j + record_len + predite_len
            temp_target_y.extend(y[record_cut_index:predite_cut_index].tolist())

        y_target = np.asarray(temp_target_y).reshape(-1, predite_len)
        return y_target

def svr_predict(C, gamma, data, targets, kernel_type='rbf'):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVR(C=C, kernel=kernel_type, gamma=gamma)
    estimator.fit(data, targets.ravel())
    cval = r2_score(targets, estimator.predict(data))
    # # cval = estimator.score(data, targets)
    # trainPredict = estimator.predict(data)
    # trainScore = math.sqrt(mean_squared_error(targets, trainPredict))
    return cval.mean()


def optimize_svr_predict(data, targets, kernel_type='rbf'):
    """Apply Bayesian Optimization to SVC parameters."""
    def svc_crossval(expC, expGamma):
        """Wrapper of SVC cross validation.
        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        C = 10 ** expC
        gamma = 10 ** expGamma
        return svr_predict(C=C, gamma=gamma, data=data, targets=targets, kernel_type=kernel_type)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"expC": (-3, 3), "expGamma": (-4, -0.5)},
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)
    return optimizer.max


def predite_svm_optimize(x, y):
    ## 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)  # random_state=0
    # 贝叶斯优化，选取最优的SVM参数值
    best_hyperparameters = optimize_svr_predict(x, y, kernel_type='rbf')
    expC = best_hyperparameters['params']['expC']
    expGamma = best_hyperparameters['params']['expGamma']
    target = best_hyperparameters['target']
    C = 10 ** expC
    gamma = 10 ** expGamma

    ## 数据SVM分类器构建
    ## 高斯核
    clf = SVR(C=C, kernel=kernel_type, gamma=gamma)
    ## 评估模型训练
    print('开始训练')
    clf.fit(x_train, y_train.ravel())
    print('结束训练')
    print(clf.score(x_train, y_train))
    print('训练集准确率：', r2_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('测试集准确率：', r2_score(y_test, clf.predict(x_test)))

    return clf, C, gamma, target




def yujing(filename, evalution_svm):
    work_name = filename.split('/')[2].split('.')[0]
    string = 'SVM预测结果'
    work_name = './data/' + work_name + '_' + string + '.csv'
    data = read_one_csv(work_name)
    x = data[:, 0:data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]
    print(evalution_svm.score(x, y))
    predict_y = evalution_svm.predict(x)
    print('预测准确率：', accuracy_score(y, predict_y))
    y = np.append(y.reshape(-1, 1), predict_y.reshape(-1, 1), axis=1)
    save_data_as_csv(filename, y, string='预警结果')





if __name__ == '__main__':


    filename = './data/Bearing3_4_M-K检测.csv'
    kernel_type = 'rbf'
    nIMFs = 5
    record_len = 10
    evaluation_svm, C, gamma = svm2evalution_optimize(filename, kernel_type=kernel_type)
    print('评估模型的最优超参数：C=', C, ',gamma=', gamma)

    ## SVM 预测模型
    #
    predict_model = []
    hyperparameter_C = []
    hyperparameter_gamma = []
    model_target = []
    # predict_IMFs = []
    for i in range(nIMFs+1):
        print('**********训练第', i, '个特征向量的预测SVM***********')
        x_to_pre, predict_y = create_predite_dataset(filename, record_len=record_len, predite_len=1, x_dim=i, get_y_target=False)
        predict_y_model, C, gamma, target = predite_svm_optimize(x=x_to_pre, y=predict_y)
        predict_IMF = predict_y_model.predict(x_to_pre)
        predict_model.append(predict_y_model)
        hyperparameter_C.append(C)
        hyperparameter_gamma.append(gamma)
        model_target.append(target)
        # predict_IMFs.extend(predict_IMF)
        if i == 0:
            predict_IMFs = predict_IMF.reshape(-1, 1)
        else:
            predict_IMFs = np.append(predict_IMFs, predict_IMF.reshape(-1, 1), axis=1)

    y_target = create_predite_dataset(filename, record_len=record_len, predite_len=1, get_y_target=True)
    predict_IMFs = np.append(predict_IMFs, y_target, axis=1)
    save_data_as_csv(filename, predict_IMFs, string='SVM预测结果')
    hyperparameter_C = np.array(hyperparameter_C).reshape(-1, 1)
    hyperparameter_gamma = np.array(hyperparameter_gamma).reshape(-1, 1)
    model_target = np.array(model_target).reshape(-1, 1)
    model_target = np.append(model_target, hyperparameter_C, axis=1)
    model_target = np.append(model_target, hyperparameter_gamma, axis=1)
    # model_target = np.append(model_target, hyperparameter_momentum, axis=1)
    save_data_as_csv(filename, model_target, string='SVM贝叶斯优化过程')
    print('参数C：', hyperparameter_C)
    print('参数gamma：', hyperparameter_gamma)



    # # LSTM预测模型
    # #
    # nIMFs = 10
    # predict_model = []
    # hyperparameter_lr = []
    # hyperparameter_decay = []
    # hyperparameter_momentum = []
    # model_target = []
    # # predict_IMFs = []
    # for i in range(nIMFs+1):
    #     print('**********训练第', i, '个特征向量的预测LSTM***********')
    #     x_to_pre, predict_y = create_predite_dataset(filename, record_len=200, predite_len=1, x_dim=i,
    #                                                  get_y_target=False)
    #     x_to_pre = np.reshape(x_to_pre, (x_to_pre.shape[0], 1, x_to_pre.shape[1]))
    #     # predict_y = np.reshape(predict_y, (predict_y.shape[0], 1, predict_y.shape[1]))
    #     best_hyperparameters= optimize_lstm(x_to_pre, predict_y)
    #     lr = best_hyperparameters['params']['lr']
    #     decay = best_hyperparameters['params']['decay']
    #     momentum = best_hyperparameters['params']['momentum']
    #     target = best_hyperparameters['target']
    #     predict_y_model = create_lstm_model(input_shape=[1, x_to_pre.shape[2]],
    #                           layer=[200, 100, 50, 1],
    #                           output_shape=1, lr=lr, decay=decay, momentum=momentum)
    #     predict_IMF = predict_y_model.predict(x_to_pre)
    #     predict_model.append(predict_y_model)
    #     hyperparameter_lr.append(lr)
    #     hyperparameter_decay.append(decay)
    #     hyperparameter_momentum.append(momentum)
    #     model_target.append(target)
    #     # predict_IMFs.extend(predict_IMF)
    #     if i == 0:
    #         predict_IMFs = predict_IMF.reshape(-1, 1)
    #     else:
    #         predict_IMFs = np.append(predict_IMFs, predict_IMF.reshape(-1, 1), axis=1)
    #
    # y_target = create_predite_dataset(filename, record_len=200, predite_len=1, get_y_target=True)
    # predict_IMFs = np.append(predict_IMFs, y_target, axis=1)
    # save_data_as_csv(filename, predict_IMFs, string='LSTM预测结果')
    # hyperparameter_lr = np.array(hyperparameter_lr).reshape(-1, 1)
    # hyperparameter_decay = np.array(hyperparameter_decay).reshape(-1, 1)
    # hyperparameter_momentum = np.array(hyperparameter_momentum).reshape(-1, 1)
    # model_target = np.array(model_target).reshape(-1, 1)
    # model_target = np.append(model_target, hyperparameter_lr, axis=1)
    # model_target = np.append(model_target, hyperparameter_decay, axis=1)
    # model_target = np.append(model_target, hyperparameter_momentum, axis=1)
    # save_data_as_csv(filename, model_target, string='LSTM贝叶斯优化过程')
    # print('参数lr：', hyperparameter_lr)
    # print('参数deacy：', hyperparameter_decay)
    # print('参数momentum：', hyperparameter_momentum)


    # 检验算法性能？
    yujing(filename, evaluation_svm)









