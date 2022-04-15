#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Module, LSTM, Linear, Conv1d
from sklearn.model_selection import train_test_split
import pandas as pd
device = torch.device('cuda:0')
from torch.utils.data import DataLoader, TensorDataset


def split_series_data(data=None, history_len=10, predict_len=2):
    sample_number = data.shape[1] - history_len - predict_len
    x = []
    y = []
    for i in range(sample_number):
        x.append(data[:, i:i+history_len])  # 这里面，将历史数据也包含在x里面
        y.append(data[0, i+history_len:i+history_len+predict_len]) # 这里，y仅取实际数据

    x = np.array(x).reshape(sample_number, history_len, -1)
    y = np.array(y).reshape(sample_number, predict_len, 1)
    print('时序数据划分样本  结束')
    return x, y


class SEAttention(nn.Module):
    '''
    不用管这个
    '''
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, a, aa, aaa):
        # b, c, _ = x.size()
        # y = self.avg_pool(x)
        # y = x.view(b, _)
        y = self.fc(x)
        # y = y.view(b, c, 1)
        return x * y.expand_as(x)

class LSTM_Base_Net(Module):
    '''
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    '''
    def __init__(self, history_len,batch_size,use_bidirectional,y_size,
                 input_size, hidden_size,lstm_layers,dropout_rate,output_size,use_attention):
        '''

        :param history_times: 无用
        :param batch_size: 不需要改
        :param use_bidirectional: 是否双向
        :param input_size: 输入尺度，即特征维度
        :param hidden_size: 隐藏层数
        :param lstm_layers: lstm的层数
        :param dropout_rate: 防止过拟合的，不用管
        :param output_size: 输出尺度，即你要预测的尺度，默认为1，即一天
        :param use_attention: 是否用注意力
        '''
        super(LSTM_Base_Net, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True,
        #                  dropout=config.dropout_rate, bidirectional=False)
        self.use_attention = use_attention

        if use_bidirectional == True:
            self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=lstm_layers, batch_first=True,
                             dropout=dropout_rate, bidirectional=True)
            self.lstm_output_size = hidden_size * 2


        else:
            self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=lstm_layers, batch_first=True,
                             dropout=dropout_rate, bidirectional=False)
            self.lstm_output_size = hidden_size

        self.attention_layer = SEAttention(channel=self.lstm_output_size, reduction=8)
        self.linear_1 = Linear(in_features=self.lstm_output_size, out_features=output_size)
        self.linear_2 = Linear(in_features=history_len, out_features=y_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        if self.use_attention ==True:
            lstm_out = self.attention_layer(lstm_out, lstm_out, lstm_out, lstm_out)
        linear_1_out = self.linear_1(lstm_out)
        linear_1_out = linear_1_out.view(linear_1_out.shape[0], linear_1_out.shape[2], linear_1_out.shape[1])
        linear_out = self.linear_2(linear_1_out)
        return linear_out, hidden


def run_model(learning_rate=0.0001, max_epoch=800, do_continue_train=None,
              input_size=10, hidden_size=128,lstm_layers=4,dropout_rate=0.1,history_len=100,output_size=1,
              train_loader=None, test_loader=None, use_bidirectional=False,use_attention=True):
    '''

    :param look_back: 无用
    :param pre_len: 无用
    :param learning_rate: 学习率
    :param max_epoch: 最大迭代次数
    :param batch_size: batch size，在数据loader中就已经设置，这里不用管
    :param x_train: 训练x
    :param x_test: 测试x
    :param y_train: 训练y
    :param y_test: 测试y
    :param do_continue_train:无用
    :param train_loader: 训练集
    :param test_loader: 测试集
    :param use_bidirectional:是否双向  ！！！ 重要！！！
    :param use_attention: 是否注意力  ！！！ 重要！！！
    :return: 模型的测试误差，可修改，后续沟通
    '''

    # 创建模型
    model=LSTM_Base_Net(history_len=history_len,output_size=output_size,y_size=1,
                        batch_size=2,use_bidirectional=use_bidirectional,use_attention=use_attention,
                        input_size=input_size, hidden_size=hidden_size,lstm_layers=lstm_layers,
                        dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()      # 这两句是定义优化器和loss
    train_cost=[]
    # 训练
    global_step = 0
    for epoch in range(max_epoch):
        print("Epoch {}/{}".format(epoch + 1, max_epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []
        hidden_train = None
        # pred_train_Y_list = []
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            if _train_X.shape[0] < batch_size:
                break
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            pred_Y, hidden_train = model(_train_X, hidden_train)    # 这里走的就是前向计算forward函数

            if not do_continue_train:
                hidden_train = None             # 如果非连续训练，把hidden重置即可
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()    # 去掉梯度信息
                hidden_train = (h_0, c_0)
            loss = criterion(pred_Y, _train_Y)  # 计算loss
            # pred_train_Y_list.append(pred_Y)
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
        train_cost.extend(train_loss_array)

    # 测试
    print('开始测试')
    test_cost = []
    model.eval()
    hidden_predict = None
    result = torch.Tensor().to(device)
    gt = torch.Tensor().to(device)
    for _data in test_loader:
        data_X ,data_Y= _data[0].to(device), _data[1].to(device)
        if data_X.shape[0] < batch_size:
            break
        pred_X, hidden_predict = model(data_X, hidden_predict)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)
        gt = torch.cat((gt,data_Y), dim=0)
    model_score = torch.mean(torch.sub(result, gt))

    return model_score.detach().cpu()




if __name__ == "__main__":
    # data = readcsv()

    # 测试用的data

    # x = np.random.random((100,100,10)) # 样本数 * 时间维度 * 数据维度
    # y = np.random.random((100,2,1))

    # 读取分解后的数据，此处以eemd为例
    data = pd.read_csv('./qi/eemd分解结果.csv')
    data = np.array(data)
    history_len= 10
    predict_len= 2
    batch_size = 2
    x, y = split_series_data(data=data, history_len=history_len, predict_len=predict_len)
    feature_len = x.shape[-1]
    # 制作数据集
    train_X, valid_X, train_Y, valid_Y = train_test_split(x, y, test_size=0.3, random_state=12)
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=batch_size)
    # lstm模型
    model_score = run_model(train_loader=train_loader, test_loader=valid_loader,
                            history_len=history_len, output_size=predict_len, input_size=feature_len,
                            max_epoch=1,use_bidirectional=False,use_attention=True)
    print(model_score)


