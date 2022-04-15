from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)

import numpy as pd

x = boston.data
y = boston.target
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 从sklearn.preprocessing 导入数据标准化模块
from sklearn.preprocessing import StandardScaler

ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

from sklearn.svm import SVR

linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train.ravel())
linear_svr_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train.ravel())
poly_svr_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train.ravel())
rbf_svr_predict = rbf_svr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('The value of default measurement of linear SVR is', linear_svr.score(x_test, y_test))
print('R-squared value of linear SVR is', r2_score(y_test, linear_svr_predict))
print('The mean squared error of linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))
print('The mean absolute error of linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))

print('\nThe value of default measurement of poly SVR is', poly_svr.score(x_test, y_test))
print('R-squared value of poly SVR is', r2_score(y_test, poly_svr_predict))
print('The mean squared error of poly SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))
print('The mean absolute error of poly SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))

print('\nThe value of default measurement of rbf SVR is', rbf_svr.score(x_test, y_test))
print('R-squared value of rbf SVR is', r2_score(y_test, rbf_svr_predict))
print('The mean squared error of rbf SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))
print('The mean absolute error of rbf SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))