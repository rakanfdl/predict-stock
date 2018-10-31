#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv 
import numpy as np
import matplotlib.pylot as plt
import sklearn.svm as SVR

dates = []
prices = []


def get_data(filename):
    with open(filename,'r') as csvfile:
        scvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
def   predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1))

    
    svr_len = SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color = 'black' , label = 'data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label ='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label ='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label ='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Sipport Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
get_data('AAPL.csv')

predicted_price = predict_price(dates, prices, 29)

print(predicted_price)


    




