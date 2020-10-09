# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:21:52 2019

@author: Lucky_Rathod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Importing the training set --- Test Does not exist for RNN
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values 

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0) #Vertical axis is labeled by zero
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values #Nothing after ':' means last index of dataset
#We have not used the iloc to get i/p i-e All input lines must be ONE Column
inputs = inputs.reshape(-1,1)
#Scaling of i/p
#sc object is already fitted so no need to fit 
#only transfor
#Loading the saved model
'''
filename='sc.pkl'
sc = pickle.load(open(filename,'rb'))

'''
#Feature scaling - Normalization is used in RNN Instead of Standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # 0,1 because all the values will be between 0 and 1 
#Fit means it will get the min and max of data  -> Transform -> For each stock price it will calculate scaled price by formula
training_set_scaled = sc.fit_transform(training_set) 

inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#predicted_stock_price = regressor.predict(X_test)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)

from keras.models import load_model
# load model from single file
model = load_model('lstm_model.h5')
# make predictions
predicted_stock_price = model.predict(X_test, verbose=2)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()