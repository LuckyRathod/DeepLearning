# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:11:57 2019

@author: Lucky_Rathod
"""
'''

RNN -> We will be predicting Stock price of the google
We will predict upward and downward trends that exist in Google stock price which will be done through LSTM
LSTM model will be trained on FIVE years of Google stock price 2012 - 2016 
We will try to predict FIRST month of 2017
will only use 2 columns Date and Open Google Stock price
Training -> Stock price of all Days from 2012 - 2016
Test     -> Stock price of all Days of Jan 2017
Days to be Considered are Financial days [ Not SAT SUN]
     
'''
#Part 1 -> Data Preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Importing the training set --- Test Does not exist for RNN
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#We will not take dates , Also we cant enter only one index, Because we dont want simple vector , we want a numpy array
training_set = dataset_train.iloc[:,1:2].values  #Numpy array of One column

#Feature scaling - Normalization is used in RNN Instead of Standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # 0,1 because all the values will be between 0 and 1 
#Fit means it will get the min and max of data  -> Transform -> For each stock price it will calculate scaled price by formula
training_set_scaled = sc.fit_transform(training_set) 
#sc = sc.fit_transform(training_set) 

'''Object Serialization'''
#Save the model to disk
filename = 'sc.pkl'
pickle.dump(sc,open(filename,'wb'))



#Creating a Specific Data structure  -> NO OF TIME STEPS -> 60 TIME STEPS AND 1 OUTPUT

#Data structure specifying what the RNN will need to remember when predicting next Stock price
'''
It means that at each time 't' RNN is going to look at 60 stock prices before time 't' i-e 60 days before time 't'
And based on the trends it is capturing during 60 time steps it will try to predict the next Stock
For Each Financial day X_train will contain 60 previous Stock prices before the Financial day
Y_train will contain stock price of next Financial day

'''
X_train = []
y_train = []
#At each time 't' take 60 previous stock prices
#So loop will start from 61 because it will require 60 previous stock prices . But it start with 0 therefore 60 
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i , 0]) #0 -> Column no
    #   Output y_train will contain value at  61th stock price i-e 60  --> 0 - 59 will be learned by Rnn which will predict value at DAY 60
    y_train.append(training_set_scaled[i , 0])
#X_train and y_train are list so we need to convert it into Numpy array to get accepted by RNN
X_train,y_train = np.array(X_train) , np.array(y_train)

#60 will be your 1st row and columns will be 0-59 days stock price value 
    
    
    
#Reshaping 

#Adding more Dimensionality to previous Data structure we made X_train
#Dimension that we are going to add is units
# i-e  No of predictors we can use to predict what we want --> Predictors are Indicators 
'''
Till now we had only One  Indicator ->OPEN
By these New Dimension we are able to add more Indicators which can be used for prediction

Anytime you want to add DIMENSION in Numpy array use reshape()

reshape() -> Two Arguments

1st Argument - Numpy Array that we want to reshape
2nd Argument -  New Shape

In Paranthesis we will add new Dimensions (batch_size,timesteps,input_dim)
batch_size - Total Number of Observations
timesteps - No of time steps - New Data structure we created x_train 
input_dim - New Financial indicators that could help predictions i-e Closed stock price or stock price from other countries
From Other Companies means - Apple and Samsung - Samsung provides many materials .
Apple is Highle dependent on Samsung
Therefore stock price of Apple and Samsung are highly correlated

But we have only one indicator now
'''
#Till now  X_train as 2 Dimensions
#X_train.shape[0] -> Rows 
#X_train.shape[1] -> Columns
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1], 1))  



#Part 2 -> Building RNN

#Importing keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising RNN as Sequence of Layers Later we will make Computational Graph with PyTorch
regressor = Sequential()  #Regressor because at these time we are predicting continous values

#Adding First LSTM Layer and some Dropout Regularisation to prevent overfitting
'''
units -> No of units ->LSTM memory units -> Neurons that you want in these LSTM Layer -> 50 NEURONS in First Layer 
return_sequences     ->True because we are building STACKED LSTM which will have several layers
                       If you are adding LSTM layer after this layer then true
                       If you are not going to add LSTM layer after these then False
input_Shape          ->Shape of the input (batch_size,timesteps,indicators)
                       But in LSTM input(timesteps,indicators)


''' 
regressor.add(LSTM(units=50,return_sequences=True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2)) #Recommended to drop 20 percent of neurons 20% of 50 -> 10 neurons

#Adding Second LSTM Layer and some Dropout Regularisation to prevent overfitting
#input_shape is only required at FIRST LSTM layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2)) 

#Adding Third LSTM Layer and some Dropout Regularisation to prevent overfitting
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2)) 

#Adding Fourth LSTM Layer and some Dropout Regularisation to prevent overfitting
regressor.add(LSTM(units=50,return_sequences=False)) #False because after these we dont add Another LSTM layer
regressor.add(Dropout(0.2)) 

#Adding the output layer
#Output layer is fully connected to previous LSTM layer
#units ->  no of neurons in output layer
regressor.add(Dense(units=1))  

#Compiling RNN
'''
optimizer = RMSprop is Recommended But we will be using adam
'''
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error') #Regression problem -> Mean squared error 

#Fitting RNN to training set
'''
input         -> X_train
ground_truth  -> y_train
epochs        -> No of iterations of whole training set


'''
regressor.fit(X_train,y_train,epochs=100,batch_size=32,verbose=2)

#Save the Model . So that you can use that later
regressor.save('lstm_model.h5')



#### Load Moedel

from keras.models import load_model
# load model from single file
regressor = load_model('lstm_model.h5')




#Part 3 -> Making the predictions and Visualizing the results

#Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values 
  
#Getting the predicted stock price of 2017
'''
To predict stock price of each Financial day of JAN 2017 
We need 60 previous stock prices of previous financial days before the actual day
Now to do these we need both training set and test set
So we need to do the concatenation of training and test set to be able to get these 60 previous i/p for each day of 2017
If we directly concatenate both training set and test set it will leads us to the problem then we have to scale these concatenation
But  if we do scaling then we will change the actual test values and we should never do these
We have to keep the actual test values as it is
So we will concatenate original dataframes 'dataset_trained' and 'dataset_test'
From these we will get inputs for these prediction  i-e 60 previous stock prices
And at last these is what we will scale 
'''
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0) #Vertical axis is labeled by zero
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values #Nothing after ':' means last index of dataset
#We have not used the iloc to get i/p i-e All input lines must be ONE Column
inputs = inputs.reshape(-1,1)
#Scaling of i/p
#sc object is already fitted so no need to fit 
#only transform
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Evaluating Rnn
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))   


