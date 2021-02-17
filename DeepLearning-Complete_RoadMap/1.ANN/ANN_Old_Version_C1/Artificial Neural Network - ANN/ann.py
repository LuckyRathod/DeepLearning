# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#Business probelem description is provided in Readme file 
#Row number,Customer Id,Surname have no impact on Dependent Variable
#But we also dont know which Independent variable have High impact on Dependent Variable
#Thats why Artificial Neural Network comes into the future
#ANN will give high wieghts to that independent variable which has high impact
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#We have categorical Variable we need to encoded that
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#No of categorical columns = No of object we need to make
#For country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#For Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#There is no relational order between categories of categorical variables -> OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() #No need for doing it for Gender as it is cateorized to 2
#Total 5 dummy variable for Category 
#Remove 1 for dummy variable trap
X = X[:, 1:]#We will include all rows and colmns from Second Column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling Compulsory in deep learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import theano
import tensorflow
#Importing keras Libraries and Packages
import keras
#Sequential module to initialize Neural Network
from keras.models import Sequential

#Dense module to build layers of ANN
from keras.layers import Dense

#Initializing the Ann
'''Defining the Sequence of Layers OR Defining a Graph --> We will do Initializing by Layers concept'''
classifier = Sequential()

#Adding the input layer and the First Hidden Layer
'''
If your data is Linearly Seperable you dont need Hidden Layer i-e No Neural Network
Dense -> Used for adding Hidden Layers

->  output-dim->NO of nodes in Hidden Layer

    No of Node in Hidden Layer = Average Number of Nodes in Input Layer and Number of Nodes in Output Layer 
    (11+1(Node for Output)/2=6)  6 Nodes in Hidden Layer -> output dim=6
            or
    Parameter Tuning(k for Cross Validation) to find Optimal no of nodes in hidden layer 

->  init -> Step1 of Stochastic Gradient Descent i-e Initialize weights close to zero -> uniform 
->  actication
->  input_dim -> No of Nodes in input Layer (No of Independent Variables) -> when we add some other Hidden Layers we dont need these
'''
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#Adding the Second Hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#Adding the output Layer
'''
output_dim=1 because there is only Node for Output 

If you have Three Categories then you have to change
output_dim = No of classes  
output_dim = 3 If there are Categories
activation = softmax (For more then 2 categories)

'''
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


#Compiling ANN
'''
Applying Stochastic Gradient Descent to ANN

optimizer -> Algorithm you want to use to find the optimal  set of weights (Because till no weights are only Initialized)
             Algorithm - Stochastic Gradient Descent -> Adam
loss      -> Adam is based on Lost function to find optimal no of weights ->Lost ->Sum of square difference -> Linear Regression        
          -> Logarithmic class
             Binary outcome      ->binary_cross_entropy
             More then 2 outcome ->categorical_cross_entropy
metrics   -> Used to evaluate a model for better performance -> Inc Accuracy ->Accepts as list 

'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit ANN to training set  
'''
batch_size -> Update the weights after each observation
                    OR
              Update the weights after batch of operation
epoch      -> When whole training set passed through ANN

'''
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)  
#Accuracy on training set-> 84%

#Part 3 -> Making the predictions and Evaluating Model
# Predicting the Test set results
y_pred = classifier.predict(X_test)

#We will apply thresold so that above that value will be 1 (Leaved the organization)
y_pred = (y_pred > 0.5) #return true or false
#And below that value will be 0 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy on test set -> 84%
