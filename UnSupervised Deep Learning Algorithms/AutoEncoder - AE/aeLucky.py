# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:00:44 2020

@author: Lucky_Rathod
"""

#### Auto Encoders ####

'''
We will create a Recommendation system which will predict whether the user will like a movie or not

Movie Lens Dataset is used for building Recommendation system
https://grouplens.org/datasets/movielens/ 

It contains Different Dataset size 100k 24 million .We will download 100k and 1m 
1M dataset contains additonal information
'''

#### STEP 1 ####

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn         #For implementing Neural Network
import torch.nn.parallel      #For Parallel Computation
import torch.optim as optim   #For Optimizers
import torch.utils.data       #Tools
from torch.autograd import Variable #Stochastic Gradient Descent

# Importing Dataset
'''
We have title of movies and some movies has comma , So we cant keep comma as sperator
Header - None 
Engine - To make sure dataset gets imported correctly

These contains all the movies which are in the dataset
For Each Row 
Column 1- Movie ID
Column 2 - Movie Title
Column 3 - Movie Genre

Similarly we will do for users

Column 1 - User ID
Column 2 - Gender
Column 3 - Age
Column 4 - Code that corresponds to user job
Column 5 - Zip code

Similarly we will do for ratings

Column 1 - User 
Column 2 - Movies ID
Column 3 - Ratings 1-5 , 1 - Didnt Like , 5 - Liked
Column 4 - Timestamps
'''
movies = pd.read_csv('ml-1m/movies.dat',sep='::',header = None,engine = 'python',encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep='::',header = None,engine = 'python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header = None,engine = 'python',encoding='latin-1')

#### Step 2 ####

# Preparing the training set and the test set

'''
We will use ml-100k dataset which is splitted into 5 pair of sets
u1 base - Training Set , u1 test  - Test set

5 pair of Sets can be used to perform K-Fold cross validation

But we will only take One Train-Test split

We will have to convert these training set into array.

Column 1 - User
Column 2 - Movies
Column 3 - Ratings
Column 4 - Timestamps

Training set and Test set have different ratings , There is no common rating of same movie by same user
Same user 1 will not have same movie in Test set

Now we will train our autoencoders on training set and try to identify patterns to find some groups of movies 
that are liked by similar segment of users.It will find some specific features of movies which will be hidden nodes in autoencoders
and these specific features can be Actor,Director,Award

So based on these features , model will be able to predict the rating of the movie , that one user hasnt seen yet and it will be able to predict
that based on features that auto encoders detected and based on history of user

It will take the ratings given by user in history .So autoencoders will take features and histort both to predict 


'''

training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set = np.array(training_set,dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


#### Step 3 ####

'''
In next we will convert our training set and test set into matrix , where Rows - Users , Columns -Movies , Cells - Ratings
We will include all the users and movies from the dataset.
In Training dataset.If user didnt rate a movie we will put 0.
We will create 2 Matrices . These matrices will have same no of movies and same no of users.
And these matrices , each cell of user u and movie i - Each cell will get Rating of Movie i by user u.
If user u didnt rate a movie i ,  we will put 0.

Each of these matrices will contains , total no of user and movies

'''
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))


#### Step 4 ####

# Converting the data into an array with users in rows and movies in columns

'''
Since we will be using PyTorch . Therefore we will create list of list
List of Users - 943 List
And Each of 943 List will have list of 1682 Elements - Ratings
'''
def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        #new_data will be the rating given by id_user for each movie
        #Extract all the moviesid for which these user id have given rating 
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        #Create a list of 1682 zeros then replace zero by rating 
        ratings = np.zeros(nb_movies)
        #We will get indexes of ratings where we need to replace
        #Indexes start at 0 and id_movies start at 1
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
    
training_set = convert(training_set)
test_set = convert(test_set) 


#### Step 5 ####

#Converting data into Torch Sensors

'''
What are sensors ? - Tensors are arrays that contain elements of single data type
Tensor is multidimensional matrix.But instead of being numpy array it is Pytorch array
Training set will be going to be one torch sensor 
Test set will be going to be another torcch sensor

FloatTensor accepts List of List 

Variable Explorer in Spyder will disappear because it is not able to recognize Torch sensors
'''   
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


#### Step 6 ####

# Creating an architecture for Auto Encoders

'''
We will be using Inheritance . We will take Module class from nn , which is parent class .
From these parent class we will create child class through process of inheritance.

So that we can use all variable and functions from Parent class

SAE - Stacked AutoEncoders
'''
class SAE(nn.Module):
    def __init__(self,):
        '''
        super() - To get inherited methods from Module class
        
        Linear class of Module class - Which is used to make different full connection between layers
        '''
        super(SAE,self).__init__()
        '''
        First we will do full connection with input features vector i-e ratings of all movies for one spceific user
        And first hidden layer which is shorter vector than input vector
        
        Now we will create an object of class that is inherited from nn module class.
        And these object will represent full connection between first input vector features and first encoded vector
        
        
        Linear() 
        
        First input is number of features in input vector.One observation contains all ratings of all movies
        Well this number of input feature is nb_movies
        
        Second input is number of nodes or neurons in first hidden layer.That is no of elements in first encoded vector
        20 neurons on Hidden layer represents 20 features that autoencoder will detect.
        So these will represent features of movies that are liked by similar people
        
        Example - So one of the node of 20 can be specific genre of movie .One of detected feature will be Horror genre movie
        When a new user comes in system , then if this new user gave some good ratings to all the horror movies 
        Then this will activate this Horror Movie Genre movie neuron.
        And therefore big weights will be attributed to this neuron in final prediction to predict the rating of horror movie
        '''
        self.fc1 = nn.Linear(nb_movies,20) 
        self.fc2 = nn.Linear(20,10)  # This will detect even more features but based on previous features that were detected
        '''
        Now we will start to decode.To reconstruct our original input vector
        '''
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        '''
        Now we will specifiy activate function that will activate neuron when observation goes into the network
        Activation of Horror genre movie 
        '''
        self.activation = nn.Sigmoid()
    
    '''
    These function will do action that will take place in auto encoder 
    Action - Encoding and Decoding 
    
    These function will also apply  to differen activation functions inside full connections.
    
    Main purpose of making this function , it will return Vector of predictive ratings that we will compare to real ratings
    
    x - input vector
    
    We will encode these x Twice and then decoding Twice to get final output vector that is decoded vector that was reconstructed
    '''
    def forward(self,x):
        '''
        Encode input vector into first shorter vector composed of 20 features with help of activation object
        Because it will activate thir first Encoded vector of 20 Elements 
        
        So we apply activation on First full conncetion 
        
        self.activation(self.fc1(x)) - Represents encoded vector
        
        Similarly we will do for fc2
        '''
        x =  self.activation(self.fc1(x))
        x =  self.activation(self.fc2(x))
        
        #Here we are now decoding
        x =  self.activation(self.fc3(x))
        
        #Here we dont apply activation because it is final part .Therefore we will only use fc4
        x = self.fc4(x)
        
        # x will be vector of predicted ratings which we will compare with real ratings to measure loss
        return x
    
sae = SAE()

'''
criterion - For loss function - MSE
'''
criterion = nn.MSELoss() 
'''
Optimizer - To update weights in order to reduce the error at each epoch - Adam/rms prop

3 Arguments
All parameter of autoencoders
Learning Rate - 0.01
Decay - Decay is used to reduce learning rate after every epoch  in order to regulate convergence
'''
optimizer = optim.RMSprop(sae.parameters(),lr = 0.01,weight_decay=0.5)
        

#Training SAE

# Training the SAE
nb_epoch = 200
'''
Because in each epoch we loop through all our observations.All our users becausse each user correponds to ratings 
'''
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    #Normaline train loss by counter s - Divide the train loss by counter s
    s = 0.
    '''
    Loop though each user 
    
    It will introduce all the actions that will take place in One Epoch 
    We will get predictive ratings by SAE()
    We will compute loss error on one epoch. We will optimize these at each epoch
    We will apply our optimizer - Stochastic Gradient descent to update weights- RMSPROP
    
    '''
    for id_user in range(nb_users):
        '''
        We will start by getting the input vector.That contains all ratings of all movies given by these particular user
        
        First we will find these specific user from training set 
        
        But Training set id user is vector.And network in pytorch .Cannot accept single vector of one dimension
        But it can accept batch of input vectors.
        That is when we applied the different functions of network like forward().Function will not take simple vector
        of one dimsension as input.So we need to add one dimension as batch like in Cat dog predict()
        
        Similarly we will do in torch which will correspond to new dimesion - batch
        '''
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        
        '''
        IF - To save memory as much as possible
        
        Only look at users who have rated at leat one movie
        If observation contains only zeros,Which means user didnt rated movie,Then we dont care of this observation
        target.data will take all values of target -All ratings of this user
        
        If that sum > 0 means observation contains atleast one rated movie
        '''
        if torch.sum(target.data > 0) > 0:
            
            '''
            output will return vector of predicted ratings
            Forward() will automatically take place when we give input to sae()
            '''
            output = sae(input)
            '''
            These will make sure that we dont compute gradient with respect to target which will save lots of computations 
            '''
            target.require_grad = False
            #We will dont take movies that user didnt rate.Only for output vector - Vector of predicted ratings
            #These will dont have impact on updates of different weights
            output[target == 0] = 0 
            loss = criterion(output, target) 
            '''
            +1E-10 - TO make sure that denominator is not equal to zero
            '''
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            '''
            backward - Will check whether we need to inc or decrease weights
            '''
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
            '''
            Difference between backward and optimizer
            Backward decides directions to which weights will be updated - inc or decrease
            Optimizer decides intensity of updates - Amount by which weights will be updated
            '''
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    '''
    We need to keep training set because.Right now we are dealing with specific user .
    These id_user.So we will take input corresponding to that user.INPUT is all ratings given by user for all movies
    We will put these input vector inro AutoEncoder network.Auto Encoder will look at ratings of movies and especially positive ratings .
    And based on these ratings , it will predict ratings of movies that usser hasnt watched.
    
    If in out input vector,User gave 5 star rating to all action movies he watched.Then when we feed this input vector into network.
    Neurons corresponding to specific features related to action movies will be activated with large weight to predict high ratngs
    for other action movies that user hasnt watched
    
    And then we will compare these predictive ratings to real ratings of test set
    
    '''
    input = Variable(training_set[id_user]).unsqueeze(0)
    #target contains real ans
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[(target == 0).unsqueeze(0)] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

'''
train loss - 0.91
test loss - 0.95 

Id you are applying these recommended system for movie you gonna watch tonight 
And lets say after watching the movie you give rating 4 star

Our model will predict betweeen 3-5 star


'''











