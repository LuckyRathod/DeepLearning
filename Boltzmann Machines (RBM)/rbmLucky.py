# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:15:22 2020

@author: Lucky_Rathod
"""

#### BOLTZMANN MACHINES ####

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

#Converting the rating into binary ratings 1(Liked) or 0(Not Liked)

'''
Restricted Boltzmann machine will predict whether the user will like a movie or not
Now we will convert ratings into Liked or Not Liked
Will Replace 0 with -1
or operator doesnt work with pytorch sensors
'''
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#### Step 7 ####

#Creating Architecture of Neural Network 

'''
We will create a class that will define how the RBM should be build 
In these RBM class we will give all info that we need to build RBM.Info such as Hidden nodes.
Weights for probability of visible nodes.Then bias for the same probability.Also bias for visible node

3 Function will be made in class
Initialize RBM 
SAMPLE H - Sample the probability of hidden nodes given the visible nodes
SAMPLE V - Sample the probability of visible nodes given the hidden nodes

'''

class RBM():
    
    '''
    Default Function 
    Arguments - Self corresponds to object that will be created afterwards 
    nv - number of visible nodes
    nh - number of hidden nodes
    
    '''
    def __init__(self,nv,nh):
        
        '''
        Initialize the parameters for future object - Objects that will be created from these class
        Since these parameters are specific to RBM model , Future objects that we are going to create
        These variables are varaibles of object therefore to initialize these we need to start with self
        W -  All parameters of the probabilities of visible nodes given hidden nodes
        All weights will be iniialized randomly with normal distribution (mean 0 , variance 1)  for matrix of nh and nv
        '''
        self.W = torch.randn(nh,nv) #Initialize tensor of nh,nv
        '''
        Bias for probability of hidden node given the visible node
        
        There will be one bias for eah hidden node and we have nh hidden node
        We will have to create vector of H element 
        First parameter will correpond to batch,Second parameter correspond to bias
        '''
        self.a = torch.randn(1,nh) #Creates 2d tensor
        '''
         Bias for probability of visible node given the hidden node
         '''
        self.b = torch.randn(1,nv)
    
    '''
    Sampling the hidden node according to the probabilities ph given v h-hidden node,v-visible node
    ph given v is Sigmoid activation function
    
    To apply GIBBS Sampling we need to compute probabilites of hidden nodes given visible nodes .And once we have these 
    probability we can sample activation of hidden nodes
    
    Will return samples of different hidden nodes of RBM
    
    Suppose we 100 HIDDEN Nodes , These function will activate them according to certain probability.
    And for each of the hidden node , probability is ph given v 
    i-e Probability that this hidden node = 1 , given v 
    
    x correponds to visible neurons v
    '''
    def sample_h(self,x):
        
        '''
        First we will compute probability of h given v which is equal to Sigmoid activation function
        These sigmoid activation funcction is applied to Wx + a 
        
        mm - to make product of two torch sensors
        
        Each input vector will not be treated indivually , it will be treated in batches
        expand_as - will exoand the bias - Will make sure that bias is applied to each mini batch
        '''
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx) 
        
        '''
        It is probability that hidden node is activated given the value of visible node
        If we have a user that only likes Dramatic movies ,  If there is hidden node that detected a specific feature corresponding 
        to that drama Genre . Well for these user who has high ratings for Drama movies.
        Probability of that node that is specific to Drama . Given the visible node of user that has all nodes of drama movies is equal to one .
        Its probability will be HIGH
        V = 1 FOR DRAMA MOVIES , H corresponds to Drama movie GENRE.So ph given v will be very HIGH
        
        Now we have to return the probability and sample h
        Will return some samples of neurons according to that probability
        We will return some Bernollu(Binary)samples of that distribution
        
        ith element of vector is probability that the hidden node is activated 
        So if the probability is below 70% , we will activate neuron
        
        zero corresponds to hidden nodes that were not activated after sampling
        one corresponds to nodes that were activated
        
        These Function will return all the probabilities of hidden neurons given values of visible nodes.
        Will also return sampling of hidden neurons
        '''
        p_h_given_v = torch.sigmoid(activation)
        
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    '''
    Will do same for visible nodes
    '''
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    '''
    Contrastive Divergence that we will use to approximate the likelihood Gradient
    For Any Deep Learning model ,To minimize Energy or maximize Log Likelihood , we need to compute gradient.
    
    v0 - Input vector containing ratings of all movies by one user.
    vk - visible nodes obtained after k samplings or k iterations
    ph0 - Vector of probabilities that at first iteration the hidden node =1 given values of v0
    phk - Probabilities of hidden nodes after k sampling given the values of visible nodes VK
    
    '''
    def train(self,v0,vk,ph0,phk):
        '''
        We will first update tensor of weights then bias b and then bias a 
        
        Product of prob that hidden node =1 given input vector v0 - ph0
        '''
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        
        
#### STEP 11 ####    
    
#Create an object of RBM class - Only parameters of init will be created

'''
nv - fixed parameter - Number of movies (Number of visible nodes) - Ratings of all the movies by a specific user
So we have one visible node for each movie

nh - Number of hidden nodes - Features that are going to be detected by RBM model.
We can have Actor,Director,Genre,Oscars etc
'''
nv = len(training_set[0]) #number of movies
nh = 100 
batch_size = 100

rbm = RBM(nv,nh)
    
#### Step 12 ####

# Training RBM
'''
In Each epoch , all our observations will go into the network and we will update weights after observation of each batch passed
In End ,we will get Final Visible Node with new ratings for movies that were not originally rated
'''
nb_epochs = 10

for epoch in range(1,nb_epochs+1):
    '''
    We need loss function to measure error between predictions and real ratings
    In Training , we will compare the predictions to the ratings we already have that is ratings of training set
    
    Measure the difference between predicted ratings and real ratings
    ''' 
    train_loss = 0
    #Normaline train loss by counter s - Divide the train loss by counter s
    s = 0. 
    
    '''
    Training happends with three function sample_h,sample_v,train.These functions were regarding to one use.
    But training has to be done on all users . But in Batch
    
    First we will get batches of users - Batch size - 100
    First Batch - 0 -99 , Second Batch - 100 -199
    '''
    for id_user in range(0,nb_users - batch_size,batch_size): 
        
        '''
        Input is the vector that will go into Gibbs chain and will be updated at each round trip
        At first vk is input batch of all observations in batch . Ratings that already existed
        
        ID_USER:ID_USER + BATCH_Size - Range from curresnt id user to 100 next users in training set
        '''
        vk = training_set[id_user:id_user+batch_size]
        '''
        Target v0 - Ratings of movies that are already rated by 100 users in batch
        '''
        v0 = training_set[id_user:id_user+batch_size]
        
        '''
        Initial Probabilities 
        ph0 -Probability of hidden node at start = 1 given rating
        Given rating of movies , that were alreadt rated by users of our batch
        
        sample_h() also return bernoulli . But we only want first return arg .So add ,_
        '''
        ph0,_ = rbm.sample_h(v0)
        
        '''
        For loop for k steps of contrastive Divergence - GIBBS CHAIN
        
        Gibbs chain consist of making Gibbs chain . There are simply several round trips from visible nodes to hidden nodes
        and then from hidden nodes to visible nodes
        
        And in each round trip of gibbs chain , Visible Nodes are updates.
        
        And step after step we get closer to good predictive ratings
        '''
        for k in range(10):
            '''
            Start with input batch of observations.Input ratings v0 - Batch of 100 users - Ratings for all movies for all 100 users
            
            From these input we are going to sample , First Hidden nodes -Using Bernoulli sampling  - ph0 given v0 - 
            Which happens in simple_h() on visible node
            
            sample_h() on v0 will give sampling of hidden node ph0
            '''
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            
            '''
            Here we will get our first update of visible node i-e Visible node after 1st sampling
            Once we have vk and hk at last we can now approximate the gradients
            
            
            We will not include cell that have -1 ratings - FREEZE
            '''
            vk[v0 < 0] = v0[v0 < 0]
            
        #Now we will train() to update weights and bias - Now for train() we also need phk
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        '''
        Now training is going to happen , Weigths and Bias will be updated towards direcction of maximum likelihood
        Now we will update train loss . Because now we will have predictions
        
        After all observations go into network , We will compute all our hk anf vk 
        and eventually get our last sample of visible nodes of Gibbs chain but with new weight that were updated at previous iteration
        So step by step we get our final vk with better weights
        
        And at last step we get weights closer to optimal weights
        
        Compare vk to v0 , Ratings that are existing
        '''
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1. 
    print('epoch: '+ str(epoch)+' loss: '+str(train_loss/s))
    

#### Step 14 ####

test_loss = 0
s = 0.
for id_user in range(nb_users):
    '''
    vt - target
    v - Input on which we will make prediction
    So we are trying to predict ratings in test set , So can we replace training set in v and vt to test_set ?
    
    vt - Contains orginal ratings of test set - That is we will compare to our prediction in end .So that why we will replace to test set
    
    We cant replace training set of vt to testset because Training set will be used to activate the hidden neurons to get the output
    Right now training set contains Ratings of training set and it doesnt contain answers of test set.
    But by using the inputs of training set we will activate the neurons of our RBM to predict the ratings of movies that were not rated yet i-e Ratings of Test Set
    
    So we need these as input to get predicted ratings of test set.Because we are getting these predicted ratings from the inputs of training set that are used to activate neurons
    of RBM to predict the ratings of our test set
    
    ''' 
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    
    '''
    To get our predictions of test set ratings , Do we need to apply Contrastive divergence ?
    Do we need to make k steps of random walk i-e 10 steps of random walk or Do we need to make one step of random walk
    
    We need to only make one step . Because principal of random walk is
    
    Imagine you are blind folded and you have to make 100 steps on straight line , without getting out of straight line
    You will be trained with GIBS SAMPLING to make 100 steps by staying on straight line but you are blind folded .
    So its not easy to make some steps .So sometimes you wil go little bit left and right .
    But your were trained to these 100 steps staying on straight line being blindfolded
    
    So when you make one step , when you have to take challenge to make only one step and still be on straight line 
    You will have high chance of doing it because you were trained on 100 steps
    '''
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))        
        
'''

If you want to check that 0.25 corresponds to 75% of success, you can run the following test:

'''

import numpy as np
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v)) # -> you get 0.25

#Evaluating Boltzmann Machine

# RMSE

## Training

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))

'''
Using the RMSE, our RBM would obtain an error around 0.46. But be careful,
although it looks similar, one must not confuse the RMSE and the Average Distance.
A RMSE of 0.46 doesn’t mean that the average distance between the prediction and
the ground truth is 0.46. In random mode we would end up with a RMSE around 0.72.
An error of 0.46 corresponds to 75% of successful prediction.

'''





         
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    















