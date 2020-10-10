# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:12:25 2020

@author: Lucky_Rathod
"""

'''
We are given a dataset that contains information of customers from this bank.
Applying for advanced credit  card.These data is what customers have to provide 
when filling the application form.

Our Mission is to detect potential fraud within these applications.
By the end of Mission ,  We have to give the explicit list of customers who 
potentially cheated

We will not use Supervised approach instead we will use Unsupervised Approach
and identify patterns in High Dimensional Dataset full of non linear relationship

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset - Staglog Australian Credit Approval Dataset
#Each Row indicates one customer - We will do segmentation of customer 
#And One of Segmentation will contain customers that are fraud 
dataset = pd.read_csv('Credit_Card_Applications.csv')

'''
First of all , All these customers are input of our neural network.
All these inputs points are going to be mapped to new output space.
Between i/p and o/p space , we have neural network composed of neurons.Each neuron
being initialized as vector of weights i-e Same size as vector of Customer
i-e Vector of 15 Elements . And for each of the customer  , Output of customer 
will be neuron that is closest customer.We pick the neuron which is closest to 
customer.

For each customer , BMU OR Winning node is more similar neuron to customer.
Then we use Gaussian Neighbour function to update weight of neighbour of winning node
to move them closer to the point.We will do these many times.For Each time we do 
our output space decreases


FRAUDS ARE BASICALLY OUTLIER OR OUTLYING NEURON.

How to detect outlier neuron in SOM ?
- For this we need MID(Mean Interneuron Distance) , i-e For each neuron ,We are 
going to compute Mean of Euclidean Distance between Neuron and Neuron in Neighbourhood

Then we will do inverse transform to identify which were the customers

'''
# 0 - Not Approved , 1 - Approved

X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Training SOM - Will use Minisom.py
from minisom import MiniSom
'''
We will try to identify patterns from Independent variables

Parameters for SOM
x,y - Dimensions of Self Organizing maps We dont have that much customer 10*10
input_len - Number of features we have in our dataset - X
sigma - Radius of different Neighbourhood in grid.Keep Default
learning-rate  - Decides by how much the weights are updated during each iteration
Higher Learning Rate - Faster will be convergence
Lower Learning Rate - Longer SOM will take time to built
decay_function - To improce convergence

'''
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)

#Before Training we have to initialize weights
som.random_weights_init(X)

#Training SOM
som.train_random(data = X,num_iteration = 100)

#Visualizing Results

'''
2D GRID  , That will contain all final winning nodes . And for each of these 
Winning nodes , we will get MID (Mean InterNeuron Distance) . MID of specific 
winning node is Mean of Distance of all neurons around that winning node inside a
neighbourhood that define (Sigma) i-e Radius of Neighborhood .

Higher MID - Winning node will be far away from its neighbours inside a neighbourhood
Higher MID - More the winning node is an outlier

For Each neuron , We will get MID.So we simply need to take winning nodes that have 
Highest MID.

Winning Nodes - Colored by Different Colors in such a way that the larger is MID,
CLOSER to white the color will be
'''

from pylab import bone,pcolor,colorbar,plot,show

#Figure Window  
bone()

#Put Winning nodes on Map.We are going to add info of MID for all winning nodes that SOM identified
# distance_map will return all MID in one matrix
pcolor(som.distance_map().T)

#We would like to see if White color corresponds to High MID or LOW MID 
#Same for Dark Colors - Whether they correspond to HIGH MID or LOW MID
colorbar()    

'''
These values are normalized values means values are scaled from zero to one

Highest MID is in WHITE COLOR 
Smallest MIF is in BLACK COLOR

FRAUDS are identified by Winning nodes,because outlying winning node are far from general rules
If all the neighboorhoods are close to each other then they will be darker.
If Winning nodes have large MIDS therfore they are outliers

'''

#We will add color markers to make distinction between customers who got approval (1) and not got approved(0)
#Because customers who got cheated and got approval are more relevant targets to fraud detection than the 
#customers who didnt got approval and cheated

'''
We will add markers to tell for each of these winning nodes (WHITE).If customers who are 
associated to these winning nodes got approval or didnt got approval
'''

#RED CIRCLE - Customers who didnt got approval
#GREEN CIRCLE - Customer who got approval

#o - circle , s - square markers
markers = ['o','s']
colors = ['r','g']

#Now we will loop through all customers and for each customer we are going to get Winning node
#and dependent on whether the customer got approval or not we are going to color this winning node
#Red Circle - If Didnt got approval Grren Square - Got Approval

#i - Different values of indexes 0-689 , x - Different vectors of customers i-e Whole row
for i,x in enumerate(X):
    #We extract the winning that customer got from SOM
    w = som.winner(x)
    #For these winning node , place marker on it Based on whether he got approval or not
    #Place marker at center of square of winning node.So for that you will have to find co-ordinates
    #w[0] - x coordinate , w[1] - y coordinate - These coordinates are of lower left of winning node
    #To put it in middle we will add 0.5
    
    #If y[i] = 0 - markers[0]-circle - red - Not Approved
    #If y[i] = 1 - markers[1] - square - green - Approved
    
    #We will add color at edge .
    #But then as for inside of marker , we will not color it ,because we can have two markers for same winning node
    
    #If thats case we will see much better the two markers if theres no color inside - markerfacecolor - inside color 
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor =  colors[y[i]],
         markerfacecolor = 'None',
         markersize=10,
         markeredgewidth = 2)
show() 
    
'''
Now we are able to see MID for all Winning nodes , but besides we are going to see if customers
associated to winning node are customers who got approval or didnt got approval


For Example 

Only RED - We can see that customers associated to these winning node didnt get approval
Only Green -  Customers associated to these winning node got an approval .But color is fine its around
0.7 - Which doesnt indicate high risk of fraud

Now lets look at outliers . Winning nodes with Highest MID . which clearly indicates that there is high risk of fraud
for the customers associated to that WHITE winning nodes

We can see that in these WHITE winning node there are both cases,Customers who got approval and who didnt got approval
Because we get a  green square and also red circle

So now we can catch these customers in winning nodes who were at HIGH RISK of FRAUD and got Approval
Because its of course much more relevant to bank to catch the cheaters who got away with this

Now with the help of this map we can catch the potential customers 

'''
    
# FINDING FRAUDS

#We will use method obtained in Minisom to obtain frauds list of customers
#mappings is dictionary . We get all the Winning nodes mappings
#key - coordinates of winnning nodes . And for that Winning node we get list of customers who are there in that winning node

mappings = som.win_map(X)

#Identify frauds from that WINNING NODE (WHITE) which has High MID

#These value will keep changing based on Highest MID
frauds = mappings[(3,4)]

#To identify Customer from more than 1 Winning nodes
frauds = np.concatenate((mappings[(3,4)],mappings[(5,3)]),axis = 0 )  #concatenate at Vertical axis
#Now we get whole list of cheaters

frauds = sc.inverse_transform(frauds)

'''
Now we will give the list of potential cheaters to Bank .Then Analyst will investigate further
and so first what he will probably do will find the values of y for all these customers.
Taking the priority the ones that got approved . So that revise Application and investigate further to 
fing whether the customer really cheated

'''










    
    
    
    
    
    
    
    
    
    
    














