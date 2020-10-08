# -*- coding: utf-8 -*-
"""
Created on Sat July 8 19:28:13 2020

@author: Lucky_Rathod
"""

# Convolution Neural Network


# Step 1 - Dataset(Data Preprocessing)
'''
Now in CNN we dont have I.V and D.V we have to make it by putting the images in the folders(train/test) 
But better solution is to make use of keras to import images -> We only need to ceate a special structure for dataset
Structure -> Dataset->1.Train->1.Cats 2. Dogs ->2.Test->1.Cats 2.Dogs

'''

#Feature Scaling is Compulsorty in Deep Learning

#Step2 -> Building CNN model

from keras.models import Sequential
from keras.layers import Conv2D       #For Adding Convolution Layers
from keras.layers import MaxPooling2D #Pooling Layers
from keras.layers import Flatten      #Flattening Layer -> Creates i/p for model
from keras.layers import Dense        #Hidden or Fully connected layers

    
#Initializing CNN
classifier = Sequential()

#Adding Convolution Layer
'''
No of Feature Detector (Filters) = No of Feature Maps

Convolution2D 

nb_filter    -> No of filters (Feature Detector) will be equal to Feature maps
                No of rows in Convolution Kernel(Filter or Feature Detector)
                No of column in Convolution Kernel
                
                Because Filter can be 3*3 matrix or any no of rows or columns
                32->Filter 3->Rows 3->Columns
                So there will 32 Feature maps

input_shape  -> Shape of an Input Image on which Filter will be applied
                All images dont have same format 
                we will have to force them to have same format
                
                Colorful image are converted to 3D Array
                
                (3,64,64) spyder theano
                (64,64,3) As per tensorflow
                3->No of channels for Colorful image
                1->No of channels for B/W image
                64 64 -> Dimension of 2D array in each channel

relu        -> To remove non Linearity from image
               Remove Black(Negative value)
                
 
''' 
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


#Adding the Pooling Layer
'''
Reducing the size of Feature map ->Max Pooling is used ->Reduce no of nodes as i/p for model
Slide the Featur map table with stride = 2 and find the max and fill it in Pooled Feature Map
Max Pooling size = Feature map / 2 

'''
classifier.add(MaxPooling2D(pool_size=(2,2)))




#To improve performance of model i added these Layer 
#We dont need to include input_shape when we add 2nd convolution layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Adding the Flattening Layer
'''
Take all featured maps and add it in One Vectore one by one
And these Vector will be the i/p for ANN

'''
classifier.add(Flatten())

#Adding the Fully Connected Layer
'''
ANN are best for Non Linear problems
And Also Image are Non Linear we can use classic ANN
units = No of neurons in layer u want

'''
classifier.add(Dense(units=128,activation='relu'))

#Adding the Ouptut Layer
'''
activation -> sigmoid (Two categories)
              softmax (More than Two Categories)

'''
classifier.add(Dense(units=1,activation='sigmoid'))


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


#PART 2 -> Fitting the CNN to the images
''' 
Image Augmentation - Preprocessing the images to prevent overfitting

Google Keras Documentation -> Image Preprocessing 

ImageDataGenerator -> First function to Generate Image Augmentation

When we dont have a Large Dataset we need to apply IMAGE AUGMENTATION to prevent overfitting

It creates many batches of our Images and on each batch it will apply random transformation on random selection of our image
Transformation like Rotation , Flipping , Shifting , Shearing
So due to these Image Augmentation we can have more images in dataset
So in these way it reduces Overfitting

flow_from_directory()
->Preprocessing (Augmentation)
->Fitting our model
->Not only Fit to training set but will also test at the same time on new observation (Test set)


Rescale -> Feature Scaling part in CNN model

'''
#Import class
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, #By these all our pixel values are between  0 and 1 -- Feature Engineering - Normalization
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


#Rescale the pixels of Images  of test set . So that they have value between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)

#We will include all images which are in train folder as well as the augmented images 
training_set = train_datagen.flow_from_directory(
        'dataset/training_set', #Directory where images to be train are
        target_size=(64, 64 ), #Size should be same as input_shape size 64,64 i-e Dimensions expected by our CNN model
        batch_size=32,         #No of images that will go through CNN after which weights will be updated 
                               #Batch of 32 images will be created and then our CNN will be trained on these images in all different batches
                               
        class_mode='binary')   #More than 2 -> Categorical


test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Training CNN on Training set and evaluating on Test Set

classifier.fit_generator(
        training_set,
        #At first it was samples_per_epoch = 8000
        steps_per_epoch=250,#No of images in Training set -> 1 epoch -> All observation must be passed through CNN
        epochs=25,
        validation_data=test_set,
        #At first it was nb_Val_samples=800
        validation_steps=2000) #No of images in test set

#We got accuracy of 75% on test set and 85% on train 
#We can Improve our accuracy ->Include another convolution layer or Add another fully connected layer
#Best solution is to Add convolution Layer'
#If we add convolution layer we also need to add max pooling layer

# Making a Single Prediction

import numpy as np
from keras.preprocessing import image

#Size of Image should be same as that of image which we trained on i-e 64*64
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size =(64,64))

#Now our Neural Network accepts 2d Array as input .So we need to convert it into 2d Array
# img_ro_array converts PIL image instance into array 
test_image = image.img_to_array(test_image)


'''
Now we need exact same format which we have done on Training .While Training we created Batches of Images
Therfore , Our CNN was not trained on Single images ,But it was trained on batch of images
Therefore we have extra dimension corresponding to batch

So Even if we want to predict for single image . That image should be in BATCH . So that CNN can reccognize that batch dimension
Now we all add fake dimensions i-e Dimesion corresponding to batch - np.expand_dims()

np.expand_dims --> 1st Arg - Input image  we will add batch dimesion
               --> 2nd Arg - Where we want to add Dimension.It is always first 
'''
test_image  = np.expand_dims(test_image,axis=0)

#Result
training_set.class_indices

result = classifier.predict(test_image)

#First we get access of Batch and there was only One - 0 , Then we get accesss to first and only element of batch
#Which corresponds to predictions of cat or dog

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)










