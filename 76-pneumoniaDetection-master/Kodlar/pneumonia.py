#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2


# In[2]:


directory = os.getcwd()
path = os.path.join(directory ,'chest_xray')


# In[7]:


trainPath = os.path.join(path ,'train')
testPath = os.path.join(path ,'test')


# In[8]:


categories = 'PNEUMONIA NORMAL'.split()


# In[3]:


imgSize = 70


# In[33]:


trainSet = []
for category in categories:
    tempTrainPath = os.path.join(trainPath,category)
    for img in os.listdir(tempTrainPath):
        try:
            tempTrainImage = cv2.imread(os.path.join(tempTrainPath,img),cv2.IMREAD_GRAYSCALE)
            tempCropedTrainImage = cv2.resize(tempTrainImage,(imgSize,imgSize))
            trainSet.append([tempCropedTrainImage,categories.index(category)])
        except Exception as e:
            continue


# In[40]:


testSet = []
for category in categories:
    tempTestPath = os.path.join(testPath,category)
    for img in os.listdir(tempTestPath):
        try:
            tempTestImage = cv2.imread(os.path.join(tempTestPath,img),cv2.IMREAD_GRAYSCALE)
            tempCropedTestImage = cv2.resize(tempTestImage,(imgSize,imgSize))
            testSet.append([tempCropedTestImage,categories.index(category)])
        except Exception as e:
            continue


# In[73]:


import random
random.shuffle(trainSet)
random.shuffle(testSet)


# In[74]:


trainInputs =[]
trainTargets = []
testInputs = []
testTargets = []


# In[75]:


for inputs, targets in trainSet:
    trainInputs.append(inputs)
    trainTargets.append(targets)
for inputs, targets in testSet:
    testInputs.append(inputs)
    testTargets.append(targets)


# In[78]:


np.savez('TrainData.npz',data = trainInputs, label = trainTargets)
np.savez('TestingData.npz',data = testInputs, label = testTargets)


# In[4]:


loadedTrainSet = np.load('TrainData.npz')
loadedTestSet = np.load('TestingData.npz')


# In[5]:


trainSetInputs = np.array(loadedTrainSet['data']).reshape(-1,imgSize,imgSize,1)
trainSetTargets = loadedTrainSet['label']
testSetInputs = np.array(loadedTestSet['data']).reshape(-1,imgSize,imgSize,1)
testSetTargets = loadedTestSet['label']


# In[6]:


scaledTrainSetInputs = trainSetInputs/255.
scaledTestSetInputs = testSetInputs/255.


# In[7]:


import time
name = "pneumonia{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = "logs/{}".format(name))
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(2,2), activation='relu', input_shape = scaledTrainSetInputs.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64,(2,2), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')  
])


# In[8]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['accuracy'])


# In[14]:


model.fit(scaledTrainSetInputs, trainSetTargets,batch_size = 32, validation_split = 0.1 ,epochs = 20,callbacks = [tensorboard])


# In[15]:


model.evaluate(testSetInputs,testSetTargets)


# In[18]:


model.save('pneumonia.model')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script ')

