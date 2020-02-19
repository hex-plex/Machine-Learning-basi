#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import cv2


# In[ ]:


traindata=[]


for i in range(7696):
    temp=cv2.imread("/kaggle/input/mlware/data_images/train/train"+str(i)+".jpg")
    temp=cv2.resize(temp,(100,100))        
    traindata.append(temp)
    
traindata=np.array(traindata,dtype='float64')


# In[ ]:



import gc


# In[ ]:





# In[ ]:


Xfull=traindata

Xfull=(Xfull-np.mean(Xfull))/np.std(Xfull)
X=Xfull

X=np.array(X,dtype='float64')
classi=pd.read_csv("/kaggle/input/mlware/train.csv")
yfull=np.array(classi)
y=yfull[:,1]
y=np.array(y,dtype='float64')

n=len(X[1,:])
classes=list(range(7))


# In[ ]:


model=keras.Sequential([
    keras.layers.Conv2D(100,(3,3),activation='relu',input_shape=(100,100,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(500,(3,3)),
    keras.layers.LeakyReLU(alpha=0.08),    
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(100,(5,5)),
    keras.layers.LeakyReLU(alpha=0.08),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(25,kernel_regularizer=keras.regularizers.l2(0.02)),
    keras.layers.LeakyReLU(alpha=0.08),
    keras.layers.Dense(7,kernel_regularizer=keras.regularizers.l2(0.02),activation="softmax")
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


model.fit(X,y,epochs=15)
del traindata
del Xfull
del X
del yfull
del classi
del y
#testdata=pd.read_csv("/kaggle/input/mlware/mnist_test_56x56_grayscale.csv")
gc.collect()
testdata=[]
for i in range(3676):
    temp=cv2.imread("/kaggle/input/mlware/data_images/test/test"+str(i)+".jpg")
    temp=cv2.resize(temp,(100,100))        
    testdata.append(temp)
    
testdata=np.array(testdata,dtype='float64')
Xtest=testdata
Xtest=np.array(Xtest,dtype='float64')
Xtest=(Xtest-np.mean(Xtest))/np.std(Xtest)
#ytest=yfull[5000:,1]
#ytest=np.array(ytest,dtype='float64')
#loss,accu=model.evaluate(Xtest,ytest)
#print(loss,accu)
prediction=model.predict(Xtest)


# In[ ]:


print(prediction)


# In[ ]:


predicclass=np.argmax(prediction,axis=1)

print(list(predicclass))


# In[ ]:


subm={'image':names,'label':predicclass}
clas=pd.DataFrame(subm)
print(clas)


# In[ ]:


clas.to_csv("/kaggle/working/submission_tsfhg.csv",index=False)


# In[ ]:


print(list(names))


# In[ ]:




