import pandas as pd
import numpy as np
import random


def sigmoid(x):
    return 1/(1+np.exp(-x))

def costfunction(x,y,theta):
    return -sum(y*np.log(sigmoid(x.dot(theta)))+(1-y)*np.log(1-sigmoid(x.dot(theta))))/(2*len(y))

def gradient(x,y,theta):
    a= ((sigmoid(x.dot(theta))-y))/len(y)
    return x.T.dot(a)
def graddescent(x,y,theta,iteration,alpha):
    i=0   
    while i<=iteration:
        print(costfunction(x,y,theta))
        theta=theta-alpha*(gradient(x,y,theta))
        i+=1
    return theta	
da=pd.read_csv('iris.data',header=None)
x=np.array(da.iloc[:,:-1])
m=len(x)
x=np.append(np.ones([m,1]),x,axis=1)
y=np.array(da.iloc[:,-1])
for i in range(len(y)):
    if y[i]=='Iris-setosa':
        y[i]=0
    if y[i]=='Iris-versicolor':
        y[i]=1
    if y[i]=='Iris-virginica':
        y[i]=2
print(x,y)

theta=np.random.rand(5,3)

for i in range(3):
    print(i+1)
    theta[:,i]=graddescent(x,y==i,theta[:,i],100,0.1)

print(theta)

tesda=pd.read_csv('test.data',header=None)
x=np.array(tesda.iloc[:,:-1])
m=len(x)
x=np.append(np.ones([m,1]),x,axis=1)
result=np.array(tesda.iloc[:,-1])
prediction=[]
print(prediction)
for i in range(3):
    print(i+1)
    h=sigmoid(x.dot(theta[:,i]))
    prediction.append([p for p in h])


prediction=np.array(prediction)
print(prediction)
onvsall=np.argmax(prediction,axis=0)
print(onvsall)

verdict=[]

for i in onvsall:
    if i==0:
        verdict.append('Iris-setosa')
    elif i==1:
        verdict.append('Iris-versicolor')
    elif i==2:
        verdict.append('Iris-virginica')	

verdict=np.array(verdict)
print("the prediction says:")
print(verdict)
print("the actual answer turns out to be:")
print(result)

z=verdict==result
print("accuracy of the model is :")
print(sum(z)/len(z)*100)



