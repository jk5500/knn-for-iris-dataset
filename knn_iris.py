# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 19:33:25 2016

@author: kamal
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import statistics
from matplotlib import colors as c

iris = datasets.load_iris()
X = iris.data[:, [1,2]]  # we only take the first two features.
Y = iris.target
c1=X[Y==0]
c2=X[Y==1]
c3=X[Y==2]
'''
plt.plot(c1[:,0],c1[:,1],'ro')
plt.plot(c2[:,0],c2[:,1],'ko')
plt.plot(c3[:,0],c3[:,1],'bo')
'''
xmin,xmax=X[:,0].min(),X[:,0].max()
ymin,ymax=X[:,1].min(),X[:,1].max()
h=0.01
XX,YY=np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
Z=np.c_[XX.ravel(),YY.ravel()]
m=Z.shape[0]
K=15
myvalue=[]
for i in range(m):
    value=X-Z[i]
    distsquare=value**2
    distsum=distsquare.sum(axis=1)
    sqrt=np.sqrt(distsum)
    sortdist=sqrt.argsort()
    sortval=sortdist[0:K]
    labels=Y[sortval]
    if statistics.mode(labels)==0:
        values=1
    elif statistics.mode(labels)==1:
        values=2
    elif statistics.mode(labels)==2:
        values=3
    myvalue.append(values)
    
myvalue=np.array(myvalue)
myvalue=myvalue.reshape(XX.shape)
cMap = c.ListedColormap(['m','c','y'])# first color label 1,second label 2 and so on
plt.pcolormesh(XX,YY,myvalue,cmap=cMap)
plt.scatter(c1[:,0],c1[:,1],color='g',label='setosa')
plt.scatter(c2[:,0],c2[:,1],color='r',label='versicolor')
plt.scatter(c3[:,0],c3[:,1],color='b',label='virginica')
plt.axis([XX.min(), XX.max(), YY.min(), YY.max()])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()