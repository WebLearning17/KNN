#coding=UTF-8
from numpy import *
import operator

fr=open("datingTestSet.txt")
lines=fr.readlines()
returnMat=zeros((len(lines),3))
classLabelVector=[]
index =0
dictLabel={'largeDoses':1,'smallDoses':2,'didntLike':3}
for line in lines:
    line=line.strip()
    listFromLine=line.split('\t')
    returnMat[index,:]=listFromLine[0:3]
    classLabelVector.append(int(dictLabel.get(listFromLine[-1])))
    index+=1
print returnMat[1,2]

import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(returnMat[:,1],returnMat[:,2],15.0*array(classLabelVector),15.0*array(classLabelVector))
#plt.show()

dataSet=array([[3,4,5],[1,2,6],[4,5,6]],dtype=float)
minVal=dataSet.min(0)
maxVal=dataSet.max(0)
ranges=maxVal-minVal
normDataSet=zeros(shape(dataSet),dtype=float)
m=dataSet.shape[0]
normDataSet2=dataSet-tile(minVal,(m,1))
normDataSet2.astype(float)
print(normDataSet2)
print(tile(ranges,(m,1)))
normDataSet3=normDataSet2/tile(ranges,(m,1))

import kNN
normMat,ranges,minVal=kNN.autoNorm(returnMat)
print(normMat)