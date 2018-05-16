#coding=UTF-8
from numpy import *
import operator
#列举文件
from os import listdir
import kNN
def img2vector(filename):
    returnVec=zeros((1,1024),dtype=float)
    fr=open(filename)
    for i in range(32):
        linstr=fr.readline()
        for j in range(32):
            returnVec[0,32*i+j]=int(linstr[j])

    return returnVec
#手写数字识别的测试代码
hwLabels=[]
trainningFilelIst=listdir('digits/trainingDigits')
m=len(trainningFilelIst)
trainingMat=zeros((m,1024),dtype=float)

for i in range(m):
    fileNAmeStr=trainningFilelIst[i]
    fileStr=fileNAmeStr.split('.')[0]
    classNumber=int(fileStr.split('_')[0])
    hwLabels.append(classNumber)
    trainingMat[i,:]=img2vector('digits/trainingDigits/%s'%fileNAmeStr)

testFileList=listdir('digits/testDigits')

errorCount=0.0
mTest=len(testFileList)
fileNAmeStr=testFileList[0]
print(fileNAmeStr)
fileStr=fileNAmeStr.split('.')[0]
classNumber=int(fileStr.split('_')[0])
print (classNumber)

for  i in range(mTest):
    fileNAmeStr=testFileList[i]
    fileStr=fileNAmeStr.split('.')[0]
    classNumber=int(fileStr.split('_')[0])
    vectorUnderTest=img2vector('digits/testDigits/%s'%fileNAmeStr)
    classifierResult=kNN.classify0(vectorUnderTest,trainingMat,hwLabels,3)
    print("测试返回结果：%s,真实的结果:%s"%(classifierResult,classNumber))
    if(classifierResult!=classNumber):errorCount+=1.0

print(r"错误率：",(errorCount/float(mTest)))


