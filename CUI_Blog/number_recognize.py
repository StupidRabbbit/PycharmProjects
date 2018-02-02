# coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

"""
def img2Vector(filename):
    returnVect=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\kNN\\3.digit_recognize\\trainingDigits'
    hwLabels=[]
    trainingFileList=listdir(filepath)
    m=len(trainingFileList)
    print m
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        classNum=int(fileNameStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:]=img2Vector(filepath+'\\'+fileNameStr)
        # if i==1:
        #     print filepath+'\\'+fileNameStr
    neigh=KNeighborsClassifier(n_neighbors=3,algorithm='auto')
    neigh.fit(trainingMat,hwLabels)
    testFile='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\kNN\\3.digit_recognize\\testDigits'
    testFileList=listdir(testFile)
    errorCount=0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumber=int(fileNameStr.split('_')[0])
        testVector=img2Vector(filepath+'\\'+fileNameStr)
        classfierNumber=neigh.predict(testVector)
        print '预测结果:', classfierNumber, '真实结果:', classNumber
        if classfierNumber!=classNumber:
            errorCount+=1.0
        # print filepath+'\\'+fileNameStr
    print 'prediction error ratio is:',errorCount/float(mTest)*100,'%'
    print 'error amount is :',errorCount



if __name__ == '__main__':
    handwritingClassTest()