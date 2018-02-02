#-*- coding:utf-8 –*-
from sklearn.linear_model import LogisticRegression
import csv
from numpy import *
from sklearn.neighbors import KNeighborsClassifier
#这两个函数在loadTrainData()和loadTestData()中被调用
#toInt()将字符串数组转化为整数，nomalizing()归一化整数
def toInt(array):
    array=mat(array)
    # print array,'array'
    m,n=shape(array)
    # print m,n
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
           newArray[i,j]=int(array[i,j])
    return newArray

def normalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

#这个函数从train.csv文件中获取训练样本:trainData、trainLabel
def loadTrainData():
    l=[]
    with open('train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
#list变成了数组
    l=array(l)
 # get the 1st-col element in every rows
    label = l[:,0]
    data = l[:, 1:]
    return normalizing(toInt(data)),toInt(label)

def loadTestData():
    l=[]
    with open('test.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data=array(l)
    return normalizing(toInt(data))

def loadTestResult():
    l=[]
    with open('benchmark.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,0])
#return an array

def saveResult(result):
    with open('result.csv','wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            temp=[]
            temp.append(i)
            myWriter.writerow(temp)
#functions above are basic array/list operation functions

#functions below encapsulate machine learning algorithm

def knnClassify(trainData,trainLabel,testData):
    knnClf=KNeighborsClassifier()
    knnClf.fit(trainData,ravel(trainLabel))#? input are two arrays
    testLabel=knnClf.predict(testData)
    saveResult(testLabel)
    return testLabel
from sklearn import svm
def svmClassify(trainData,trainLabel,testData):
    svmClf=svm.SVC(C=5.0)
    svmClf.fit(trainData,ravel(trainLabel))
    testLabel=svmClf.predict(testData)
    saveResult(testLabel)
    return testLabel

#调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
def GaussianNBClassify(trainData,trainLabel,testData):
    nbClf=GaussianNB()
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel)
    return testLabel



if __name__ == '__main__':
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    resultKNN=knnClassify(trainData,trainLabel,testData)
    print resultKNN,'prediction result'
    resultGiven=loadTestResult()
    print resultGiven,'result given'

