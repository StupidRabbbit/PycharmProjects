# coding=utf-8
from math import log
import operator
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    # labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性
    labels = ['age', 'work', 'house', 'credit']
    return dataSet, labels                #返回数据集和分类属性
#calculate entropy of dataSet
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    shannonEnt=0
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEnt=calcShannonEnt(dataSet)
    bestInfoeGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEnt=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            print calcShannonEnt(subDataSet)
            newEnt+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEnt-newEnt
        # print '%d feature_s info gain is' %(i),infoGain
        if infoGain>bestInfoeGain:
            bestInfoeGain=infoGain
            bestFeature=i

    return  bestFeature

def splitDataSet(dataSet,i,value):
    retDataSet=[]
    for featVec in dataSet:
        if(featVec[i]==value):
            retDataSet.append(featVec)
    return retDataSet

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount:
            classCount[vote]=0
        classCount[vote]+=1
    sortClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return classCount[0][0]
def createTree(dataSet,labels,featLabels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeat)
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueValues=set(featValues)
    for value in uniqueValues:
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree


if __name__ == '__main__':
   dataSet,labels=createDataSet()
   # shannoEnt=calcShannonEnt(dataSet)
   score=chooseBestFeatureToSplit(dataSet)
   print 'we choose:',str(score)
   featLabels=[]
   print createTree(dataSet,labels,featLabels)