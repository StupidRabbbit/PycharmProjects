# -*- coding: UTF-8 -*-
import numpy as np
import operator

def CreateDataSet():
    # a matrix
    group=np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    # compute the differentiation matrix of test and dataSet,through copying matrix test dataSetSize times along axis Y
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet

    # sqDiffMat=np.square(diffMat)
    # print sqDiffMat
    sqDiffMat=diffMat**2
    # print sqDiffMat
    sqDistance=sqDiffMat.sum(axis=1)
    distances=np.sqrt(sqDistance)
    print distances
    sortedDistanceIndices=distances.argsort()
    print sortedDistanceIndices
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistanceIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print sortedClassCount[0][0]
    return sortedClassCount[0][0]





if __name__ == '__main__':
    group,labels=CreateDataSet()
    test=[101,20]

    classify0(test,group,labels,3)