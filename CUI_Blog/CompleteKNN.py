# -*- coding: UTF-8 -*-
#helen dating
#filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\kNN\\2.helen\datingTestSet.txt'
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator
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
    # print distances
    sortedDistanceIndices=distances.argsort()
    # print sortedDistanceIndices
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistanceIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # print sortedClassCount[0][0]
    return sortedClassCount[0][0]
def file2matrix(filename):
        classLabelVector = []
        fr=open(filepath)
        arrayOLines=fr.readlines()
        numberOfLines=len(arrayOLines)
        returnMat=np.zeros((numberOfLines,3))
        index=0
        for line in arrayOLines:
            line=line.strip()
            listFromLine=line.split('\t')
            returnMat[index,:]=listFromLine[0:3]

            if listFromLine[-1]=='didntLike':
                classLabelVector.append(1)
            elif listFromLine[-1]=='smallDoses':
                classLabelVector.append(2)
            elif listFromLine[-1]=='largeDoses':
                classLabelVector.append(3)
            index+=1
        return returnMat,classLabelVector

def showdatas(datingDataMat,datingLabels):
    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    numberOfLabels=len(datingLabels)
    LabelColors=[]
    for i in datingLabels:
        if i==1:
            LabelColors.append('black')
        if i==2:
            LabelColors.append('orange')
        if i==3:
            LabelColors.append('red')

    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelColors,s=15, alpha=.5)
    # axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    # axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    # axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    # plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    # plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    # plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    axs[0][0].set_title('axis1')
    axs[0][0].set_xlabel('game time')
    axs[0][0].set_ylabel('plane time')
    # didntLike = mlines.Line2D([], [], color='black', marker='.',
    #                           markersize=6, label='didntLike')
    # smallDoses = mlines.Line2D([], [], color='orange', marker='.',
    #                            markersize=6, label='smallDoses')
    # largeDoses = mlines.Line2D([], [], color='red', marker='.',
    #                            markersize=6, label='largeDoses')
    # # 添加图例
    # axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    # axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    #图例
    axs[0][0].legend()



    # plt.show()

def autoNorm(dataSet):
    # add min(0) 0 to get the min vector  min()to get min value of matrix
    minVal=dataSet.min(0)
    maxVal=dataSet.max(0)
    ranges=maxVal-minVal

    normDataSet=np.zeros(np.shape(dataSet))

    m=dataSet.shape[0]
    #this is oldvalue-min differention vector
    normDataSet=dataSet-np.tile(minVal,(m,1))
    #原始值-最小值/最大值-最小值
    #ranges is max-min differention vector
    normDataSet=normDataSet/np.tile(ranges,(m,1))

    return normDataSet,ranges,minVal

def datingClassTest():
    filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\kNN\\2.helen\datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filepath)
    hoRatio=0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0

    for i in range(numTestVecs):
        classifyResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print '预测结果:',classifyResult,'真实结果:',datingLabels[i]
        if classifyResult!=datingLabels[i]:
            errorCount+=1.0
    print 'errorCount is:',errorCount/float(numTestVecs)*100,'%'







if __name__ == '__main__':
   filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\kNN\\2.helen\datingTestSet.txt'
   datingDataMat,datingLabels=file2matrix(filepath)
   # print datingDataMat
   # print datingLabels
   showdatas(datingDataMat,datingLabels)
   normDataSet, ranges, minVals = autoNorm(datingDataMat)
   # print normDataSet
   datingClassTest()
   #over 2017.11.14