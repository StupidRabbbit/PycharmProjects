# coding=utf-8
import numpy as np
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def setOfWord2Vect(vocabList,inputSet):
    returnVect=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVect[vocabList.index(word)]=1
        else:
            print 'word is not in my V'
    return returnVect

def createVocabList(dataSet):
    vocabSet=set([])
    for data in dataSet:
        vocabSet=vocabSet|set(data)
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
#calculate the probability of word in abusive/not abusive docs
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1NumVect=np.log(p1Num/p1Denom)
    p0NumVect=np.log(p0Num/p0Denom)
    return p0NumVect,p1NumVect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    print vec2Classify*pClass1

    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0






if __name__ == '__main__':
  postingList,classVec=loadDataSet()
  myVocabList=createVocabList(postingList)
  trainMat=[]
  for element in postingList:
      mat=setOfWord2Vect(myVocabList,element)
      trainMat.append(mat)
  print trainMat
  p0V,p1V,pAbusive=trainNB0(np.array(trainMat),np.array(classVec))
  print p0V,p1V,pAbusive
  test=['love', 'my', 'dalmation']
  thisDoc=np.array(setOfWord2Vect(myVocabList,test))
  if classifyNB(thisDoc, p0V, p1V, pAbusive):
      print test, '属于侮辱类'  # 执行分类并打印分类结果
  else:
      print test, '属于非侮辱类'

  # a=np.array([1,2,3])
  # b=np.array([1,2,3])
  # print a*b
  # print reduce(lambda x, y: x * y, a*b)