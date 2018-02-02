# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

def loadData(filepath):
    dataMax=[]
    labelMax=[]
    f=open(filepath)
    for line in f.readlines():
        lineArr=line.strip().split()
        dataMax.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMax.append(int(lineArr[2]))
    f.close()
    return dataMax,labelMax

def plotData(filepath,weights):
    dataMat,labelMat=loadData(filepath)
    print dataMat
    n=dataMat.__len__()
    print n
    xcord1 = []
    ycord1 = []  # 正样本
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # scale:enlarge the marker marker: shape of marker
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=0.5)
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=0.5)
    plt.title('DataSet')  # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')  # 绘制label
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')  # 绘制label
    plt.show()


def sigmoid(inX):
    return 1.0/(1.0+np.exp(-inX))

def gradAscent(dataMat,labelMat):
    dataMat=np.mat(dataMat)
    labelMat=np.mat(labelMat).transpose()
    # print dataMat,labelMat
    m,n=dataMat.shape
    print m,n
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights=weights+alpha*dataMat.transpose()*error
    return weights.getA()


if __name__ == '__main__':
    filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\Logistic\\testSet.txt'
    dataMat,labelMat=loadData(filepath)
    weights=gradAscent(dataMat,labelMat)
    plotData(filepath,weights)
