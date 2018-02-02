# coding=utf-8
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plr_segment import segment
from plr_segment import fit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#取到第data_n行的数据(不包括）
data_n=300
#测试数据个数
test_n=20

def draw_plot(data,plot_title):
    plot(range(len(data)),data,alpha=0.5,color='red')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))

def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)

#plr把数据分段线性表示
def load_data(filepath):
    with open(filepath) as f:
        file_lines = f.readlines()

    # data = [float(x.split("\t")[2].strip()) for x in file_lines[100:320]]
    # print(data)
    #2是最高价 high
    data = [float(x.split(',')[2]) for x in file_lines[1:]]
    #csv文件里面数据的时间顺序是倒过来的
    data=list(reversed(data))
    print (type(data))
    max_error = 0.005

    # sliding window with regression
    #segment是蓝色的线段
    figure()
    segments = segment.slidingwindowsegment(data, fit.regression, fit.sumsquared_error, max_error)
    # print (segments)
    draw_segments(segments)
    draw_plot(data, "Sliding window with regression")
    show()
    #返回分段的坐标的列表
    return segments

#这个函数是用来对每个点状态进行标记，根据分段斜率
#波峰为1，波谷为2，持续下跌为3，持续上涨为4
#普通上涨点改为5，普通下降点改为6,0为完全平稳点
def label_data_peak(segments_list):
    slope_list=[]
    label_dic={}
    #item example：(0, 7.6199999999999992, 1, 7.4499999999999993)
    #分别是 分段起点索引，值，终点索引，终点值
    for item in segments_list:
        slope=(item[3]-item[1])/(item[2]-item[0])
        slope_list.append([slope,item[0],item[2]])
    print slope_list,'slope_list'
    # slope list是斜率列表
    #example：[-0.16999999999999993, 0, 1]
    #分别是斜率，分段起点和终点
    #普通点的标记，普通上涨点改为5，普通下降点改为6,0为完全平稳点
    for i in range(len(slope_list)):
        slope=slope_list[i][0]
        start=slope_list[i][1]
        end=slope_list[i][2]
        if slope>0:
            for j in range(start,end):
                label_dic[j]=5
        elif slope<0:
            for j in range(start, end):
                label_dic[j]=6
        else:
            for j in range(start, end):
                label_dic[j]=0
       # 这四段判断的是拐点的状态，因为判断拐点要取上一段和下一段斜率，
       # 所以这里循环时候索引去掉最后一个分段
    for i in range(len(slope_list) - 1):
        if slope_list[i][0] > 0 and slope_list[i + 1][0] < 0:
            label_dic[slope_list[i][2]] = 1
            #print '00000',运行到了 但是没能改得了字典里面的值，为什么呢
        elif slope_list[i][0] < 0 and slope_list[i + 1][0] > 0:
            label_dic[slope_list[i][2]] = 2
        elif slope_list[i][0] < 0 and slope_list[i + 1][0] < 0:
            label_dic[slope_list[i][2]] = 3
        elif slope_list[i][0] > 0 and slope_list[i + 1][0] > 0:
            label_dic[slope_list[i][2]] = 4
        else:
            label_dic[slope_list[i][2]] = 9

    print label_dic
    print label_dic.values().count(6)
    return label_dic

#把所有数据都标上标签，返回用于训练的数据和标签
def real_label_data(filepath,label_dic):
    df=pd.read_csv(filepath,usecols=[ 'open', 'high', 'close', 'low', 'volume', 'turnover','alt','itl'])
    df=df.iloc[::-1]
    label_n=len(label_dic)
    #分段的时候，最后一小段总是会分不了，所以分段照原数据会丢失一部分
    #label里面标了多少数据，就把特征矩阵里面这些数据切分出来用来训练
    df=df[0:label_n]
    # label_start_idx=label_dic.keys()[0]
    # label_end_idx=label_dic.keys()[-1]
    # df=df[label_start_idx:label_end_idx+1]

    print df.shape,'df.shape'
    label_n=df.shape[0]
    #变成算法要的二维数组
    label_mat=np.array(label_dic.values())
    print label_mat.shape,'label_mat.shape'
    features=df.as_matrix()
    print features.shape,'features.shape'
    return features,label_mat
#向训练数据文件增加特征 ALT（价格振幅指标），ITL（K线指数）
def add_features(filepath):
    df = pd.read_csv(filepath, usecols=['date','open', 'high', 'close', 'low', 'volume', 'turnover'])
    df = df.iloc[::-1]
    high_mat=df['high'].as_matrix()
    low_mat=df['low'].as_matrix()
    close_mat=df['close'].as_matrix()
    open_mat=df['open'].as_matrix()
    #构造alt矩阵
    alt_mat=(high_mat-low_mat)/low_mat
    df_alt= pd.DataFrame(alt_mat, columns=['alt'])
    #构造差值矩阵
    dif_mat=close_mat-open_mat
    itl_mat=[1 if item>0 else -1 for item in dif_mat]
    df_itl=pd.DataFrame(itl_mat,columns=['itl'])
    # print df_itl.shape
    # print df_alt.shape
    df=pd.concat([df,df_alt,df_itl],axis=1)
    print df.shape
    print df.columns
    df.to_csv(filepath)



#svm分类器
def training(features,label,testfilepath):

    # clf=SVC()
    # clf.fit(features,label)
    clf=LogisticRegression()
    clf.fit(features,label)
    #我手动分的测试集
    df = pd.read_csv(testfilepath,usecols=[ 'open', 'high', 'close', 'low', 'volume', 'turnover','alt','itl'])
    df = df.iloc[::-1]

    # df=df[0:265]

    # 变成算法要的二维数组
    test = df.as_matrix()
    test_show=df['high'].as_matrix()
    print test.shape,'test.shape'
    # test_d=[[4.24,4.24,4.17,4.17,34317.34,0.36]]
    #predict可以放test_show
    print clf.predict(test),'predict'
    print clf

    #下面截取的代码是我把测试数据里面的high值画出来了
    max_error = 0.005
    # sliding window with regression
    # segment是蓝色的线段
    figure()
    segments = segment.slidingwindowsegment(test_show, fit.regression, fit.sumsquared_error, max_error)
    # print (segments)
    draw_segments(segments)
    draw_plot(test_show, "test data trend")
    show()

if __name__ == '__main__':
    #这个数据在plr_segment/example_data里面
    filepath = 'E:\JetBrains\PycharmProjects\ml-project\plr_segment\example_data\\000005train.csv'
    testfilepath='E:\JetBrains\PycharmProjects\ml-project\plr_segment\example_data\\test.csv'
    # add_features(testfilepath)
    segments_list=[]
    segments_list=load_data(filepath)
    print segments_list
    label_dict=label_data_peak(segments_list)
    #real_label_data(filepath)
    features, label_mat=real_label_data(filepath,label_dict)
    training(features,label_mat,testfilepath)



