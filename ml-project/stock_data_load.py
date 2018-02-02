# coding=utf-8
import tushare as ts
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib as mpl

# acquire stock data
def load_data(filepath):
    print ts.__version__
    # print  ts.get_hist_data('600848')  # 一次性获取全部日k线数据
    #da1=pd.DataFrame

    df=ts.get_today_all()
    print df.shape
    selected_columns=['code','name']
    df.to_csv(filepath,columns=selected_columns,encoding='utf-8')
def get_day_data(filepath):
    print filepath+os.path.sep+'stock_code.csv'
    code_df=pd.read_csv(filepath+os.path.sep+'stock_code_test.csv',dtype={'code':str})
    code_df=code_df['code']
    n=code_df.shape[0]
    nutlist=[]
    codelist=[]
    # filelist=os.listdir(filepath + '\HistoryQuotations\Day')
    # for i in range(n):
    #    code = code_df[i]
    #    print code
    #    print type(code)
    #    codelist.append(str(code))
    # for file in filelist:
    #     size=os.path.getsize(filepath + '\HistoryQuotations\Day'+os.path.sep+file)
    #     size =size/float(1024)
    #     if round(size)==0:
    #         nutlist.append(str(file))

        # if filename not in codelist:
        #     nutlist.append(filename)

    #从所有的股票代码里面读出代码一一去查历史数据，并且保存到相应文件中
    for i in range(n):

        code = code_df[i]
        # print code
        path = filepath + '\HistoryQuotations\Month' + os.path.sep + str(code) + '.csv'
        if os.path.exists(path):
            pass
            print path,'has exists'
        else:
            f = open(path, 'w')
            df_data=ts.get_hist_data(str(code),ktype='M')
            df_data.to_csv(path)
            print path,'**********new'
def  concatdf(filepath1,filepath2):
    df1=pd.read_csv(filepath1,index_col='date')
    df2=pd.read_csv(filepath2,index_col='date')
    df2=df2.loc['2017-11-28':'2017-11-15','close_index':'turnover_index']
    all_df=pd.concat([df1,df2],axis=1)

    print df2
    all_df.to_csv('D:\stock_data\stock\\union.csv')


def label_maxmin(filepath):
    df=pd.read_csv(filepath,index_col='date')
    print df.index
    df_open=df.loc['2017-12-04':'2017-10-09','open']
    df_open = df_open.iloc[::-1]
    print df_open.head()
    print df_open.shape
    df_open.plot()
    plt.show()
    rate=2
    window=5
    label_dict={}
    open=df_open.as_matrix()
    print open,'matrix'

    open=list(open)
    print len(open)
    # for i in range(0,len(open)-window):
    #     # print open[i:i+window]
    #     # print open[i+window:i+window*2]
    #     if len(open[i+window:i+window*2])<5:
    #         print 'break'
    #         break
    #
    #     label_dict[i]=compare(window,open[i:i+window],open[i+window:i+window*2])
    # i是在数组也是在df_open里面的索引

    label_dict=compare1(window,open)
    date_list=[df_open.index[i] for i in label_dict.keys()]
    print label_dict
    print date_list
    df_label=pd.DataFrame(label_dict.values(),index=date_list,columns=['label'])
    print df_label
    df_label.to_csv('D:\stock_data\stock\\label.csv')

#比较一段时间的
def compare(window,list1,list2):
    result_list=[]
    value=0
    flag=0
    if window%2!=0:
        value=(list2[0]-list1[0])+(list2[window/2]-list1[window/2])+(list2[window-1]-list1[window-1])
    else:
        value = (list2[0] - list1[0]) + ((list2[window/2]+list2[window/2+1])/2 - (list1[window/2]+list1[window/2+1])/2 ) + (
        list2[window - 1] - list1[window - 1])

    if value<0:
        #下降
        flag=0
    elif value==0:
        #平稳
        flag=1
    else:
        #上升
        flag=2
    result_list=[flag,value]
    return result_list

def compare1(window,df_list):
    temp_dict={}
    values_dict={}
    result_dict={}
    #一个窗口距离的是near的权重，一个窗口两个窗口之间距离的是far的权重
    weight_near=1
    weight_far=1
    #波动，绝对值小于这个值算为平稳
    wave=1.5
    label=0
    if len(df_list)<=window*2+1:
        return temp_dict
    for i in range(len(df_list)):
        if i-window*2>=0 and i+window*2<len(df_list):
            print i
            # values_dict[i]=[df_list[i-window*2],df_list[i-window],df_list[i+window],df_list[i+window*2]]

            value_left=weight_far*(df_list[i]-df_list[i-window*2])+weight_near*(df_list[i]-df_list[i-window])
            value_right=weight_near*(df_list[i+window]-df_list[i])+weight_far*(df_list[i+window*2]-df_list[i])

            if abs(value_left)<=wave and abs(value_right)<=wave:
                #平平
                label=0
            elif abs(value_left)<=wave and value_right>wave:
                #平上
                label=1
            elif abs(value_left)<=wave and value_right<wave:
                #平下
                label=2
            elif value_left>wave and value_right>wave:
                #上上
                label=3
            elif value_left>wave and abs(value_right)<=wave:
                #上平
                label=4
            elif value_left>wave and value_right<-1*wave:
                #上下
                label=5
            elif  value_left<-1*wave and value_right<-1*wave:
                #下下
                label=6
            elif value_left<-1*wave and abs(value_right)<=wave:
                #下平
                label=7
            elif value_left<-1*wave and value_right>wave:
                #下上
                label=8
            # temp_dict[i]=[label,value_left,value_right]
            temp_dict[i]=label
    return temp_dict

















if __name__ == '__main__':

    filepath='D:\stock_data\stock\HistoryQuotations\Day\\000001.csv'
    filepath1='C:\Users\Administrator\Desktop\\turn.csv'
    filepath2='D:\stock_data\stock\\000001.csv'
    # what the fuck am i doing
    # load 一个弯弯曲曲的数据进来,标上label，极大值极小值
    label_maxmin(filepath)
    # concatdf(filepath1,filepath2)
    # get_day_data(filepath)
    # find2nuts(filepath)
    #load_data(filepath)
