# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
def time_analysis(filepath):
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 100)
    data=pd.read_csv(filepath,index_col='date',usecols=['date','open'])

    #可以显示中文，可以正确显示正负号
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    # data.plot()
    #
    # plot_acf(data)
    # print 'ADF',ADF(data['open'])
    forcast_n=20
    training_n=150

    rows = data.shape[0]
    n = rows-forcast_n
    # reverse the df
    data=data.iloc[::-1]
    m_start=n-training_n
    D1_data = data[m_start:n]
    ori_data=data[m_start:n]
    ori_b = data['open'][n:]
    print ori_b
    # ori_data.plot()
    print 'ori_ADF',ADF(ori_data['open'])

    # np.log2(D1_data).plot()
    #这块我处理了一下元数据，变成log和平方根
    D1_data_log= np.log(D1_data)
    D1_data_sqrt=np.sqrt(D1_data)

    D1_data=D1_data_sqrt.diff(2).fillna(0)
    # D1_data=D1_data.diff().fillna(0)
    new_name='open_sqrt_diff'
    D1_data.columns=[new_name]
    print 'D1_data',D1_data

    print 'D1_data:shape',D1_data.shape
    # print 'current month ',D1_data.iloc[n-m_start-1,]
    D1_data.plot()
    #画偏？？相关系数图
    plot_acf(D1_data)
    plot_pacf(D1_data)
    print 'ADF', ADF(D1_data[new_name])

    print 'white noise: ',acorr_ljungbox(D1_data,lags=1)
    plt.show()

    from statsmodels.tsa.arima_model import ARIMA
    #寻找合适的p,q
    bic_matrix=[]
    for p in range(5):
        tmp=[]
        for q in range(5):
            try:
                tmp.append(ARIMA(D1_data_sqrt,(p,2,q)).fit(disp=-1).bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix=pd.DataFrame(bic_matrix)
    p,q=bic_matrix.stack().idxmin()
    print p,q



    result_model= ARIMA(D1_data_sqrt, order=(p, 2, q)).fit(disp=-1)

    plt.plot(D1_data)
    plt.plot(result_model.fittedvalues, color='red')

    plt.show()
    # print D1_data_log[2:].shape
    print 'result_model:shape',result_model.fittedvalues.shape
    rss=(result_model.fittedvalues.reshape((training_n-2,1))- D1_data[2:])
    # result=rss.sum()
    print 'rss',sum(rss.as_matrix()**2)
    print result_model.forecast(forcast_n)[0]
    print np.power(result_model.forecast(forcast_n)[0],2)

    # plt.plot(ori_data)
    # plt.plot(np.exp(result_model.fittedvalues),color='yellow')
    # plt.show()
    #还原被log或差分数据（差分数据没还原出来）
    a=np.power(result_model.forecast(forcast_n)[0],2)
    b=pd.Series(a)
    # print type(result_model.forecast(forcast_n)[0])

    print ori_b.as_matrix()
    plt.plot(ori_b,color='red')
    plt.plot(b)
    plt.show()
    # print result


    #
    # model=ARIMA(ori_data,(0,2,q)).fit()
    # # model.summary2()
    # print model.forecast(5)
    # print 'p,q', p, q

if __name__ == '__main__':
    filepath='D:\stock_data\stock\HistoryQuotations\Week\\000001.csv'
    time_analysis(filepath)