# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn import preprocessing
import zipfile
import os
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process,Pool
import time
import seaborn as sns
import random
# df=None
# time1=time.time()
# result=pool.map(lambda x: x['type'].value_counts(),reader )
# print 'time1 is ',(time.time()-time1)
# print result[0]
#time1:53.3s

# time2=time.time()
# for chunk in reader:
#     chunk['type'].value_counts()
# print 'time2 is',(time.time()-time2)
#time2:56s
def concat(df):
    return df
def cut_data():
    # pool = ThreadPool(5)
    filepath = 'D:\paysim1\\PS_20174392719_1491204439457_log.csv'
    time3 = time.time()
    raw_data = pd.read_csv(filepath)
    print raw_data[raw_data['isFraud']==1]['nameOrig'].head()
    # 求出type和isFlaggedFraud标记之间的关系，即是否有欺诈记录和交易类型间的关系
    # type和欺诈有着很直接的关系，只有cashout和transfer的存在欺诈
    ax = raw_data.groupby(['type', 'isFraud']).size().plot(kind='bar')
    ax.set_title('# of transactions vs (type+isFraud)')
    ax.set_xlabel('(type,isFraud)')  # 好棒棒
    ax.set_ylabel('# of transaction')

    # ？？
    for p in ax.patches:
        ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height() * 1.01))
    print 'start printing'
    plt.show()
    print raw_data.groupby(['isFlaggedFraud','isFraud']).size()
    ax1=raw_data.groupby(['isFraud','isFlaggedFraud']).size().plot(kind='bar')
    ax1.set_title('isFlaggedFraud vs isFraud')
    ax1.set_xlabel('isFlagged')
    ax1.set_ylabel('isFraud')
    plt.show()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    transfer_data = raw_data[(raw_data['type'] == 'TRANSFER') | (raw_data['type'] == 'CASH_OUT')]
    # x y都是数据的column name
    a = sns.boxplot(x='isFraud', y='amount', data=transfer_data, ax=axs[0][0])
    # b = sns.regplot(x='oldbalanceOrg', y='amount', data=transfer_data[transfer_data['isFraud'] == 1],
    #                 ax=axs[0, 1])
    # c = sns.barplot(x='isFraud',y='isFlaggedFraud',data=transfer_data,ax=axs[1,0])
    plt.show()
    # 数据清洗
    # 保留cashout和transfer的
    used_data = raw_data[(raw_data['type'] == 'TRANSFER') | (raw_data['type'] == 'CASH_OUT')]
    # inplace 是指删除了几列以后把
    used_data.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
    used_data = used_data.reset_index(drop=True)
    print used_data.head()
    #index为false 就不会往里面写column name为空的一列index
    used_data.to_csv('D:\paysim1\\used_data.csv',index=False)
    print 'success'
    # 不想删，是傻了吧唧尝试的结果
    # reader = pd.read_csv(filepath, chunksize=100000, engine='python')
    # p = Pool(4)
    # result =None
    # count=0
    # for chunk in reader:
    #     # result1 = p.apply_async(concat, (chunk['type','isFlaggedFraud'],))
    #     # result.append(result1)
    #     # print item.index
    #     # count+=1
    #     # if count==1:
    #     #     result=chunk
    #     # else:
    #     #     result=pd.merge(result,chunk,on=['type','isFlaggedFraud'])
    #     # result=pd.concat([result,item],axis=0)
    #     result=pd.concat([result,chunk],ignore_index=True,axis=0)
    # print result.groupby(['type','isFlaggedFraud']).size()

    # p=Process(target=map_result,args=(reader,))
    # p.daemon=True
    # p.start()
    # p.join()
    # p.close()
    # p.join()
    print 'time', (time.time() - time3)
if __name__ == '__main__':
    #清洗数据，清洗后的数据重新保存至一个csv文件
    # cut_data()
    filepath='D:\paysim1\\used_data.csv'
    data=pd.read_csv(filepath)
    # ax=data['isFraud'].value_counts().plot(kind='bar')
    # plt.show()
    #encoder是变成连续数值 binarize是二值化
    type_label_encoder=preprocessing.LabelEncoder()
    type_category=type_label_encoder.fit_transform(data['type'])
    data['typeCategory']=type_category
    print data.head()
    #value是ndarray，上面fit_transform直接放入series就ok
    # print type(data['type'].values)

    #变量间的相关性
    sns.heatmap(data.corr(),cbar=True)
    # plt.show()
    #数据集不平衡
    #数据集下采样
    df_label_1 = data.loc[data.isFraud == 1, :]
    df_label_0 = data.loc[data.isFraud == 0, :]
    resample_n=df_label_1.shape[0]
    label_0_index = df_label_0.index
    # print label_0_index
    slice = random.sample(label_0_index, resample_n)
    print len(slice)
    df_samples = data.loc[slice, :]
    df_result = pd.concat([df_label_1, df_samples], axis=0, ignore_index=True)
    print df_result.shape, 'df_result.shape'
    #df是
    label= df_result['isFraud']
    feature = df_result.loc[:, 'amount':'typeCategory']
    from sklearn.feature_selection import SelectKBest
    from scipy.stats import pearsonr
    from numpy import array
    from sklearn.feature_selection import chi2
    from sklearn import model_selection
    print feature.shape
    Kbest=SelectKBest(chi2,k=5).fit_transform(feature,label)
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression()
    from sklearn.model_selection import cross_val_score
    # print cross_val_score(clf,feature,label,cv=5)
    from  sklearn import metrics
    X_train, X_test, y_train, y_test = model_selection.train_test_split(feature, label, test_size=0.3, random_state=0)
    clf.fit(X_train, y_train)
    predictions=clf.predict(X_test)
    print metrics.accuracy_score(y_test,predictions),'acc'
    print metrics.roc_auc_score(y_test,predictions),'roc_auc'
    #被预测正例中真正正例的比率
    #即预测出欺诈的案例中有多少是真正的欺诈者
    print metrics.precision_score(y_test,predictions),'precision'
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
    print fpr,tpr,thresholds
    

















