# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.decomposition import PCA

def component_attack(filepath):
    df=pd.read_csv(filepath)
    print 'df_s shape',df.shape
    #delete columns with no values at all

    if df.isnull().values.any():
        df=df.dropna(axis=1,how='all')
        df=df.fillna(0.)
    print df.shape



    # print type(df['intentfilter']),df.dtypes
    df['intentfilter']=df['intentfilter'].astype('int')
    # print df['intentfilter']

    df['exported']=df['exported'].astype('str')
    df['exported']=df['exported'].replace({'FALSE':0,'TRUE':1,'None':0})
    # print df['exported']
    dummies_permission=pd.get_dummies(df['permissionlevel'],prefix='permission')
    df=pd.concat([df,dummies_permission],axis=1)
    print df.dtypes

    df.drop(['permissionlevel'],axis=1,inplace=True)
    label = df.as_matrix()[:, 0]
    features = df.as_matrix()[:, 1:]
    print type(features)
    #相关系数
    print np.corrcoef(features)[0]
    corrcoef=list(np.corrcoef(features)[0])
    dict_corr={}
    for i in corrcoef:
        if i>0.8:
            dict_corr[i]=corrcoef.index(i)
    print dict_corr
    # #pca降维
    # pca=PCA(n_components=10,copy=True)
    # features_pca=pca.fit_transform(features)
    # print features_pca[0]
    # print pca.n_components,'n_components'



    #定义模型
    regression_clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    knn_clf=KNeighborsClassifier(n_neighbors=5,algorithm='auto')
    nb_clf = MultinomialNB()

    #定义超参数网格
    grid_knn=GridSearchCV(knn_clf,param_grid={'n_neighbors':range(1,10)},cv=5,scoring='roc_auc')
    grid_logisticR=GridSearchCV(regression_clf,param_grid={'C':np.arange(0.1,2,0.3),'penalty':['l1','l2']},cv=5,scoring='roc_auc')

    #训练数据
    grid_knn.fit(features,label)
    grid_logisticR.fit(features,label)

    #网格搜索结果，评估
    print 'KNN grid_bestparam',grid_knn.best_params_,'KNN grid_best',grid_knn.best_score_
    print 'logisticR grid_bestparam',grid_logisticR.best_params_,'logisticR grid_bestscore',grid_logisticR.best_score_

    # print type(features)
    # print type(label)
    # print 'knn score',model_selection.cross_val_score(knn_clf,features,label,cv=5)
    # print 'regression score',model_selection.cross_val_score(regression_clf,features,label,cv=5)
    # print 'MultinomialNB score',model_selection.cross_val_score(nb_clf,features,label,cv=5)

if __name__ == '__main__':
    #wait for a dataset
    filepath='C:\Users\Administrator\Documents\Tencent Files\\595213784\FileRecv\\result2.csv'
    component_attack(filepath)