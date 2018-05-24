# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import resample
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.svm import SVC


def component_attack(filepath):
    # df=pd.read_csv(filepath)

    #delete columns with no values at all
    #用重采样的方式我是把正负拼在一起的，这样就会导致前多少个是正/负样本，没法画学习曲线
    resampler=resample.Resample(filepath)
    df=resampler.resample_action()
    print 'df_s shape', df.shape
    # resampler.resample_action()

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
    # print df.dtypes

    df.drop(['permissionlevel'],axis=1,inplace=True)
    label = df.as_matrix()[:, 0]
    features = df.as_matrix()[:, 1:]
    X_train, X_test, y_train, y_test=model_selection.train_test_split(features,label,test_size=0.2,random_state=0)
    print type(features)
    #相关系数,这里就是用pearson相关，可以用来做特征选择
    print np.corrcoef(features)[0],'all corrcoef list'
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

    #查看分类情况
    print 'count 1,vulnerability samples',list(label).count(1)

    #定义模型
    regression_clf = LogisticRegression(C=0.35, penalty='l2', tol=1e-6)
    dct_clf=DecisionTreeClassifier(criterion='entropy',max_depth=9,min_samples_split=4,class_weight='balanced')
    knn_clf=KNeighborsClassifier(n_neighbors=3,algorithm='auto')
    nb_clf=MultinomialNB(alpha=0.1)
    ada_clf=AdaBoostClassifier(n_estimators=40,learning_rate=0.6)
    rf_clf=RandomForestClassifier(max_depth=21,n_estimators=20,min_samples_split=2)
    SVM_clf=SVC(C=1.0,class_weight='balanced')




    #定义超参数网格
    #评估标准为roc_auc可以减轻数据不平衡带来的伤害
    grid_knn=GridSearchCV(knn_clf,param_grid={'n_neighbors':range(1,10,1)},cv=5,scoring='roc_auc')
    grid_SVM=GridSearchCV(SVM_clf,param_grid={'C':np.arange(0.01,1,0.01)},cv=5,scoring='roc_auc')
    # regression_clf.fit(X_train, y_train)
    # grid_SVM.fit(X_train, y_train)
    # nb_clf.fit(X_train, y_train)
    # knn_clf.fit(X_train, y_train)
    #
    dct_clf.fit(X_train, y_train)
    SVM_clf.fit(X_train, y_train)
    # ada_clf.fit(X_train, y_train)
    # rf_clf.fit(X_train, y_train)

    grid_logisticR=GridSearchCV(regression_clf,
                                param_grid={'C':np.arange(0.01,1,0.01),'penalty':['l2']},cv=5,scoring='roc_auc')
    grid_DCT = GridSearchCV(dct_clf,
                                param_grid={'max_depth':np.arange(1,25,1),'criterion':np.array(['entropy','gini']),'min_samples_split':np.arange(2,5)},cv=5,scoring='roc_auc')

    grid_NB = GridSearchCV(nb_clf, param_grid={'alpha': np.arange(0.1,1,0.1)}, cv=5, scoring='roc_auc')
    #训练数据
    # grid_logisticA = GridSearchCV(ada_clf, param_grid={'n_estimators':np.arange(30,50,5),'learning_rate':np.arange(0.1,1,0.1)}, cv=5, scoring='roc_auc')
    grid_RF = GridSearchCV(rf_clf, param_grid={'n_estimators':np.arange(5,30,5),'max_depth':np.arange(1,25,1),'min_samples_split':np.arange(2,5)}, cv=5, scoring='roc_auc')
    # grid_DCT.fit(X_train,y_train)
    # grid_knn.fit(X_train,y_train)
    # grid_logisticR.fit(X_train,y_train)
    # grid_NB.fit(X_train,y_train)
    # grid_logisticA.fit(X_train, y_train)
    # grid_RF.fit(X_train, y_train)
    #预测数据
    predictions_rf=SVM_clf.predict(X_test)
    #网格搜索结果，评估
    # print  'SVM grid_bestparam',grid_SVM.best_params_,'SVM grid_best ',grid_SVM.best_score_
    # print 'KNN grid_bestparam',grid_knn.best_params_,'KNN grid_best',grid_knn.best_score_
    # print 'logisticR grid_bestparam',grid_logisticR.best_params_,'logisticR grid_bestscore',grid_logisticR.best_score_
    # print 'DCT grid_bestparam', grid_DCT.best_params_, 'DCT grid_best', grid_DCT.best_score_
    # print 'NB grid_bestparam', grid_NB.best_params_, 'NB grid_best', grid_NB.best_score_
    # print 'Ada best_param',grid_logisticA.best_params_,grid_logisticA.best_score_
    # print 'Random best_param', grid_RF.best_params_, grid_RF.best_score_
    # print type(features)
    # print type(label)
    # print 'knn score',model_selection.cross_val_score(knn_clf,features,label,cv=5)
    # print 'regression score',model_selection.cross_val_score(regression_clf,features,label,cv=5)
    # print 'adaboost score',model_selection.cross_val_score(ada_clf,features,label)
    # print 'MultinomialNB score',model_selection.cross_val_score(nb_clf,features,label,cv=5)

    #计算错误率
    test_nd = np.array(y_test)
    # print abs(predictions_reg - test_nd)
    # numpy可以直接做数学运算
    # error = sum(abs(predictions_reg - test_nd)) / len(test_nd)
    #
    # count1=list(test_nd).count(1)
    # count0=list(test_nd).count(0)
    # print count0,count1
    # i1,j0=0,0
    # for i in range(0,len(predictions_reg)):
    #     if predictions_reg[i]==1 and test_nd[i]==1:
    #         i1+=1
    #     if predictions_reg[i]==1 and test_nd[i]==0:
    #         j0+=1
    #
    # print j0,i1
    # print 'FP:',float(j0)/float(count0)
    # print 'TP:',float(i1)/float(count1)
    #计算各种评估条件
    print metrics.accuracy_score(y_test,predictions_rf), 'accuracy rate'
    print metrics.roc_auc_score(y_test,predictions_rf),'roc_auc_score'
    
    fpr, tpr, thresholds=metrics.roc_curve(y_test,predictions_rf)
    print fpr,tpr
    #画学习曲线

    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    # 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                            train_sizes=np.linspace(.05, 1., 30), verbose=0, plot=True):
        """
        画出data在某模型上的learning curve.
        参数解释
        ----------
        estimator : 你用的分类器。
        title : 表格的标题。
        X : 输入的feature，numpy类型
        y : 输入的target vector
        ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
        cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        n_jobs : 并行的的任务数(默认1)
        """
        print 'start to print...'
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if plot:
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel('train samples')
            plt.ylabel('scores')
            #翻转y轴，虽然不知道为啥要翻转，这样翻转以后最大值到下面去了
            # plt.gca().invert_yaxis()
            plt.grid()
            #填充区间浅红色和浅蓝色
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.1, color="b")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                             alpha=0.1, color="r")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label='train score')
            plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label='test score')
            plt.legend(loc="best")
            # plt.draw()
            plt.show()
            # plt.gca().invert_yaxis()

        midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
        diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
        return midpoint, diff

    # plot_learning_curve(regression_clf, 'learning curve', features, label,cv=3)

if __name__ == '__main__':
    #wait for a dataset
    filepath='D:\\result3.csv'
    component_attack(filepath)