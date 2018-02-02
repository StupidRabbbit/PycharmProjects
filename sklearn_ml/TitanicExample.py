#-*- coding:utf-8 –*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#相当庞大的一个工程，这里只是做一个尝试

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train=pd.read_csv('D:\kaggle_data\Kaggle_Titanic-master\Kaggle_Titanic-master\\train.csv')
# print data_train.info()
# print data_train.describe()
fig=plt.figure()
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'survived(1 for surv)')
plt.ylabel(u'number')

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("number")
plt.title(u"passenger class distribution")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)#散点图
plt.ylabel(u'age')                         # 设定纵坐标名称
#plt.grid(b=True, which='major', axis='x')
plt.title(u"survive distribution from age (1 for surv)")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density")
plt.title(u"passenger's age distribution from each cls")
plt.legend((u'1-class', u'2-class',u'3-class'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"number from each port")
plt.ylabel(u"number")

# plt.show()


fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#每一个标签下，港口数据的情况,有三种S,C,Q
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
#df=pd.DataFrame({u'survived':Survived_1})
df.plot(kind='bar', stacked=True)
plt.title(u"survied situation")
plt.xlabel(u"port")
plt.ylabel(u"number")

# plt.show()
#以这两个分组，然后在分组中计算个数
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
# print df
#scikit-learn中的RandomForest来拟合一下缺失的年龄数据
# (注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，
# 再进行average等等来降低过拟合现象，提高结果的机器学习算法

from sklearn.ensemble import  RandomForestRegressor
#处理空缺值：回归算法拟合数据/聚类算法忽略分散数据
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df=df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age=age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y=known_age[:,0]
    # X即特征属性值
    X=known_age[:,1:]
    #n_estimators: integer, optional(default=10)The number oftrees in the forest.
    # n_jobs: integer, optional(default=1)The number ofjobs to run in parallel
    # for both `fit` and `predict`.If - 1, then the number of jobs is set to the number ofc ores.
    #random-state: the seed used for random number generator
    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    #Purely label-location based indexer for selection by label.
    df.loc[(df.Age.isnull()),'Age']=predictedAges


    return df,rfr
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
# some operations--fill the vacancy of age and Calbin
data_train,rfr=set_missing_ages(data_train)
data_train=set_Cabin_type(data_train)

# turn data to dummy matrix, set the prefix for each matrix column
# turn categorical features to 0and 1
dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis=1,inplace=True)
print '----------------------'
print df
#各属性值之间scale差距太大，将对收敛速度造成几万点伤害值！甚至不收敛！age&fare
import sklearn.preprocessing as preprocessing
#这块fit_transform的输入必须是2D的，有个fit，所以是二维的feature和sample（标签）
# ，如果只有一个feature或sample的话 需要将array.reshape(-1,1)变成二维的。
#不知道原文里面怎么想的 没有做转换也没报错？
scaler=preprocessing.StandardScaler()
#age_scale_param=scaler.fit(df['Age'])
df['Age_scaled']=scaler.fit_transform(df['Age'].reshape(-1,1))

#fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1,1))
# print df
# then comes to regression!!!
from sklearn import linear_model

train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#去掉了属性名称
train_np=train_df.as_matrix()
#print train_np
y=train_np[:,0]
X=train_np[:,1:]
#可调节参数？
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)
#print clf
#观察因数 因数很重要 看每一个属性 与结果的相关程度
#print pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})
from sklearn import cross_validation
from sklearn import model_selection
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X=all_data.as_matrix()[:,1:]
y=all_data.as_matrix()[:,0]
# print cross_validation.cross_val_score(clf,X,y)
# print model_selection.cross_val_score(clf,X,y)
# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train,split_cv=model_selection.train_test_split(df,test_size=0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions=clf.predict(cv_df.as_matrix()[:,1:])
#把错误的例子都摘出来分析
origin_data_train=pd.read_csv('D:\kaggle_data\Kaggle_Titanic-master\Kaggle_Titanic-master\\train.csv')
bad_cases=origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]['PassengerId'].values)]
#print bad_cases




















