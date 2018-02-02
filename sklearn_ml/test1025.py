#-*- coding:utf-8 –*-

from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
#
#
# A=StandardScaler().fit_transform(X)
# print A
# #L1 norm则是变换后每个样本的各维特征的绝对值和为1
# B=normalize(X,norm='l1')
# #0.4^2+0.4^2+0.81^2=1,这就是L2 norm，变换后每个样本的各维特征的平方和为1
# C=normalize(X,norm='l2')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# iris=load_iris()
# X,y=iris.data,iris.target
# X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

# #数据可视化
# from sklearn.datasets import make_classification
# #使用make_classification构造100个样本，每个样本有20个feature
# X,y=make_classification(100,n_features=20,n_informative=2,n_redundant=2,n_classes=2,random_state=0)
#
# #存为dataframe格式
# from pandas import DataFrame
# df=DataFrame(np.hstack((X,y[:,None])),columns = range(20) + ["class"])
# # print df[:6]
# import seaborn as sns
# figure1=sns.pairplot(df[:50],vars=[1,2,3,4,5],hue='class',size=1.5)
# plt.show()
#
# plt.figure(figsize=(12, 10))
# figure2 = sns.corrplot(df, annot=False)
# plt.show()
A=[[1,2,3],[4,5,6],[7,8,9]]

A=array(A)
print A[:,1::]
