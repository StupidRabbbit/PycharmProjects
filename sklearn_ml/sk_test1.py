#-*- coding:utf-8 –*-
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
from numpy import vstack,array,nan,hstack,median
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import  SelectKBest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.datasets import load_iris
from numpy.random import choice
from numpy import log1p
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#includes 4 features
iris=load_iris()
#feature matrix
print iris.data

#target vector
print iris.target
print '-----------'

size=iris.data.shape[0]

class FeatureUnionExt(FeatureUnion):
    #相比FeatureUnion，多了idx_list参数，其表示每个并行工作需要读取的特征矩阵的列
    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self, transformer_list=map(lambda trans:(trans[0], trans[1]), transformer_list), n_jobs=n_jobs, transformer_weights=transformer_weights)

    #由于只部分读取特征矩阵，方法fit需要重构
    def fit(self, X, y=None):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit方法
            delayed(_fit_one_transformer)(trans, X[:,idx], y)
            for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    #由于只部分读取特征矩阵，方法fit_transform需要重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit_transform方法
            delayed(_fit_transform_one)(trans, name, X[:,idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    #由于只部分读取特征矩阵，方法transform需要重构
    def transform(self, X):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入transform方法
            delayed(_transform_one)(trans, name, X[:,idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs


# StandardScaler().fit_transform(iris.data)
#
# MinMaxScaler().fit_transform(iris.data)
#a=VarianceThreshold(threshold=3).fit_transform(iris.data)
# the result is 3rd col of matrix
a = np.array([1, 2, 3, 4])

# print a
# b=np.linspace(1,3,4)
# print b
# print vstack((a,b))
# d=a.reshape(-1,1)
#
# print d
#a=SelectKBest(lambda X, Y: tuple(map(tuple,array(list(map(lambda x:pearsonr(x, Y), X.T))).T)), k=2).fit_transform(iris.data, iris.target)
#a=SelectKBest(chi2,k=2).fit_transform(iris.data,iris.target)
#the result is 2nd and 3rd col

#a=PCA(n_components=2).fit_transform(iris.data)

a=(choice([0,1,2],size=iris.data.shape[0]+1)).reshape(-1,1)
#vstack : Stack arrays in sequence vertically (row wise).
b=vstack((iris.data,array([nan,nan,nan,nan]).reshape(1,-1)))
#Stack arrays in sequence horizontally (column wise).
iris.data=hstack((a,b))
#print hstack((a,b))
iris.target=hstack((iris.target,array([median(iris.target)])))
print iris.target

#Feature union--partial parallel processing
# 新建计算缺失值的对象
step1=('Imputer',Imputer())
#新建将部分特征矩阵进行定性特征编码的对象
step2_1=('OneHotEncoder',OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
step2_2=('ToLog',FunctionTransformer(log1p))
#新建将部分特征矩阵进行二值化类的对象
step2_3=('ToBinary',Binarizer())
#新建部分并行处理对象，返回值为每个并行工作的输出的合并
step2=('FeatureUnion',FeatureUnionExt(transformer_list=[step2_1,step2_2,step2_3],idx_list=[[0],[1,2,3],[4]]))
#新建无量纲化对象
step3=('MinMaxScaler',MinMaxScaler())
#新建卡方校验选择特征的对象
step4 = ('SelectKBest', SelectKBest(chi2, k=3))
#新建PCA降维的对象
step5 = ('PCA', PCA(n_components=2))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#新建流水线处理对象
# 参数steps为需要水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象,前面引号里面就是名称
pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])

from sklearn.grid_search import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid={'FeatureUnionExt__ToBinary__threshold': [1.0, 2.0, 3.0, 4.0],
                                                 'LogisticRegression__C': [0.1, 0.2, 0.4, 0.8]})
grid_search.fit(iris.data, iris.target)





