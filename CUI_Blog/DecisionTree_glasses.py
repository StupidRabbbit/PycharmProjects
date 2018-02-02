# coding=utf-8
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


if __name__ == '__main__':
    filepath='D:\CUIblog\Machine-Learning-master\Machine-Learning-master\Decision Tree\\lenses.txt'
    f=open(filepath)
    lenses=[line.strip().split('\t') for line in f.readlines()]
    # print lenses
    label=[line[-1] for line in lenses]
    # print label
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list=[]
    lenses_dict={}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label]=lenses_list
        lenses_list=[]
    lenses_pd=pd.DataFrame(lenses_dict)
    print lenses_pd

    transformer=LabelEncoder()

    for column in lenses_pd.columns:
        lenses_pd[column]=transformer.fit_transform(lenses_pd[column])
    print lenses_pd


    clf=tree.DecisionTreeClassifier(max_depth=4)
    clf=clf.fit(lenses_pd.as_matrix(),label)

    print clf.predict([[1,1,1,0]])

    import matplotlib.pyplot as plt
    pd.cut()



