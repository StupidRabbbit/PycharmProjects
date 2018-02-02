# coding=utf-8
import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.rand(3,5))
df.index=pd.date_range('2018/1/2',periods=df.shape[0])
print df
col=list('ABCDE')
print col
df.columns=col
# print df['C']
print df.loc['2018/1/2']
print df.iloc[0]
print '---------------------------------'
# print df[(df['A']>0.5) & (df['C']<0.5)]
# print df.sort_values(['A'],ascending=[False])
# print df.sort_values(['A','C'],ascending=[False,True])
print df.mean(axis=1)

df1 = pd.DataFrame({'A':np.array(['foo','foo','foo','foo','bar','bar']),
      'B':np.array(['one','one','two','two','three','three']),
     'C':np.array(['small','medium','large','large','small','small']),
     'D':np.array([1,2,2,3,3,5])})

print df1.groupby('C').sum()
print df.apply(np.mean)
print df.apply(np.mean,axis=1)

df2=pd.Series({'A':'foo','B':'bar'})
print df2







year=2061
def operation(x):
 year = x- 100 if x> 1989 else x
 return year
print operation(year)



