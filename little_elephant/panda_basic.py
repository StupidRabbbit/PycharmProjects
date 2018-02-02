# coding=utf-8
import pandas as pd
import numpy as np
ser_obj=pd.Series(range(10,20))
# print type(ser_obj)
#
# print ser_obj.head(3)
# print ser_obj.values
# print type(ser_obj.index)
# print ser_obj.name
array=np.random.randn(5,4)
df_obj=pd.DataFrame(array,columns=['a','b','c','d'])
print df_obj

a_obj=df_obj.apply(lambda x:x.max())
print a_obj
print df_obj.sort_index(ascending=False)
