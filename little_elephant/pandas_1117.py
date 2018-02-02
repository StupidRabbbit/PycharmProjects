# coding=utf-8
import pandas as pd
import numpy as np

ser_obj=pd.Series(np.random.rand(8),index=[['a','a','a','a','b','b','b','b'],[1,2,3,4,1,2,3,4]])
print ser_obj

