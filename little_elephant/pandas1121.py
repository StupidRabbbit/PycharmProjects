# coding=utf-8
import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.randint(0,10,(5,2)),columns=['data1','data2'])
print df,'origin'
series=df.stack()
print df.duplicated()