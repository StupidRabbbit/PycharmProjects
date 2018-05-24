# coding=utf-8
#这个模块是用来将数据不均衡的文件重采样的
#提供给component_attack模块
import pandas as pd
import numpy as numpy
import random
#复制正样本100到200 采样负样本到500
class Resample:
    def __init__(self,filepath):
        self.filepath=filepath
    def resample_action(self):
        df = pd.read_csv(self.filepath)
        # df_label=df['label']
        #label==1??跟人家标签有什么关系，你要的不是value吗
        #df_label_1 = df.loc['label'==1, :]
        df_label_1=df.loc[df.label==1,:]
        df_label_0 = df.loc[df.label == 0, :]
        label_0_index=df_label_0.index
        print label_0_index
        slice=random.sample(label_0_index,500)
        print len(slice)
        df_samples=df.loc[slice,:]
        df_result=pd.concat([df_label_1,df_samples,df_label_1],axis=0,ignore_index=True)
        # print df_result.head()
        return df_result
if __name__ == '__main__':
    filepath='D:\\result3.csv'
    handler=Resample(filepath)
    handler.resample_action()

