# coding=utf-8
#lecture 05
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from pandas_tools import inspect_dataset
def inspect_dataset(filepath):
   df_data=pd.read_csv(filepath)
   # print df_data.info()
   # print df_data.shape
   return df_data
def process_missing_data(df_data):
    if df_data.isnull().values.any():
        # print type(df_data.isnull().values)
        df_data.fillna(0.)

        
def visualize_league_attribute_status(df_data, attr_label,
                                     ):
    league_idx_list=range(1,9)
    stats_min=[]
    stats_max=[]
    stats_mean=[]

    for index in league_idx_list:
        filtered_data=df_data.loc[index==df_data['LeagueIndex'],attr_label]
        stats_min.append(filtered_data.min())
        stats_max.append(filtered_data.max())
        stats_mean.append(filtered_data.mean())

    league_ser=pd.Series(league_idx_list,name='LeagueIndex')
    stats_min_ser=pd.Series(stats_min,name='min')
    stats_max_ser = pd.Series(stats_max, name='max')
    stats_mean_ser = pd.Series(stats_mean, name='mean')

    stats_data=pd.concat([league_ser,stats_min_ser,stats_max_ser,stats_max_ser,stats_mean_ser],axis=1)

    print stats_data

    fig=plt.figure(figsize=(10.0,10.0))
    # axs=fig.add_subplot(1,1,1)
    plt.xlabel(u'战队')
    plt.title(u'APM statics')
    plt.plot(stats_data['LeagueIndex'],stats_data['min'],color='green')
    plt.plot(stats_data['LeagueIndex'], stats_data['max'], color='red')
    plt.plot(stats_data['LeagueIndex'], stats_data['mean'], color='blue')
    blue_patch = mpatches.Patch(color='blue', label='Average ' + attr_label)
    green_patch = mpatches.Patch(color='green', label='Min ' + attr_label)
    red_patch = mpatches.Patch(color='red', label='Max ' + attr_label)
    plt.legend(handles=[blue_patch,red_patch,green_patch])
    plt.show()

    return stats_data






if __name__ == '__main__':
    filepath = 'D:\little_elephant\lecture05_codes\codes\lecture05_proj\dataset\\starcraft.csv'
    df_data=inspect_dataset(filepath)
    process_missing_data(df_data)
    column_names=['LeagueIndex','APM']
    # print df_data[column_names]
    status_data=visualize_league_attribute_status(df_data[column_names],'APM')

