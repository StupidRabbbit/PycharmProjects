# -*- coding: utf-8 -*-

'''
Created on Dec 5, 2016

@author: Bin Liang
'''
import pandas as pd
from pandas_tools import inspect_dataset, visualize_league_attributes,\
    visualize_league_attribute_stats, process_missing_data

dataset_path = './dataset/starcraft.csv'


def run_main():
    """
            主函数
    """
    
    ## Step.0 加载数据
    df_data = pd.read_csv(dataset_path)
    
    ## Step.1 查看数据
    inspect_dataset(df_data)
    
    ## Step.2 处理缺失数据
    df_data = process_missing_data(df_data)
    
    ## Step.3.1 可视化战队属性，这里选4个属性作为例子展示
    column_names = ['LeagueIndex',  # 战队索引号
                    'HoursPerWeek', # 每周游戏时间
                    'Age',          # 战队中玩家的年龄
                    'APM',          # 手速
                    'WorkersMade'   # 单位时间的建造数
                    ]
    visualize_league_attributes(df_data[column_names])
    
    ## Step3.2 可视化战队属性统计值
    visualize_league_attribute_stats(df_data[column_names], 
                                     'APM',
                                     savedata_path='./league_apm_stats.csv',
                                     savefig_path='./league_apm_stats.png',)
    
    visualize_league_attribute_stats(df_data[column_names], 
                                     'HoursPerWeek',
                                     savedata_path='./league_hrs_stats.csv',
                                     savefig_path='./league_hrs_stats.png',)
    

if __name__ == '__main__':
    run_main()
