# -*- coding: utf-8 -*-

'''
Created on Dec 5, 2016

@author: Bin Liang
'''
import pandas as pd
import os

def inspect_dataset(df_data):
    """
            查看加载的数据基本信息
    """
    print '数据集基本信息：'
    print df_data.info()
    
    print '数据集有%i行，%i列' %(df_data.shape[0], df_data.shape[1])
    print '数据预览:'
    print df_data.head()


def process_missing_data(df_data):
    """
            处理缺失数据
    """
    if df_data.isnull().values.any():
        # 存在缺失数据
#         df_data = df_data.fillna(0.)    # 填充nan
        df_data = df_data.dropna()    # 过滤nan
    return df_data.reset_index()


def analyze_gross(df_data, groupby_columns, csvfile_path):
    """
            分析票房数据并保存结果
            可尝试将该方法进行扩展，如analyze_imdb等
    """
    grouped_data = df_data.groupby(groupby_columns, as_index=False)['gross'].sum()
    sorted_grouped_data = grouped_data.sort_values(by='gross', ascending=False)
    sorted_grouped_data.to_csv(csvfile_path, index=None)
    

def get_genres_data(df_data):
    """
            重新构造基于电影类型的数据集
    """
    genre_data_path = './output/genre_data.csv'
    if os.path.exists(genre_data_path):
        print '读取电影类型数据...'
        df_genre = pd.read_csv(genre_data_path)
    else:
        print '生成电影类型数据...'
        df_genre = pd.DataFrame(columns = ['genre', 'budget', 'gross', 'year'])
        
         
        
        df_genre.to_csv('./output/genre_data.csv', index=None)
    return df_genre
    
    
def convert_row_to_df(row_data):
    """
                用于将每行数据重构成dataframe
    """
    genres = row_data['genres'].split('|')
    n_genres = len(genres)
    
    dict_obj = {}
    dict_obj['budget'] = [row_data['budget']] * n_genres
    dict_obj['gross'] = [row_data['gross']] * n_genres
    dict_obj['year'] = [row_data['title_year']] * n_genres
    dict_obj['genre'] = genres
    
    return pd.DataFrame(dict_obj)