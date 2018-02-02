# -*- coding: utf-8 -*-

'''
Created on Dec 8, 2016

@author: Bin Liang
'''
import pandas as pd
import matplotlib.pyplot as plt
from pandas_tools import inspect_dataset, process_missing_data, analyze_gross,\
    get_genres_data

dataset_path = './dataset/movie_metadata.csv'


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
    
    ## Step.3 使用分组统计数据集的基本信息
    ## Step.3.1 查看票房收入统计 (可传入多个列名进行分析)
    # 导演vs票房总收入
    analyze_gross(df_data, ['director_name'], './output/director_gross.csv')
    
    # 主演vs票房总收入
    analyze_gross(df_data, ['actor_1_name'], './output/actor_gross.csv')
    
    # 导演+主演vs票房收入
    analyze_gross(df_data, ['director_name', 'actor_1_name'], './output/director_actor_gross.csv')
     
    # Step.3.2 查看imdb评分统计
    # 查看各imdb评分的电影个数
    df_ratings = df_data.groupby('imdb_score')['movie_title'].count()
    plt.figure()
    df_ratings.plot()
    plt.savefig('./output/imdb_scores.png')
    plt.show()
     
    # 查看top20导演的平均imdb评分
    df_director_mean_ratings = df_data.groupby('director_name')['imdb_score'].mean()
    top20_imdb_directors = df_director_mean_ratings.sort_values(ascending=False)[:20]
    plt.figure(figsize=(18.0, 10.0))
    top20_imdb_directors.plot(kind='barh')
    plt.savefig('./output/top20_imdb_directors.png')
    plt.show()
    
    # Step.3.3 电影产量趋势
    df_movie_years = df_data.groupby('title_year')['movie_title'].count()
    plt.figure()
    df_movie_years.plot()
    plt.savefig('./output/movie_years.png')
    plt.show()
    
    # Step.4 电影类型分析
    # 电影类型个数统计
    df_genres = get_genres_data(df_data)
    genres_count = df_genres.groupby('genre').size()
    plt.figure(figsize=(15.0, 10.0))
    genres_count.plot(kind='barh')
    plt.savefig('./output/genres_count.png')
    plt.show()
    
    # 电影类型票房统计
    genres_gross = df_genres.groupby('genre')['gross'].sum()
    plt.figure(figsize=(15.0, 10.0))
    genres_gross.plot(kind='barh')
    plt.savefig('./output/genres_gross.png')
    plt.show()
    
    
if __name__ == '__main__':
    run_main()