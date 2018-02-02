# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import os.path
def load_data(filepath):
    df_data=pd.read_csv(filepath)
    # print df_data.shape
    # print df_data.info()
    df_data = df_data.dropna()
    return df_data
def process(df_data):
    df_data=df_data.dropna()
    return df_data
def analyze_gross(df_data,groupby_columns,save_path):
    grouped_data=df_data.groupby(groupby_columns,as_index=False)['gross'].sum()
    # print grouped_data
    sorted_data=grouped_data.sort_values(by='gross',ascending=False)
    # print sorted_data
    sorted_data.to_csv(save_path,index=None)
def get_genres_data(df):
    genres_path='D:\little_elephant\lecture06_codes\codes\lecture06_proj\dataset\genre_data.csv'
    if os.path.exists(genres_path):
        df_genre=pd.read_csv(genres_path)
    else:
        df_genre=pd.DataFrame(columns=['genre','budget','gross','year'])

        for i,row_data in df.iterrows():
            if (i+1)%100 == 0:
                print '共%i条记录，已处理%i' %(df.shape[0], i+1)
            df_genre_df=convert_row_to_df(row_data)
            df_genre=df_genre.append(df_genre_df,ignore_index=True)
        df_genre.to_csv(genres_path,index=None)
    return df_genre

def convert_row_to_df(row_data):
    genres_split=row_data['genres'].split('|')
    # print genres_split
    n=len(genres_split)
    dict={}
    dict['budget']=[row_data['budget']]*n
    dict['gross'] = [row_data['gross']] * n
    dict['year']=[row_data['title_year']]*n
    dict['genre']=genres_split
    return pd.DataFrame(dict)

if __name__ == '__main__':
    filepath='D:\little_elephant\lecture06_codes\codes\lecture06_proj\dataset\movie_metadata.csv'
    df=load_data(filepath)
    df=process(df)
    # analyze_gross(df,['director_name','actor_1_name'],'D:\little_elephant\lecture06_codes\codes\lecture06_proj\dataset\gross.csv')
    # grouped_df=df.groupby('director_name')['imdb_score'].mean()
    # top_20_sorted_df=grouped_df.sort_values(ascending=False)[:20]
    # plt.figure(figsize=(18,18))
    # top_20_sorted_df.plot(kind='bar')
    get_genres_data(df)
