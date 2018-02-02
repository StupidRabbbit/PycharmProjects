# coding=utf-8
import pandas as pd

def linear_forcast(filepath):
    df=pd.read_csv(filepath);
    df=df.iloc[::-1]
    rows=df.shape[0]
    n=rows-10
    mstart=n-
if __name__ == '__main__':
    filepath='D:\stock_data\stock\HistoryQuotations\Day\\000001.csv'
    linear_forcast(filepath)