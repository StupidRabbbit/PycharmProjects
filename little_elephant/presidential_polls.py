#-*- coding:utf-8 –*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import datetime
def is_convert_float(s):
    """
                判断一个字符串能否转换为float
        """
    try:
        float(s)
    except:
        return  False
    return True

def get_sum(str_array):
    """
             返回字符串数组中数字的总和
     """
    # 去掉不能转换成数字的数据
    #filter(delete some):apply function filer(the 1st param) to a iterator object (2nd param)
    cleaned_data=filter(is_convert_float,str_array)
    # 转换数据类型
    float_array=np.array(cleaned_data,np.float)
    return np.sum(float_array)



def run_main():
    filename='D:\kaggle_data\python_analysis\lecture02_codes\codes\\presidential_polls.csv'
#step1 column name preprocessing
#read the 1st column data, column name
    with open(filename,'r') as f:
        col_names_str=f.readline()[:-1]#不读取末尾换行符

    col_names_list=col_names_str.split(',')

    # 使用的列名
    use_col_name_lst = ['enddate', 'rawpoll_clinton', 'rawpoll_trump', 'adjpoll_clinton', 'adjpoll_trump']
    # 获取相应列名的索引号
    #use_col_name are the columns's names we want
    use_col_idx_list=[col_names_list.index(use_col_name) for use_col_name in use_col_name_lst]
    #读取数据
    data_array=np.loadtxt(filename,
                          delimiter=',',# 分隔符
                          dtype=str, # 数据类型
                          skiprows=1,# 跳过第一行，即跳过列名
                          usecols=use_col_idx_list)# 指定读取的列索引号
    ## Step3. 数据处理
    # 处理日期格式数据
    enddate_idx=use_col_name_lst.index('enddate')
    #1D array to list
    enddate_lst=data_array[:,enddate_idx].tolist()
    enddate_lst=[enddate.replace('-','/') for enddate in enddate_lst]
    date_lst=[datetime.datetime.strptime(enddate,'%m/%d/%Y') for enddate in enddate_lst]
    #print date_lst[:10]
    # 构造年份-月份列表
    month_lst=['%d-%02d' %(date_obj.year,date_obj.month) for date_obj in date_lst]
    #print month_lst
    month_array=np.array(month_lst)
    months=np.unique(month_array)
    print months

    ## Step4. 数据分析
    # 统计民意投票数
    # cliton
    # 原始数据 rawpoll
    rawpoll_clinton_idx=use_col_name_lst.index('rawpoll_clinton')
    rawpoll_clinton_data=data_array[:,rawpoll_clinton_idx]
    # 调整后的数据 adjpool
    adjpoll_clinton_idx = use_col_name_lst.index('adjpoll_clinton')
    adjpoll_clinton_data = data_array[:, adjpoll_clinton_idx]

    # trump
    # 原始数据 rawpoll
    rawpoll_trump_idx = use_col_name_lst.index('rawpoll_trump')
    rawpoll_trump_data = data_array[:, rawpoll_trump_idx]

    # 调整后的数据 adjpoll
    adjpoll_trump_idx = use_col_name_lst.index('adjpoll_trump')
    adjpoll_trump_data = data_array[:, adjpoll_trump_idx]

    results=[]
    # conditional index
    #month_array is in the order of every split data_array,the index of single month is
    #also the index of data which is in that very month
    # so you can locate the data in one specific month through this condition index
    for month in months:
        rawpoll_clinton_month_data=rawpoll_clinton_data[month_array==month]
        rawpoll_clinton_month_sum=get_sum(rawpoll_clinton_month_data)

        # 调整数据 adjpoll
        adjpoll_clinton_month_data = adjpoll_clinton_data[month_array == month]
        # 统计当月的总票数
        adjpoll_clinton_month_sum = get_sum(adjpoll_clinton_month_data)

        # trump
        # 原始数据 rawpoll
        rawpoll_trump_month_data = rawpoll_trump_data[month_array == month]
        # 统计当月的总票数
        rawpoll_trump_month_sum = get_sum(rawpoll_trump_month_data)

        # 调整数据 adjpoll
        adjpoll_trump_month_data = adjpoll_trump_data[month_array == month]
        # 统计当月的总票数
        adjpoll_trump_month_sum = get_sum(adjpoll_trump_month_data)

        results.append((month,rawpoll_clinton_month_sum,adjpoll_clinton_month_sum,rawpoll_trump_month_sum,adjpoll_trump_month_sum))
    #print results
    #split results into tuples,return a list of tuples, attention!every return value is a tuple!
    # it will return you a tuple which is a element sequence of the same order with how you input
    months, raw_cliton_sum, adj_cliton_sum, raw_trump_sum, adj_trump_sum=zip(*results)
    print raw_cliton_sum
    #print months
    #visualization
    import matplotlib.pyplot as plt
    #return fig and axes
    fig,subplot_arr=plt.subplots(2,2,figsize=(15,10))
    subplot_arr[0,0].plot(raw_cliton_sum,'r')
    subplot_arr[0,0].plot(raw_trump_sum, 'g')
    width=0.25
    #x is horizontal coordinate
    x=np.arange(len(months))

    subplot_arr[0,1].bar(x,raw_cliton_sum,width,color='r')
    subplot_arr[0,1].bar(x+width, raw_trump_sum, width, color='g')
    #ticks?ticks? tips?
    subplot_arr[0,1].set_xticks(x+width)
    subplot_arr[0,1].set_xticklabels(months,rotation='vertical')

    # 调整数据趋势展示
    subplot_arr[1, 0].plot(adj_cliton_sum, color='r')
    subplot_arr[1, 0].plot(adj_trump_sum, color='g')

    width = 0.25
    x = np.arange(len(months))
    subplot_arr[1, 1].bar(x, adj_cliton_sum, width, color='r')
    subplot_arr[1, 1].bar(x + width, adj_trump_sum, width, color='g')
    subplot_arr[1, 1].set_xticks(x + width)
    subplot_arr[1, 1].set_xticklabels(months, rotation='vertical')


    plt.show()







if __name__ == '__main__':
    run_main()


