#coding=utf-8
from bs4 import BeautifulSoup
import urllib2
import scrapy
import cookielib
#URL management
#use set to store links, keeping each link not duplicate
url='http://stockpage.10jqka.com.cn/HQ_v3.html#hs_000703'
try:
    respond=urllib2.urlopen(url)
    bs_obj=BeautifulSoup(respond,'html.parser',from_encoding='utf-8')
    print bs_obj
    print bs_obj.title

    # link_list=bs_obj.find_all('a')
    # for link in link_list:
    #     # print link.name,link['href'],link.get_text()
    #     # find content from css
    #     pass
    nav_items=bs_obj.find('div',{'id':'zjlxTip'})
    print nav_items
    print nav_items.get_text()
    # for child in bs_obj.find('table',{'class':''}).children:
    #     print child


except Exception as e:
    print e

