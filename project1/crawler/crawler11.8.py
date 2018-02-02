#coding=utf-8
from bs4 import BeautifulSoup
import urllib2
import scrapy
import cookielib
#URL management
#use set to store links, keeping each link not duplicate
url='http://movie.douban.com/chart'
try:
    respond=urllib2.urlopen(url)
    bs_obj=BeautifulSoup(respond,'html.parser',from_encoding='utf-8')
    # print bs_obj.title

    link_list=bs_obj.find_all('a')
    for link in link_list:
        # print link.name,link['href'],link.get_text()
        # find content from css
        pass
    nav_items=bs_obj.find('div',{'class':'nav-items'})
    print nav_items.get_text()
    # for child in bs_obj.find('table',{'class':''}).children:
    #     print child


except Exception as e:
    print e


#webpage download
#webpage analysis