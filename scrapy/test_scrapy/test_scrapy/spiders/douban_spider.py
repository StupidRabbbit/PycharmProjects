# -*- coding: utf-8 -*-
import scrapy


class DoubanSpider(scrapy.Spider):
    name = 'douban_spider'
    allowed_domains = ['movie.douban.com']
    start_urls = ['https://movie.douban.com/chart']
    #callback
    def parse(self, response):
        filepath='page.html'
        with open(filepath,'w') as f:
            f.write(response.body)

