# -*- coding: utf-8 -*-
import scrapy


class MeituanSpider(scrapy.Spider):
    name = 'meituan_spider'
    allowed_domains = ['bj.meituan.com/meishi/']
    start_urls = ['http://bj.meituan.com/meishi/']

    def parse(self, response):
        pass
