# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TestScrapyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    top_cat=scrapy.Field()
    sub_cat = scrapy.Field()
    pass

class MeituanItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    top_cat=scrapy.Field()
    sub_cat = scrapy.Field()