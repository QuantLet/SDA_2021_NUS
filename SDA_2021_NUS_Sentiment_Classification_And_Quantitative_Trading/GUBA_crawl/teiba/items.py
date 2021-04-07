# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ArticleItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    item_id = scrapy.Field()
    stock_id = scrapy.Field()
    news_id = scrapy.Field()
    page = scrapy.Field()
    timestamp = scrapy.Field()    
    post_article = scrapy.Field()
    reply_list = scrapy.Field()
    is_test = scrapy.Field()
    is_fake = scrapy.Field()
    comment_count = scrapy.Field()
    
    # exception = scrapy.Field()

class PageItem(scrapy.Item):
    url = scrapy.Field()