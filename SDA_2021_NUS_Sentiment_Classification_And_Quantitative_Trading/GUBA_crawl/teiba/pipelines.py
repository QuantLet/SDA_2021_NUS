# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
import jsonlines
import os
import pymongo

import logging
logger = logging.getLogger(__name__)

class TeibaPipeline(object):
    def process_item(self, item, spider):
        return item

# class JsonlinePipeline(object):
#     def open_spider(self, spider):
#         db_dir = r'C:\Users\ASUS\Desktop\teiba\teiba\db'
#         # stock_set = set(file.split('.')[0] for file in os.listdir(db_dir))
#         # stock_set = set(file for file in os.listdir(db_dir))
#         self.db_dir = db_dir
#         # self.stock_set = stock_set
#         # self.file = open(r'C:\Users\ASUS\Desktop\HotelData\Hoteldata\database.jsonl','a',encoding='utf-8')

#     def close_spider(self, spider):
#         # self.file.close()
#         pass

#     @staticmethod
#     def parse_item_id(item_id):
#         type_, stock_id, news_id = item_id.split(',')
#         new_id, page = news_id.split('_')
#         return type_, stock_id, new_id, page

#     def process_item(self, item, spider):
#         item_id = item.get('item_id')
#         if item_id:
#             type_, stock_id, new_id, page = self.parse_item_id(item_id)
#             stock_path = os.path.join(self.db_dir,stock_id)
#             new_path = os.path.join(stock_path,new_id)
#             if not os.path.exists(stock_path):
#                 os.mkdir(stock_path)
#             elif not os.path.exists(new_path):
#                 os.mkdir(new_path)
#             # if stock_id not in self.stock_set:
#             #     stock_file_path = os.path.join(self.db_dir,stock_id)
#             #     if not os.path.exists(stock_file_path):
#             #         os.makedirs(stock_file_path)
#             #     self.stock_set.add(stock_id)
#             with open(os.path.join(self.db_dir,stock_id,new_id,f'{page}.json'),'w',encoding='utf-8') as f:
#                 jsontext = json.dumps(dict(item),ensure_ascii=False,indent=4)
#                 f.write(jsontext)
#         return item

class MongoPipeline(object):
    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DB')
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def process_item(self, item, spider):
        if 'url' in item.keys():
            name = 'url'
            col = self.db[name]
            x = col.insert_one(dict(item))
        else:
            name = item.get
            name = item['stock_id']
            col = self.db[name]
            x = col.insert_one(dict(item))
            col_ = self.db['url']
            stock_id = item['news_id']
            x = col_.delete_one({"url":{"$regex":f'/news,{name},{stock_id}.html'}})

        return item

    def close_spider(self, spider):
        self.client.close()