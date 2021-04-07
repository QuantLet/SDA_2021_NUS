# -*- coding: utf-8 -*-
import scrapy
import urllib.parse
from bs4 import BeautifulSoup
from teiba.items import ArticleItem,PageItem
import time
import random
import json
import pymongo

import logging
logger = logging.getLogger(__name__)

class SingleStockSpider(scrapy.Spider):
    name = 'single_stock'
    allowed_domains = ['guba.eastmoney.com']
    start_urls = ['http://guba.eastmoney.com/remenba.aspx?type=1&tab=1']

    def start_requests(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["guba"]
        col = db['url']
        item_list = list(col.find())
        random.shuffle(item_list)
        for item in item_list:
        # for item in col.find():
            url = urllib.parse.urljoin('http://guba.eastmoney.com/',item['url'])
            url = self.convert_first_page_url(url)
            # yield scrapy.Request(url=url,callback=self.parse_single_article_page,meta={'proxy': proxyMeta})
            time.sleep(0.2*random.random())
            yield scrapy.Request(url=url,callback=self.parse_single_article_page)

    @staticmethod
    def convert_first_page_url(url):
        parse_url_path = urllib.parse.urlparse(url).path
        path_ = '_1.'.join(parse_url_path.split('.'))
        return urllib.parse.urljoin(url,path_)

    @staticmethod
    def scroll_to_next_page(url):
        parse_url_path = urllib.parse.urlparse(url).path
        split_path = parse_url_path.split('.')
        split_page = split_path[0].split('_')
        next_page = str(int(split_page[-1])+1)
        split_page[-1] = next_page
        join_page = '_'.join(split_page)
        split_path[0] = join_page
        join_path = '.'.join(split_path)
        return urllib.parse.urljoin(url,join_path)

    @staticmethod
    def item_id_to_url(item_id):
        url = '/' + item_id + '.html'
        return url

    @staticmethod
    def url_to_item_id(url):
        item_id = urllib.parse.urlparse(url).path[1:-5]
        return item_id

    @staticmethod
    def parse_item_id(item_id):
        type_, stock_id, news_id = item_id.split(',')
        new_id, page = news_id.split('_')
        return type_, stock_id, new_id, page

    @staticmethod
    def root_page_href_to_stock_id(stock_id):
        return stock_id.split(',')[-1].split('.')[0]

    # @staticmethod
    # def bar_page_href_to_stock_and_pages(stock_id):
    #     path_ = stock_id.split(',')[-1].split('.')[0]
    #     stock, page = path_.split('_')
    #     return stock, page

    def parse(self, response):
        # 改为soup格式方便操作
        soup = BeautifulSoup(response.body,features="lxml")

        all_stock_tag = soup.find('div',class_='ngbggulbody list clearfix')
        # all_stock_tag_list = all_stock_tag.find_all(lambda tag:tag.name == 'a' and tag.has_attr('href'))
        all_stock_tag_list = all_stock_tag.find_all(lambda x:x.name == 'a' and x.has_key('href') and not x.has_key('target'))
        all_stock_id_list = [self.root_page_href_to_stock_id(a_tag['href']) for a_tag in all_stock_tag_list]
        self.all_stock_id_list = all_stock_id_list
        dict_ = {
        # '21' : {'stock_id':'600028', 'pages':'500'},
        '22' : {'stock_id':'600029', 'pages':'500'},
        '23' : {'stock_id':'600030', 'pages':'1000'},
        '27' : {'stock_id':'600036', 'pages':'300'},
        '31' : {'stock_id':'600048', 'pages':'500'},
        '389' : {'stock_id':'600485', 'pages':'1000'},
        '419' : {'stock_id':'600519', 'pages':'2500'},
        '442' : {'stock_id':'600547', 'pages':'750'},
        '522' : {'stock_id':'600637', 'pages':'750'},
        '704' : {'stock_id':'600837', 'pages':'500'},
        '749' : {'stock_id':'600887', 'pages':'750'},
        '755' : {'stock_id':'600893', 'pages':'300'},
        '774' : {'stock_id':'600958', 'pages':'200'},
        '810' : {'stock_id':'600999', 'pages':'200'},
        '816' : {'stock_id':'601006', 'pages':'300'},
        '837' : {'stock_id':'601088', 'pages':'300'},
        '860' : {'stock_id':'601166', 'pages':'500'},
        '862' : {'stock_id':'601169', 'pages':'350'},
        '865' : {'stock_id':'601186', 'pages':'500'},
        '867' : {'stock_id':'601198', 'pages':'200'},
        '871' : {'stock_id':'601211', 'pages':'250'},
        '885' : {'stock_id':'601288', 'pages':'500'},
        '888' : {'stock_id':'601318', 'pages':'1250'},
        '891' : {'stock_id':'601328', 'pages':'300'},
        '894' : {'stock_id':'601336', 'pages':'300'}
        }
        # list(list(stock['stock_id'] in tag['href'] for tag in all_title_tag).index(True) for stock in list_) # for index getting
        # for num, a_tag in enumerate(all_stock_tag_list): # crawl 1 stocks bar first !
        for num, key in enumerate(dict_.keys()):
            a_tag = all_stock_tag_list[int(key)]
            href = a_tag['href']
            logger.info(f'start parsing single stock page : {num+1}th stock, {self.root_page_href_to_stock_id(href)}')
            single_stock_bar_url = urllib.parse.urljoin(response.url,a_tag['href'])
            single_stock_bar_url = self.convert_first_page_url(single_stock_bar_url)
            # auto flip page
            # yield scrapy.Request(url=single_stock_bar_url,callback=self.parse_bar)
            # manually flip page
            yield scrapy.Request(url=single_stock_bar_url,callback=self.parse_bar_manually)
            # pages = 3000
            pages = int(dict_[key]['pages'])
            for page in range(1+1,pages+1):
                single_stock_bar_url = self.scroll_to_next_page(single_stock_bar_url)
                time.sleep(0.2*random.random())
                yield scrapy.Request(url=single_stock_bar_url,callback=self.parse_bar_manually)

    def parse_bar(self, response):
        # stock, page = self.bar_page_href_to_stock_and_pages(response.url)
        # logger.info(f'start parsing single stock bar page : stock {stock}, page {page}') 
        logger.info(f'start parsing single stock bar page : {urllib.parse.urlparse(response.url).path}') 
        # 改为soup格式方便操作        
        soup = BeautifulSoup(response.body,features="lxml")

        all_title_tag = soup.find_all('div',class_='articleh normal_post')
        for title_tag in all_title_tag:
            title_href = title_tag.find('a')['href']
            title_url = urllib.parse.urljoin(response.url,title_href)
            title_url = self.convert_first_page_url(title_url)
            yield scrapy.Request(url=title_url,callback=self.parse_single_article_page)
        if all_title_tag:
            # crawl next bar page
            next_bar_page_url = self.scroll_to_next_page(response.url)
            yield scrapy.Request(url=next_bar_page_url,callback=self.parse_bar)

    def parse_bar_manually(self, response):
        # stock, page = self.bar_page_href_to_stock_and_pages(response.url)
        # logger.info(f'start parsing single stock bar page : stock {stock}, page {page}') 
        logger.info(f'start parsing single stock bar page : {urllib.parse.urlparse(response.url).path}') 
        # 改为soup格式方便操作        
        soup = BeautifulSoup(response.body,features="lxml")

        all_title_tag = soup.find_all('div',class_='articleh normal_post')
        for title_tag in all_title_tag:
            title_href = title_tag.find('a')['href']
            item = PageItem()
            item['url'] = title_href
            yield item
            # title_url = urllib.parse.urljoin(response.url,title_href)
            # title_url = self.convert_first_page_url(title_url)
            # yield scrapy.Request(url=title_url,callback=self.parse_single_article_page)
        # if all_title_tag:
        #     # crawl next bar page
        #     next_bar_page_url = self.scroll_to_next_page(response.url)
        #     yield scrapy.Request(url=next_bar_page_url,callback=self.parse_bar)

    def parse_single_article_page(self, response):
        # 改为soup格式方便操作        
        soup = BeautifulSoup(response.body,features="lxml")

        item = ArticleItem()

        script_tag = soup.find(lambda x:x.name == 'script' and 'var post_article' in x.text)
        if script_tag:
            try:
                script_text = script_tag.text.strip()
                item_id = self.url_to_item_id(response.url)
                type_, stock_id, news_id, page = self.parse_item_id(item_id)

                # if stock_id in self.all_stock_id_list:
                item['item_id'] = item_id
                # item['type'] = type_
                item['stock_id'] = stock_id
                item['news_id'] = news_id
                item['page'] = page
                item['timestamp'] = time.strftime("%Y-%m-%d %X")
                for expression in script_text.split(';\r\n    '):
                    expression = expression.strip(';')
                    key,value = expression.split('=',1)
                    if not key.startswith('var '):
                        raise Exception(f'While analyzing Item, key of expreesion {key} occurs error')
                    key = key[4:].strip()
                    value = value.strip()
                    json_value = json.loads(value)
                    item[key] = json_value
                page_ = int(page)
                has_comments = item['reply_list']['re']
                if page_ == 1:
                    yield item
                # elif has_comments:
                #     item['post_article'] = dict()
                #     yield item
                # if has_comments:
                #     next_article_page_url = self.scroll_to_next_page(response.url)
                #     yield scrapy.Request(url=next_article_page_url,callback=self.parse_single_article_page)
            except json.decoder.JSONDecodeError:
                # item_ = ArticleItem()
                # item_['exception'] = 'JSONDecodeError'
                # yield item_
                raise Exception(f'Decoding JSON Error : {json_value}')
        else:
            # PageNotExists
            # item['exception'] = 'PageNotExists'
            raise Exception(f'Pages Already Not Exists : {response.url}')
