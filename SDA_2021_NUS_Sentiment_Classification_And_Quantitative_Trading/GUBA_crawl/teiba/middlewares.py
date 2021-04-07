# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
import logging
logger = logging.getLogger(__name__)


import base64
# import logging
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware

# 隧道服务器
tunnel_host = "tps168.kdlapi.com"
tunnel_port = "15818"

# 隧道id和密码
tid = "t17332291843366"
password = "5gzmpenl"

logger = logging.getLogger(__name__)

# 代理中间件
class ProxyMiddleware(object):

    def process_request(self, request, spider):
        proxy_url = 'http://%s:%s@%s:%s' % (tid, password, tunnel_host, tunnel_port)
        request.meta['proxy'] = proxy_url  # 设置代理
        logger.debug("using proxy: {}".format(request.meta['proxy']))
        # 设置代理身份认证
        # Python3 写法
        auth = "Basic %s" % (base64.b64encode(('%s:%s' % (tid, password)).encode('utf-8'))).decode('utf-8')
        # Python2 写法
        # auth = "Basic " + base64.b64encode('%s:%s' % (tid, password))
        request.headers['Proxy-Authorization'] = auth


class AgentMiddleware(UserAgentMiddleware):
    """
        User-Agent中间件, 设置User-Agent
    """
    def __init__(self, user_agent=''):
        self.user_agent = user_agent

    def process_request(self, request, spider):
        ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:39.0) Gecko/20100101 Firefox/39.0'
        request.headers.setdefault('User-Agent', ua)