"""
Description:
 - Scraping multiple pages of news from nasdaq.com.
 - Saving result it as .csv file at local disk.
 - Saving web page source code temporarily after request.

Usage: Executing this whole file in your IDE or from Terminal.

Memo:
Adjusting range(start, end) to define how many pages you want to scrape
Toggling refresh to decide if you want to refresh the pages that have been stored locally.

Author: Junjie Hu, hujunjie@hu-berlin.de
Last modified date: 19-11-2017
"""

from bs4 import BeautifulSoup as soup
import requests
import datetime
import os
import pandas as pd
import pickle


def nasdaq_news_scraping(page=1, refresh=False):
    # Argument page equals 1 by default
    nasdaq_url = f'https://www.nasdaq.com/news-and-insights/topic/markets/stocks/page/{page}'

    if (not os.path.exists(direct + '/temp/' + str(page) + '.pkl')) or (refresh is True):
        # connect to the website if the webpage source code file is not exist of we need to refresh it
        url_request = requests.get(nasdaq_url)
        # save the request object after scraping
        with open(direct + '/temp/' + str(page) + '.pkl', 'wb') as url_file:
            pickle.dump(url_request, url_file)
    else:
        # else, we open the source code file from local disk to save time
        with open(direct + '/temp/' + str(page) + '.pkl', 'rb') as url_file:
            url_request = pickle.load(url_file)
    url_content = url_request.content
    parsed_content = soup(url_content)
    containers = parsed_content.find_all('div', {'data-type': 'article'})
    page_info = list()

    for container in containers:
        """
        Scrape 1 item each time, each item contains title, time, author_name, tag, author_url 
        """
        unit_info = dict()

        container_title = container.find_all('a', {"class": "content-feed__card-title-link"})
        if container_title == []:
            continue

        news_link = container_title[0].get('href').strip()
        news_link = f"https://www.nasdaq.com{news_link}"
        news_title = container_title[0].text.strip()
        unit_info['news_title'] = news_title
        unit_info['news_link'] = news_link

        try:
            news_date = datetime.datetime.strptime(news_link[-10:], '%Y-%m-%d')
            news_date = news_date.date()

        except:
            container_time = container.find_all('div', {'class': 'content-feed__card-timestamp'})
            try:
                news_date = datetime.datetime.strptime(container_time[0].text.strip(), '%b %d,%Y')
                news_date = news_date.date()

            except:
                t_day = datetime.date.today()
                date_series = container_time[0].text.strip().split(' ')
                if (date_series[1].startswith('hour')) or (date_series[1].startswith('minute')):
                    news_date = t_day

                elif date_series[1].startswith('day'):
                    news_date = t_day - datetime.timedelta(days=int(date_series[0]))
                else:
                    news_date = t_day

        news_date = datetime.datetime(news_date.year, news_date.month, news_date.day)
        unit_info['news_time'] = news_date
        page_info.append(unit_info)

    return page_info


direct = os.getcwd() + '/DEDA_WebScrapingIntro/'
# Or
# direct = os.getcwd()

head_lines_info = list()

for page_num in range(1, 30):
    print("This is Page: ", page_num)
    # Using the function defined previously with certain arguments as input
    page_info = nasdaq_news_scraping(page=page_num, refresh=True)
    head_lines_info.extend(page_info)

headlines_info_df = pd.DataFrame(head_lines_info, columns=['news_title', 'news_link', 'news_time'])
headlines_info_df.to_csv(direct + '/Nasdaq_News_MultiPages.csv')
