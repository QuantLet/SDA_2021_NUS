[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **StockCorr_Data_Downloader** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'StockCorr_Data_Downloader'

Published in: 'SDA_2021_NUS/SDA_2021_NUS_Stock_Return_Correlation_Clustering'

Description: 'Uses python script to download NASDAQ and NYSE mega/large/mid/small cap stock daily price data from Yahoo!Finance'

Keywords: 'stock, data, downloader, correlation, clustering, concurrency'

Author: 'Li Yilin, Mei Yuxin, Sun Qingwei, Xie Chuda, Zhang Yingxin'

See also: 'StockCorr_Clustering_Analysis, StockCorr_Asset_Allocation'

Submitted:  '01. Apr 2021'

Datafile: 'NASDAQ and NYSE stock mega/large/mid/small cap stock daily prices'

```

### PYTHON Code
```python

import time 
import logging
import downloader
import os
import argparse
from concurrent import futures

FILEDIR = os.path.join('downloaded_data', 'data')
initial = '2011-3-16'
final = '2021-3-16'

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='auto_download.log', filemode='w', level=logging.INFO, format=formatter)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--initial", help="initial date in %Y-%m-%d format")
parser.add_argument("-f", "--final", help="final date in %Y-%m-%d format")
args = parser.parse_args()
if args.initial: 
    initial = args.initial

if args.final:
    final = args.final

logging.info('date range: from {} to {}'.format(initial, final))

start = int(time.mktime(time.strptime(str(initial), '%Y-%m-%d'))) 
end = int(time.mktime(time.strptime(str(final), '%Y-%m-%d')))
    
if not os.path.exists(FILEDIR):
    os.makedirs(FILEDIR)

def read_symbols(symfile):
    syms = []
    with open(symfile, 'r') as f:
        content = f.readlines()
        for s in content:
            syms.append(s.strip())
       
    return syms

def download_one(symbol):
    if os.path.exists(FILEDIR + '{}.csv'.format(symbol)):
        # logging.info('already downloaded, skip')
        return

    logging.info('downloading for symbol: {}'.format(symbol))
    try: 
        abc = downloader.downloader(symbol,start,end)
        abc = abc.decode('utf-8')

        with open(FILEDIR + '{}.csv'.format(symbol), "w") as f:
            f.writelines(str(abc))
        logging.info('downloaded successfully')
    except:
        logging.error('got error when trying to download: {}'.format(symbol))


def download_many(symbols: list[str]) -> None: 
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        res = executor.map(download_one, symbols)

def main():
    logging.info('start downloading process...') 

    nasdaq_symbols = read_symbols('nasdaq_symbols.csv')
    nyse_symbols = read_symbols('nyse_symbols.csv')
    symbols = nasdaq_symbols + nyse_symbols 

    download_many(symbols)

    logging.info('finished downloading process!')


if __name__ == '__main__':
    main()

```

automatically created on 2021-04-01