import requests
import re
import json
import time as time

def run(symbol, initial, final):
    start = int(time.mktime(time.strptime(str(initial), '%Y-%m-%d'))) # convertir fechas
    end = int(time.mktime(time.strptime(str(final), '%Y-%m-%d')))

    abc = downloader(symbol,start,end)
    abc = abc.decode('utf-8')

    with open('{}.csv'.format(symbol), "w") as f:
        f.writelines(str(abc))


def downloader(symbol, start, end):
    ses = requests.session()
    resp = ses.get(
        f'https://finance.yahoo.com/quote/{symbol}/history?period1={start}&period2={end}&interval=1d&filter=history&frequency=1d'
    )
    resp.raise_for_status()
    data = resp.content.decode('utf-8')
    crumb_m = re.search(r'"CrumbStore":\{"crumb":("[^"]+")\}', data)
    assert crumb_m
    crumb = json.loads(crumb_m.group(1))
    csvresp = ses.get(
        f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}'
    )
    csvresp.raise_for_status()
    return csvresp.content


if __name__=="__main__":
    # s = ['^VIX', '^GSPC', '^HSI', '^IXIC', '^VIX3M', '^VXN', '^VXO']
    s = ['^IXR']
    for symbol in s:
        run(symbol, '2010-3-18', '2021-3-17')
