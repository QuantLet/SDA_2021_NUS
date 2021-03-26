
d_sector = {}
d_industry = {}
s_industry = set()

with open('industries_ticker.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        sector, industry, ticker = line.strip().split(',')
        sector = sector.strip().replace(' ', '')
        industry = industry.strip().replace(' ', '')
        ticker = ticker.strip().replace(' ', '')
        if sector not in d_sector:
            d_sector[sector] = set([industry])
        else:
            d_sector[sector].add(industry)

        if industry not in d_industry:
            d_industry[industry] = set([ticker])
        else:
            d_industry[industry].add(ticker)
        s_industry.add(industry)

for (k,v) in d_sector.items():
    industries = list(v)
    with open('../industries/{}_industry.csv'.format(k), 'w') as f:
        for elem in industries:
            f.write('{}_ind_index\n'.format(elem))

for (k,v) in d_industry.items():
    tickers = list(v)
    with open('../industries/{}_sym.csv'.format(k), 'w') as f:
        for elem in tickers:
            f.write('{}\n'.format(elem))

with open('../industries/industry_list.csv', 'w') as f:
    for i in list(s_industry):
        f.write('{}\n'.format(i))
    
