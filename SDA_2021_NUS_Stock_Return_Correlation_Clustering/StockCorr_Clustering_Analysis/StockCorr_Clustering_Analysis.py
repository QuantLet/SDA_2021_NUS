import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.cluster.hierarchy as sch
import numpy as np
import logging
import imageio
import moviepy.editor as mp
from datetime import datetime
from datetime import timedelta
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='cluster_analysis.log', filemode='w', level=logging.INFO, format=formatter)

sectors = ['basic_industries', 'capital_goods', 'consumer_durables', 'consumer_nondurables', 'consumer_services', \
            'energy', 'finance', 'health_care','miscellaneous','public_utilities','technology','transportation']

def symbol_to_path(symbol):
    # please download from below link and put under downloaded_data/data folder 
    # https://drive.google.com/file/d/1Uy0VmrkbKUAskGKAAQo45F8unrphAF14/view?usp=sharing
    if os.path.exists(os.path.join('downloaded_data', 'data')):
        return os.path.join('downloaded_data', 'data', '{}.csv'.format(str(symbol)))
    else:
        logging.error('please make sure path \'downloaded_data/data/\' is present under current working directory!')
        exit(1)

def get_symbols_from_file(file_path):
    symbols = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for s in lines:
            s = s.strip()
            if os.path.exists(symbol_to_path(s)):
                symbols.append(s)
            else:
                logging.info('data file for symbol {} does not exist!'.format(s))
    return symbols

def get_data(symbols):
    ''' 
    function to return dataframe of Adj Close price for symbols from downloaded_data set
    '''
    first = True
    for symbol in symbols:
        if first:
            # if first = True, construct df
            df = pd.read_csv(symbol_to_path(symbol), parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
            df.rename(columns={'Adj Close': symbol}, inplace=True)
            first = False

        else:
            try:
                df_temp = pd.read_csv(symbol_to_path(symbol),
                                    parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])

                df_temp = df_temp.rename(columns={'Adj Close': symbol})
                df = pd.merge(left=df, right=df_temp, how='outer', left_on='Date', right_on='Date')
            except Exception as e: 
                logging.error('error received when trying to append df for symbol {}: {}'.format(symbol, e))

    # df.set_index('Date')
    # df.sort_index(inplace=True)
    return df

def construct_index(sym_file, index_file_name):
    if not os.path.exists(sym_file):
        logging.error('Error when constructing index: not able to find file: {}'.format(sym_file))
        return 

    components = get_symbols_from_file(sym_file)
    data = get_data(components)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    df = pd.DataFrame(index=data.index)
    # use simple average of price here for index construction
    df['Adj Close'] = data.mean(axis=1)
    df.to_csv(index_file_name) 

def sector_index():
    f = open(os.path.join('sectors', 'index_names.csv'), 'w')
    for s in sectors:
        sym_file = os.path.join('sectors', '{}_sym.csv'.format(s))
        output = os.path.join('downloaded_data', 'data', '{}_index.csv'.format(s))
        try: 
            logging.info('Constructing index for sector: {}'.format(s))
            construct_index(sym_file, output)
            f.write('{}_index\n'.format(s))
        except Exception as e:
            logging.error('Error when constructing sector index for sector: {}, {}'.format(s,e))

    f.close()

def industry_index():
    industries = []
    with open(os.path.join('industries', 'industry_list.csv'), 'r') as f:
        lines = f.readlines()
        industries.extend([line.strip() for line in lines])

    f = open(os.path.join('industries', 'industry_index_names.csv'), 'w')
    for i in industries:
        sym_file = os.path.join('industries', '{}_sym.csv'.format(i))
        output = os.path.join('downloaded_data', 'data', '{}_ind_index.csv'.format(i))
        try: 
            logging.info('Constructing index for industry: {}'.format(i))
            construct_index(sym_file, output)
            logging.info('Constructed successfully, place file at {}'.format(output))
            f.write('{}_ind_index\n'.format(i))
        except Exception as e:
            logging.error('Error when constructing industry index for industry: {}, {}'.format(i,e))

    f.close()


def correlDist(corr):
    dist = ((1-corr)/2.)**.5
    return dist

def getQuasiDiag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1,0],link[-1,1]])
    numItems = link[-1,3]
    while sortIx.max()>=numItems:
        sortIx.index = range(0,sortIx.shape[0]*2,2)
        df0 = sortIx[sortIx>=numItems]
        i=df0.index
        j=df0.values-numItems
        sortIx[i] = link[j,0]
        df0 = pd.Series(link[j,1],index=i+1)
        sortIx = sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def cluster_plot(data, start, end, figname, cluster=True):
    data_tmp = data.loc[start:end, :]
    corr = data_tmp.pct_change().corr()
    plt.figure(figsize = (20,20))

    if cluster:
        dist = correlDist(corr)
        dist_n = dist.fillna(0)
        try:
            link = sch.linkage(dist_n, 'single')
            sortIx = getQuasiDiag(link)
        except Exception as e:
            logging.error('received error when trying to get sortIx: {}'.format(e))

        sortIx = corr.index[sortIx].tolist()
        df0 = corr.loc[sortIx, sortIx]

        sns_plot = sns.heatmap(df0, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
        fig = sns_plot.get_figure()
        fig.savefig(figname)
    else:
        sns_plot = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
        fig = sns_plot.get_figure()
        fig.savefig(figname)

    plt.close()

def run(sym_file, category):
    start_dates = ['2011-03-16', 
                   '2012-03-16', 
                   '2013-03-16', 
                   '2014-03-16', 
                   '2015-03-16', 
                   '2016-03-16',
                   '2017-03-16', 
                   '2018-03-16',
                   '2019-03-16',
                   '2020-03-16']

    end_dates = ['2012-03-16', 
                 '2013-03-16', 
                 '2014-03-16', 
                 '2015-03-16', 
                 '2016-03-16',
                 '2017-03-16', 
                 '2018-03-16',
                 '2019-03-16',
                 '2020-03-16', 
                 '2021-03-16']

    assert len(start_dates) == len(end_dates), 'length of start dates and end dates must be same'
    if not os.path.exists(sym_file):
        logging.error('{} does not exist, please make sure the file path is valid!'.format(sym_file))
        return 

    symbols = get_symbols_from_file(sym_file)
    data = get_data(symbols)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    # data.to_csv('data_{}.csv'.format(category))

    figure_path = os.path.join('figures', '{}'.format(category))

    if not (os.path.exists(figure_path) and os.path.isdir(figure_path)):
        os.makedirs(figure_path)

    logging.info('{}: plotting corr graph for 10 years interval'.format(category))
    cluster_plot(data, '2011-03-16', '2021-03-15', os.path.join('{}'.format(figure_path), '{}_corr_10y.png'.format(category)), False)
    logging.info('{}: plotting clustered corr graph for 10 years interval'.format(category))
    cluster_plot(data, '2011-03-16', '2021-03-15', os.path.join('{}'.format(figure_path), '{}_corr_cluster_10y.png'.format(category)), True)

    for (i,(s,e)) in enumerate(zip(start_dates, end_dates)):
        logging.info('clustering plotting for {} during time period: {} to {}'.format(category, s, e))
        cluster_plot(data, start_dates[i], end_dates[i], os.path.join('{}'.format(figure_path), '{}_corr_time_{}.png'.format(category, i)))
    logging.info('{}: plotted corr clustering for the time intervals'.format(category))

def animate(sym_file, category, start, end, interval=90, window=360, cluster=True):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    if not os.path.exists(sym_file):
        logging.error('{} does not exist, please make sure the file path is valid!'.format(sym_file))
        return 

    symbols = get_symbols_from_file(sym_file)
    data = get_data(symbols)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    animation_path = os.path.join('animations', '{}'.format(category))

    if not (os.path.exists(animation_path) and os.path.isdir(animation_path)):
        os.makedirs(animation_path)

    logging.info('{}: plotting corr animation for window {} and interval {}, start: {}, end: {}'.format(category, window, interval, start, end))
    images = []

    i = 0
    while start_date + timedelta(days=window) < end_date: 
        s = start_date.strftime('%Y-%m-%d')
        e = (start_date + timedelta(days=window)).strftime('%Y-%m-%d')
        logging.info('Animation: plotting for period: {} to {}'.format(s, e))

        data_tmp = data.loc[s:e, :]
        corr = data_tmp.pct_change().corr()

        plt.figure(figsize=(20,20))
        if cluster:
            dist = correlDist(corr)
            dist_n = dist.fillna(0)
            try:
                link = sch.linkage(dist_n, 'single')
                sortIx = getQuasiDiag(link)
            except Exception as e:
                logging.error('received error when trying to get sortIx: {}'.format(e))

            sortIx = corr.index[sortIx].tolist()
            df0 = corr.loc[sortIx, sortIx]

            sns_plot=sns.heatmap(df0, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
            fig = sns_plot.get_figure()
            ax = fig.add_subplot(111)
            ax.set_title('Corr {} - {}'.format(s, e))
            figname = os.path.join(animation_path, 'animate_{}.png'.format(i))
            fig.savefig(figname)
            images.append(imageio.imread(figname))
            plt.close()

        else:
            sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
            fig = sns_plot.get_figure()
            ax = fig.add_subplot(111)
            ax.set_title('Corr {} - {}'.format(s, e))
            figname = os.path.join(animation_path, 'animate_{}.png'.format(i))
            fig.savefig(figname)
            images.append(imageio.imread(figname))
            plt.close()

        start_date += timedelta(days=interval)
        i += 1

    imageio.mimsave(os.path.join(animation_path,'{}.gif'.format(category)), images, fps=2)
    clip = mp.VideoFileClip(os.path.join(animation_path,'{}.gif'.format(category)))
    clip.write_videofile(os.path.join(animation_path,'{}.mp4'.format(category)))
    logging.info('{}: corr animation saved at {}'.format(category, animation_path))


def main():
    # run('sp500_symbol.csv', 'SP500')
    logging.info('Starting the process...')

    logging.info('{}: process for {} symbols'.format('SP500', 'SP500'))
    run(os.path.join('sectors', 'sp500_symbol.csv'), 'SP500')

    for s in sectors:
        logging.info('{}: process for {} symbols'.format(s, s))
        try:
            run(os.path.join('sectors', '{}_sym.csv'.format(s)), s)
        except Exception as e:
            logging.error('Error when trying to process for sector: {}, {}'.format(s, e))

    if not os.path.exists(os.path.join('sectors', 'index_names.csv')):
        logging.info('Constructing sector index...')
        sector_index()
        logging.info('Sector index constructed!')
    
    logging.info('Plotting for sector index corr...')
    try:
        run(os.path.join('sectors', 'index_names.csv'), 'sector')
    except Exception as e:
        logging.error('Error when trying to process sector index corr: {}'.format(e))

    if not os.path.exists(os.path.join('industries', 'industry_index_names.csv')):
        logging.info('Constructing industry index...')
        industry_index()
        logging.info('Industry index constructed!')
    
    logging.info('Plotting for industry index corr...')
    try:
        run(os.path.join('industries', 'industry_index_names.csv'), 'industry')
    except Exception as e:
        logging.error('Error when trying to process industry index corr: {}'.format(e))

    logging.info('Start creating animation for S&P500...')
    animate(os.path.join('sectors', 'sp500_symbol.csv'), 'SP500', '2011-03-15', '2021-03-15', 90)
    logging.info('Created  animation for S&P500...')

    logging.info('Start creating animation for industries...')
    animate(os.path.join('industries', 'industry_index_names.csv'), 'industry', '2011-03-15', '2021-03-15', 90)
    logging.info('Created  animation for industries...')

    logging.info('Start creating animation for sectors...')
    animate(os.path.join('sectors', 'index_names.csv'), 'sector', '2011-03-15', '2021-03-15', 90)
    logging.info('Created  animation for sectors...')

    logging.info('End of the process. BYE!')


if __name__ == '__main__':
    main()
