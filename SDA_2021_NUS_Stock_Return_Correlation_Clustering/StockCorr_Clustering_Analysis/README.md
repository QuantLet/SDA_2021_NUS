[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **StockCorr_Clustering_Analysis** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'StockCorr_Clustering_Analysis'

Published in: 'SDA_2021_NUS/SDA_2021_NUS_Stock_Return_Correlation_Clustering'

Description: 'Uses hierarchical clustering method to analyse the correlation of stock returns along time'

Keywords: 'stock, correlation, hierarchical clustering, temporal, sectors, industries'

Author: 'Li Yilin, Mei Yuxin, Sun Qingwei, Xie Chuda, Zhang Yingxin'

See also: 'StockCorr_Data_Downloader, StockCorr_Asset_Allocation'

Submitted:  '01. Apr 2021'

Datafile: 'NASDAQ and NYSE stock mega/large/mid/small cap stock daily prices'

```

![Picture1](SP500_corr_10y.png)

![Picture2](SP500_corr_cluster_10y.png)

![Picture3](industry_corr_10y.png)

![Picture4](industry_corr_cluster_10y.png)

![Picture5](sector_corr_10y.png)

![Picture6](sector_corr_cluster_10y.png)

### IPYNB Code
```ipynb

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "driven-prague",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: imageio==2.9.0 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 1)) (2.9.0)\n",
      "Requirement already satisfied: matplotlib==3.3.4 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 2)) (3.3.4)\n",
      "Requirement already satisfied: moviepy==1.0.3 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.0.3)\n",
      "Requirement already satisfied: numpy==1.20.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 4)) (1.20.1)\n",
      "Requirement already satisfied: pandas==1.2.3 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 5)) (1.2.3)\n",
      "Requirement already satisfied: scikit-learn==0.24.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 6)) (0.24.1)\n",
      "Requirement already satisfied: scipy==1.6.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 7)) (1.6.1)\n",
      "Requirement already satisfied: seaborn==0.11.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from -r requirements.txt (line 8)) (0.11.1)\n",
      "Requirement already satisfied: pillow in /Users/sunqingwei/venv/lib/python3.9/site-packages (from imageio==2.9.0->-r requirements.txt (line 1)) (8.1.2)\n",
      "Requirement already satisfied: cycler>=0.10 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from matplotlib==3.3.4->-r requirements.txt (line 2)) (0.10.0)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from matplotlib==3.3.4->-r requirements.txt (line 2)) (2.8.1)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from matplotlib==3.3.4->-r requirements.txt (line 2)) (1.3.1)\n",
      "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from matplotlib==3.3.4->-r requirements.txt (line 2)) (2.4.7)\n",
      "Requirement already satisfied: imageio-ffmpeg>=0.2.0 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from moviepy==1.0.3->-r requirements.txt (line 3)) (0.4.3)\n",
      "Requirement already satisfied: proglog<=1.0.0 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from moviepy==1.0.3->-r requirements.txt (line 3)) (0.1.9)\n",
      "Requirement already satisfied: decorator<5.0,>=4.0.2 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from moviepy==1.0.3->-r requirements.txt (line 3)) (4.4.2)\n",
      "Requirement already satisfied: requests<3.0,>=2.8.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from moviepy==1.0.3->-r requirements.txt (line 3)) (2.25.1)\n",
      "Requirement already satisfied: tqdm<5.0,>=4.11.2 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from moviepy==1.0.3->-r requirements.txt (line 3)) (4.56.0)\n",
      "Requirement already satisfied: pytz>=2017.3 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from pandas==1.2.3->-r requirements.txt (line 5)) (2021.1)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from scikit-learn==0.24.1->-r requirements.txt (line 6)) (2.1.0)\n",
      "Requirement already satisfied: joblib>=0.11 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from scikit-learn==0.24.1->-r requirements.txt (line 6)) (1.0.1)\n",
      "Requirement already satisfied: six in /Users/sunqingwei/venv/lib/python3.9/site-packages (from cycler>=0.10->matplotlib==3.3.4->-r requirements.txt (line 2)) (1.15.0)\n",
      "Requirement already satisfied: chardet<5,>=3.0.2 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from requests<3.0,>=2.8.1->moviepy==1.0.3->-r requirements.txt (line 3)) (4.0.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from requests<3.0,>=2.8.1->moviepy==1.0.3->-r requirements.txt (line 3)) (2020.12.5)\n",
      "Requirement already satisfied: idna<3,>=2.5 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from requests<3.0,>=2.8.1->moviepy==1.0.3->-r requirements.txt (line 3)) (2.10)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/sunqingwei/venv/lib/python3.9/site-packages (from requests<3.0,>=2.8.1->moviepy==1.0.3->-r requirements.txt (line 3)) (1.26.3)\n"
     ]
    }
   ],
   "source": [
    "!pip install -r requirements.txt "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "peripheral-messenger",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "import seaborn as sns\n",
    "import scipy.cluster.hierarchy as sch\n",
    "import numpy as np\n",
    "import logging\n",
    "import imageio\n",
    "import moviepy.editor as mp\n",
    "from datetime import datetime\n",
    "from datetime import timedelta\n",
    "from scipy.cluster.hierarchy import ClusterWarning\n",
    "from warnings import simplefilter\n",
    "simplefilter(\"ignore\", ClusterWarning)\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\",category=matplotlib.cbook.mplDeprecation)\n",
    "\n",
    "formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n",
    "logging.basicConfig(filename='cluster_analysis.log', filemode='w', level=logging.INFO, format=formatter)\n",
    "\n",
    "sectors = ['basic_industries', 'capital_goods', 'consumer_durables', 'consumer_nondurables', 'consumer_services', \\\n",
    "            'energy', 'finance', 'health_care','miscellaneous','public_utilities','technology','transportation']\n",
    "\n",
    "def symbol_to_path(symbol):\n",
    "    # download data from https://drive.google.com/file/d/1Uy0VmrkbKUAskGKAAQo45F8unrphAF14/view?usp=sharing\n",
    "    # and save to downloaded_data/data/ folder\n",
    "    if os.path.exists(os.path.join('downloaded_data','data')):\n",
    "        return os.path.join('downloaded_data','data', '{}.csv'.format(str(symbol)))\n",
    "    else:\n",
    "        logging.error('please make sure path \\'downloaded_data/data/\\' is present under current working directory!')\n",
    "        exit(1)\n",
    "\n",
    "def get_symbols_from_file(file_path):\n",
    "    symbols = []\n",
    "\n",
    "    with open(file_path, 'r') as f:\n",
    "        lines = f.readlines()\n",
    "        for s in lines:\n",
    "            s = s.strip()\n",
    "            if os.path.exists(symbol_to_path(s)):\n",
    "                symbols.append(s)\n",
    "            else:\n",
    "                logging.info('data file for symbol {} does not exist!'.format(s))\n",
    "    return symbols\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "decent-porcelain",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_data(symbols):\n",
    "    ''' \n",
    "    function to return dataframe of Adj Close price for symbols from downloaded_data set\n",
    "    '''\n",
    "    first = True\n",
    "    for symbol in symbols:\n",
    "        if first:\n",
    "            # if first = True, construct df\n",
    "            df = pd.read_csv(symbol_to_path(symbol), parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])\n",
    "            df.rename(columns={'Adj Close': symbol}, inplace=True)\n",
    "            first = False\n",
    "\n",
    "        else:\n",
    "            try:\n",
    "                df_temp = pd.read_csv(symbol_to_path(symbol),\n",
    "                                    parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])\n",
    "\n",
    "                df_temp = df_temp.rename(columns={'Adj Close': symbol})\n",
    "                df = pd.merge(left=df, right=df_temp, how='outer', left_on='Date', right_on='Date')\n",
    "            except Exception as e: \n",
    "                logging.error('error received when trying to append df for symbol {}: {}'.format(symbol, e))\n",
    "\n",
    "    # df.set_index('Date')\n",
    "    # df.sort_index(inplace=True)\n",
    "    return df\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "geographic-indiana",
   "metadata": {},
   "outputs": [],
   "source": [
    "def construct_index(sym_file, index_file_name):\n",
    "    if not os.path.exists(sym_file):\n",
    "        logging.error('Error when constructing index: not able to find file: {}'.format(sym_file))\n",
    "        return \n",
    "\n",
    "    components = get_symbols_from_file(sym_file)\n",
    "    data = get_data(components)\n",
    "    data.set_index('Date', inplace=True)\n",
    "    data.sort_index(inplace=True)\n",
    "    df = pd.DataFrame(index=data.index)\n",
    "    # use simple average of price here for index construction\n",
    "    df['Adj Close'] = data.mean(axis=1)\n",
    "    df.to_csv(index_file_name) \n",
    "\n",
    "def sector_index():\n",
    "    f = open(os.path.join('sectors','index_names.csv'), 'w')\n",
    "    for s in sectors:\n",
    "        sym_file = os.path.join('sectors', '{}_sym.csv'.format(s))\n",
    "        output = os.path.join('downloaded_data','data', '{}_index.csv'.format(s))\n",
    "        try: \n",
    "            logging.info('Constructing index for sector: {}'.format(s))\n",
    "            construct_index(sym_file, output)\n",
    "            f.write('{}_index\\n'.format(s))\n",
    "        except Exception as e:\n",
    "            logging.error('Error when constructing sector index for sector: {}, {}'.format(s,e))\n",
    "\n",
    "    f.close()\n",
    "\n",
    "def industry_index():\n",
    "    industries = []\n",
    "    with open(os.path.join('industries', 'industry_list.csv'), 'r') as f:\n",
    "        lines = f.readlines()\n",
    "        industries.extend([line.strip() for line in lines])\n",
    "\n",
    "    f = open(os.path.join('industries','industry_index_names.csv'), 'w')\n",
    "    for i in industries:\n",
    "        sym_file = os.path.join('industries','{}_sym.csv'.format(i))\n",
    "        output = os.path.join('downloaded_data', 'data', '{}_ind_index.csv'.format(i))\n",
    "        try: \n",
    "            logging.info('Constructing index for industry: {}'.format(i))\n",
    "            construct_index(sym_file, output)\n",
    "            logging.info('Constructed successfully, place file at {}'.format(output))\n",
    "            f.write('{}_ind_index\\n'.format(i))\n",
    "        except Exception as e:\n",
    "            logging.error('Error when constructing industry index for industry: {}, {}'.format(i,e))\n",
    "\n",
    "    f.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "floral-installation",
   "metadata": {},
   "outputs": [],
   "source": [
    "def correlDist(corr):\n",
    "    dist = ((1-corr)/2.)**.5\n",
    "    return dist\n",
    "\n",
    "def getQuasiDiag(link):\n",
    "    link = link.astype(int)\n",
    "    sortIx = pd.Series([link[-1,0],link[-1,1]])\n",
    "    numItems = link[-1,3]\n",
    "    while sortIx.max()>=numItems:\n",
    "        sortIx.index = range(0,sortIx.shape[0]*2,2)\n",
    "        df0 = sortIx[sortIx>=numItems]\n",
    "        i=df0.index\n",
    "        j=df0.values-numItems\n",
    "        sortIx[i] = link[j,0]\n",
    "        df0 = pd.Series(link[j,1],index=i+1)\n",
    "        sortIx = sortIx.append(df0)\n",
    "        sortIx = sortIx.sort_index()\n",
    "        sortIx.index = range(sortIx.shape[0])\n",
    "    return sortIx.tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "conscious-arthur",
   "metadata": {},
   "outputs": [],
   "source": [
    "def cluster_plot(data, start, end, figname, cluster=True):\n",
    "    data_tmp = data.loc[start:end, :]\n",
    "    corr = data_tmp.pct_change().corr()\n",
    "    plt.figure(figsize = (20,20))\n",
    "\n",
    "    if cluster:\n",
    "        dist = correlDist(corr)\n",
    "        dist_n = dist.fillna(0)\n",
    "        try:\n",
    "            link = sch.linkage(dist_n, 'single')\n",
    "            sortIx = getQuasiDiag(link)\n",
    "        except Exception as e:\n",
    "            logging.error('received error when trying to get sortIx: {}'.format(e))\n",
    "\n",
    "        sortIx = corr.index[sortIx].tolist()\n",
    "        df0 = corr.loc[sortIx, sortIx]\n",
    "\n",
    "        sns_plot = sns.heatmap(df0, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)\n",
    "        fig = sns_plot.get_figure()\n",
    "        fig.savefig(figname)\n",
    "    else:\n",
    "        sns_plot = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)\n",
    "        fig = sns_plot.get_figure()\n",
    "        fig.savefig(figname)\n",
    "\n",
    "    plt.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "modern-kernel",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run(sym_file, category):\n",
    "    start_dates = ['2011-03-16', \n",
    "                   '2012-03-16', \n",
    "                   '2013-03-16', \n",
    "                   '2014-03-16', \n",
    "                   '2015-03-16', \n",
    "                   '2016-03-16',\n",
    "                   '2017-03-16', \n",
    "                   '2018-03-16',\n",
    "                   '2019-03-16',\n",
    "                   '2020-03-16']\n",
    "\n",
    "    end_dates = ['2012-03-16', \n",
    "                 '2013-03-16', \n",
    "                 '2014-03-16', \n",
    "                 '2015-03-16', \n",
    "                 '2016-03-16',\n",
    "                 '2017-03-16', \n",
    "                 '2018-03-16',\n",
    "                 '2019-03-16',\n",
    "                 '2020-03-16', \n",
    "                 '2021-03-16']\n",
    "\n",
    "    assert len(start_dates) == len(end_dates), 'length of start dates and end dates must be same'\n",
    "    if not os.path.exists(sym_file):\n",
    "        logging.error('{} does not exist, please make sure the file path is valid!'.format(sym_file))\n",
    "        return \n",
    "\n",
    "    symbols = get_symbols_from_file(sym_file)\n",
    "    data = get_data(symbols)\n",
    "    data.set_index('Date', inplace=True)\n",
    "    data.sort_index(inplace=True)\n",
    "    # data.to_csv('data_{}.csv'.format(category))\n",
    "\n",
    "    figure_path = os.path.join('figures','{}'.format(category))\n",
    "\n",
    "    if not (os.path.exists(figure_path) and os.path.isdir(figure_path)):\n",
    "        os.makedirs(figure_path)\n",
    "\n",
    "    logging.info('{}: plotting corr graph for 10 years interval'.format(category))\n",
    "    cluster_plot(data, '2011-03-16', '2021-03-15', os.path.join('{}'.format(figure_path), '{}_corr_10y.png'.format(category)), False)\n",
    "    logging.info('{}: plotting clustered corr graph for 10 years interval'.format(category))\n",
    "    cluster_plot(data, '2011-03-16', '2021-03-15', os.path.join('{}'.format(figure_path), '{}_corr_cluster_10y.png'.format(category)), True)\n",
    "\n",
    "    for (i,(s,e)) in enumerate(zip(start_dates, end_dates)):\n",
    "        logging.info('clustering plotting for {} during time period: {} to {}'.format(category, s, e))\n",
    "        cluster_plot(data, start_dates[i], end_dates[i], os.path.join('{}'.format(figure_path),'{}_corr_time_{}.png'.format(category, i)))\n",
    "    logging.info('{}: plotted corr clustering for the time intervals'.format(category))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "tamil-mortgage",
   "metadata": {},
   "outputs": [],
   "source": [
    "def animate(sym_file, category, start, end, interval=30, window=360, cluster=True):\n",
    "    start_date = datetime.strptime(start, '%Y-%m-%d')\n",
    "    end_date = datetime.strptime(end, '%Y-%m-%d')\n",
    "    if not os.path.exists(sym_file):\n",
    "        logging.error('{} does not exist, please make sure the file path is valid!'.format(sym_file))\n",
    "        return \n",
    "\n",
    "    symbols = get_symbols_from_file(sym_file)\n",
    "    data = get_data(symbols)\n",
    "    data.set_index('Date', inplace=True)\n",
    "    data.sort_index(inplace=True)\n",
    "    animation_path = os.path.join('animations', '{}'.format(category))\n",
    "\n",
    "    if not (os.path.exists(animation_path) and os.path.isdir(animation_path)):\n",
    "        os.makedirs(animation_path)\n",
    "\n",
    "    logging.info('{}: plotting corr animation for window {} and interval {}, start: {}, end: {}'.format(category, window, interval, start, end))\n",
    "    images = []\n",
    "\n",
    "    i = 0\n",
    "    while start_date + timedelta(days=window) < end_date: \n",
    "        s = start_date.strftime('%Y-%m-%d')\n",
    "        e = (start_date + timedelta(days=window)).strftime('%Y-%m-%d')\n",
    "        data_tmp = data.loc[s:e, :]\n",
    "        corr = data_tmp.pct_change().corr()\n",
    "\n",
    "        plt.figure(figsize=(20,20))\n",
    "        if cluster:\n",
    "            dist = correlDist(corr)\n",
    "            dist_n = dist.fillna(0)\n",
    "            try:\n",
    "                link = sch.linkage(dist_n, 'single')\n",
    "                sortIx = getQuasiDiag(link)\n",
    "            except Exception as e:\n",
    "                logging.error('received error when trying to get sortIx: {}'.format(e))\n",
    "\n",
    "            sortIx = corr.index[sortIx].tolist()\n",
    "            df0 = corr.loc[sortIx, sortIx]\n",
    "\n",
    "            sns_plot=sns.heatmap(df0, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)\n",
    "            fig = sns_plot.get_figure()\n",
    "            ax = fig.add_subplot(111)\n",
    "            ax.set_title('Corr {} - {}'.format(s, e))\n",
    "            figname = os.path.join(animation_path, 'animate_{}.png'.format(i))\n",
    "            fig.savefig(figname)\n",
    "            images.append(imageio.imread(figname))\n",
    "            plt.close()\n",
    "\n",
    "        else:\n",
    "            sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)\n",
    "            fig = sns_plot.get_figure()\n",
    "            figname = os.path.join(animation_path, 'animate_{}.png'.format(i))\n",
    "            fig.savefig(figname)\n",
    "            images.append(imageio.imread(figname))\n",
    "            plt.close()\n",
    "\n",
    "        start_date += timedelta(days=interval)\n",
    "        i += 1\n",
    "\n",
    "    imageio.mimsave(os.path.join(animation_path,'{}.gif'.format(category)), images, fps=1)\n",
    "    clip = mp.VideoFileClip(os.path.join(animation_path,'{}.gif'.format(category)))\n",
    "    clip.write_videofile(os.path.join(animation_path,'{}.mp4'.format(category)))\n",
    "    logging.info('{}: corr animation saved at {}'.format(category, animation_path))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "proved-helping",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    # run('sp500_symbol.csv', 'SP500')\n",
    "    logging.info('Starting the process...')\n",
    "\n",
    "    logging.info('{}: process for {} symbols'.format('SP500', 'SP500'))\n",
    "    run(os.path.join('sectors','sp500_symbol.csv'), 'SP500')\n",
    "\n",
    "    for s in sectors:\n",
    "        logging.info('{}: process for {} symbols'.format(s, s))\n",
    "        try:\n",
    "            run(os.path.join('sectors','{}_sym.csv'.format(s)), s)\n",
    "        except Exception as e:\n",
    "            logging.error('Error when trying to process for sector: {}, {}'.format(s, e))\n",
    "\n",
    "    if not os.path.exists(os.path.join('sectors','index_names.csv')):\n",
    "        logging.info('Constructing sector index...')\n",
    "        sector_index()\n",
    "        logging.info('Sector index constructed!')\n",
    "    \n",
    "    logging.info('Plotting for sector index corr...')\n",
    "    try:\n",
    "        run(os.path.join('sectors','index_names.csv'), 'sector')\n",
    "    except Exception as e:\n",
    "        logging.error('Error when trying to process sector index corr: {}'.format(e))\n",
    "\n",
    "    if not os.path.exists(os.path.join('industries', 'industry_index_names.csv')):\n",
    "        logging.info('Constructing industry index...')\n",
    "        industry_index()\n",
    "        logging.info('Industry index constructed!')\n",
    "    \n",
    "    logging.info('Plotting for industry index corr...')\n",
    "    try:\n",
    "        run(os.path.join('industries','industry_index_names.csv'), 'industry')\n",
    "    except Exception as e:\n",
    "        logging.error('Error when trying to process industry index corr: {}'.format(e))\n",
    "        \n",
    "    # produce animations\n",
    "    animate(os.path.join('sectors','sp500_symbol.csv'), 'SP500', '2017-03-15', '2021-03-15', 90)\n",
    "    animate(os.path.join('sectors','index_names.csv'), 'sector', '2017-03-15', '2021-03-15', 90)\n",
    "    animate(os.path.join('industries','industry_index_names.csv'), 'industry', '2017-03-15', '2021-03-15', 90)\n",
    "\n",
    "    logging.info('End of the process. BYE!')\n",
    "    print('Process Done!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "special-fleece",
   "metadata": {},
   "outputs": [],
   "source": [
    "main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

```

automatically created on 2021-04-01

### PYTHON Code
```python

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

```

automatically created on 2021-04-01