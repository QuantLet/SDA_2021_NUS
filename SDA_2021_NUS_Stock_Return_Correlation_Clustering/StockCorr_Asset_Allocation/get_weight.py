# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 14:19:42 2021

@author: 24978
"""

import pandas as pd
import process as p
import numpy as np
import scipy.cluster.hierarchy as sch
import os

def getIVP(cov):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    # set inf to 0
    ivp[np.isinf(ivp)] = 0
    ivp /= np.nansum(ivp)
    return ivp

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems]
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getHRP(data,sortIx):
    cov = data.cov()
    #cov.replace({np.nan:0},inplace=True)
    # Compute HRP alloc
    w = pd.Series([1]*len(sortIx), index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w.sort_index()

def cluster(data):
    corr = data.corr()
    dist = p.correlDist(corr)
    dist_n = dist.fillna(0)
    link = sch.linkage(dist_n, 'single')
    sortIx = p.getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    return sortIx

if __name__ == '__main__':
    symbols = p.get_symbols_from_file(os.path.join('sectors', 'sp500_symbol.csv')
    data = p.get_data(symbols)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    ret = data.pct_change()
    ret.drop(ret.head(1).index,inplace=True)
    info_date = ret.index.tolist()
    weight = pd.DataFrame(index=ret.columns.tolist())
    for i in range(252,len(info_date)):
        print(i)
        start = info_date[i-252]
        end = info_date[i]
        temp = ret.loc[start:end,:].dropna(how='all',axis=1)
        sortIx = cluster(temp)
        weight[info_date[i]] = getHRP(temp,sortIx)
    weight.to_pickle('weight1.pk')
