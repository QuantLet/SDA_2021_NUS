import pandas as pd
import numpy as np
import copy
import os

# data: rowname: date colname : stocks
class Backtestlite(object):
    def __init__(self, factor_path, direction, fwdrtn, mkt_index, save_path_factor, cycle, start, end, quantile, statuslimit):
        self.data = pd.read_csv(factor_path, index_col = 0, parse_dates = True).loc[start:end][::cycle]
        self.mkt_fwrt = mkt_index
        self.fwdrtn = pd.concat([fwdrtn, self.mkt_fwrt], axis = 1)
        self.statuslimit = statuslimit
        self.fwdata = self.data.mul(self.statuslimit,axis = 0)
        self.rank = self.fwdata.rank(ascending = (direction == 'Ascending'), axis = 1, pct = True)
        self.quantile = quantile
        self.save_path = save_path_factor
        
    # For a certain quantile (choose_quantile), return weight (dataframe)
    def getQuantileWeight(self, choose_quantile):
        weight = pd.DataFrame(0., index = self.rank.index, columns = self.rank.columns)
        weight[(self.rank <= ((choose_quantile * 1.0) / self.quantile)) & (self.rank > ((choose_quantile - 1) / self.quantile))] = 1
        weight = weight.div(weight.sum(axis = 1), axis = 0)
        return weight

    # Return weight (dataframe)
    def getWeight(self, Type):
        #Long Short stocks
        if Type == "LS":
            weight_stocks = self.getQuantileWeight(self.quantile) - self.getQuantileWeight(1.)
            weight_future = pd.Series(0., index = self.data.index, name = self.mkt_fwrt.name)
        # Hedge the future.
        else:
            weight_stocks = self.getQuantileWeight(self.quantile)        
            weight_future = pd.Series(-1.0 * weight_stocks.sum(axis = 1), index = self.data.index, name = self.mkt_fwrt.name)
        weight = pd.concat([weight_stocks, weight_future], axis = 1)
        self.weight = weight
        return weight
    
    #  quantile return
    def cal_net_value(self, Type):
        self.rtn_data = pd.DataFrame(index = self.rank.index)
        for choose_quantile in range(1, self.quantile + 1):
            weight = self.getQuantileWeight(choose_quantile)
            rtn = weight * self.fwdrtn.iloc[:,:-1]
            rtn = rtn.sum(axis = 1)
            rtn.name = "Q_" + str(choose_quantile)
            self.rtn_data = pd.concat([self.rtn_data, rtn], axis = 1)
        
        finalweight = self.getWeight(Type)
        self.finalrtn = finalweight * self.fwdrtn
        self.finalrtn = self.finalrtn.sum(axis = 1)
        self.finalrtn.name = "Q_"+Type
        self.rtn_data = pd.concat([self.rtn_data, self.finalrtn], axis = 1)
        self.rtn_data.to_csv(os.path.join(self.save_path, 'Quantile Return.csv'))
        self.net_value = (1. + self.finalrtn).cumprod().shift(1).fillna(1.)
        return
    
    def cal_coverage(self):
        self.coverage = self.data.count(axis = 1)
        return
    
    def cal_turnover(self):
        pos = self.weight.mul(self.net_value, axis = 0)        
        self.turnover = pos - (pos * (1. + self.fwdrtn)).shift(1).fillna(0.)
        self.turnover = self.turnover.abs().sum(axis = 1)
        self.turnover = (100*self.turnover / self.net_value).iloc[1:]
        
    def run(self,Type):
        print('calc net value')
        self.cal_net_value(Type)
        print('cal coverage')
        self.cal_coverage()
        print('cal turnover')
        self.cal_turnover()
        return