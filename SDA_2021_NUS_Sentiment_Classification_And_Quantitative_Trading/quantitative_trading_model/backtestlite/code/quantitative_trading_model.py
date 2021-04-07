from config import *
from helper import *
from backtestlite import *
import os


def singlefactor(factor,save_path_factor,factor_path_all,Type,direction,fwdrtn,mkt_index,cycle, start, end, quantile,win,year, statuslimit):

    
    factor_path = os.path.join(factor_path_all, factor + '.csv')
    # Run the backtest
    if Type != "ALL":
        
        save_path_type = os.path.join(save_path_factor, Type)
        if not os.path.exists(save_path_type):
            os.makedirs(save_path_type) 
            
        B = Backtestlite(factor_path, direction, fwdrtn, mkt_index, save_path_type, cycle, start, end, quantile, statuslimit)
        B.run(Type)

        # Drawing
        pic = DrawPicture(factor,direction, B.data,  B.fwdrtn, B.data.index , mkt_index,  B.rtn_data, B.coverage, B.turnover, save_path_type ,Type)
        pic.main(win, year, B.rtn_data.columns)


for factor in factors:
    # save path for each factor
    save_path_factor = os.path.join(save_path_all, factor) + '/'
    if not os.path.exists(save_path_factor):
        os.makedirs(save_path_factor)
#
    # factor direction, default is Descending
    if(factor in directions.index):
        direction = directions.loc[factor, "direction"]
    else:
        direction = "Not Defined"
    if(direction != "Ascending"):
        direction = "Descending"
    singlefactor(factor,save_path_factor,factor_path_all,Type,direction,fwdrtn,mkt_index,cycle, start, end, quantile,win,year, statuslimit)

if __name__ == "__main__":
    plt.switch_backend('agg')
    working_directory = r'G:\Github\FE5225 Group8 Project\SDA_2021_NUS\Project_Group8\quantitative_trading_model\backtestlite'
    # Directory to import stock return, market return & trade status.
    path = os.path.join(working_directory, "csv")
    # Directory to import factor.
    factor_path_all = os.path.join(working_directory, 'factors')
    # Directory to output the result.
    save_path_all = os.path.join(working_directory, 'result')


    factor_csv = os.listdir(factor_path_all)
    factors = [filename[:-4] for filename in factor_csv]  # MAKE SURE the number here is right and we can find the right file.
    factors_exist = os.listdir(save_path_all)
    # factors = list(set(factors) - set(['.DS_S', '', '.DS_Store']) - set(factors_exist))  # Please delete result folders with the same name.
    factors.sort()
    # del factor_csv, factors_exist

    # Define the direction of factors.
    directions = pd.read_csv(os.path.join(path, 'factororder.csv'), index_col=0)

    ###############################################################

    # Parameters of backtest.
    start = '2018-01-02'  # Time horizon, can be vacation, progrmme will extract the maximum subset.
    end = '2019-09-18'
    quantile = 5
    cycle = 1  # Adjustment Cycle (trading frequency), delay for holidays. DON'T change this parameter.

    win = 220
    year = 2018

    fwdrtn = pd.read_csv(os.path.join(path, "foward_return.csv"), index_col=0, parse_dates=True).loc[start:end][::cycle]
    mkt_index = pd.read_csv(os.path.join(path, "market_forward_return.csv"), index_col=0, parse_dates=True).loc[str(start):str(end), "market_forward_return"][::cycle]

    # up down limit: 1 is close at up limit price, -1 is close at down limit price,0 for normal close price and nan means not in the market.
    # updownlimit = pd.read_csv(os.path.join(path, "UpDownLimitStatus.csv"), index_col = 0, parse_dates = True).loc[start:end][::cycle]

    # trade status: 0 means suspension,1 means on transaction, nan means not in the market.
    TradeStatus = pd.read_csv(os.path.join(path, "trade_status.csv"), index_col=0, parse_dates=True).loc[start:end][::cycle]

    # Stock chosen pool, can't be in suspension.
    status = (TradeStatus == 1)
    # status = (updownlimit == 0) & (TradeStatus == 1)
    statuslimit = pd.DataFrame(index=fwdrtn.index, columns=fwdrtn.columns)
    statuslimit[status == 1] = 1

    # Choose long short:"LS" or future hedge:"L" , "ALL" for L & LS.
    Type = "LS"

    for factor in factors:
        # save path for each factor
        save_path_factor = os.path.join(save_path_all, factor) + '/'
        if not os.path.exists(save_path_factor):
            os.makedirs(save_path_factor)
        #
        # factor direction, default is Descending
        if (factor in directions.index):
            direction = directions.loc[factor, "direction"]
        else:
            direction = "Not Defined"
        if (direction != "Ascending"):
            direction = "Descending"
        singlefactor(factor, save_path_factor, factor_path_all, Type, direction, fwdrtn, mkt_index, cycle, start, end,
                     quantile, win, year, statuslimit)
