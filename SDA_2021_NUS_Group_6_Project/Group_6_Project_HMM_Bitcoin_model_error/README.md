[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **Group_6_Project_HMM_Bitcoin_model_error** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'Group_6_Project_HMM_Bitcoin_model_error'

Published in: 'SDA_2021_NUS'

Description: 'Calculation of the prediction error of the constructed model'

Keywords: 'HMM, Bitcoin, prediction, RMSE, states' 

Author: 'Changjie Jin, Ng Zhi Wei Aaron, You Pengxin, Chio Qi Jun, Zhang Yuxi'

Submitted:  '3 April 2021'

Datafile: 'BTC-USD.xlsx'

```

### PYTHON Code
```python

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:41:45 2021

@author: QJ
"""


import os
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm


os.chdir(os.path.dirname(os.path.abspath(__file__)))


warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
plt.style.use('ggplot')

rmse_arr = []
nrmse_err_arr = []
mean_err_arr = []
nmean_err_arr = [] 


for i in range(2,6):
    #Read Bitcoin data
    data = pd.read_excel('BTC-USD.xlsx')
    open_price = np.array(data['Open'])
    close_price = np.array(data['Close'])
    high_price = np.array(data['High'])
    low_price = np.array(data['Low'])
     
    # Compute the fraction change in close, high and low prices
    # which would be used a feature
    frac_change = (close_price - open_price) / open_price
    frac_high = (high_price - open_price) / open_price
    frac_low = (open_price - low_price) / open_price
    
    #Set constants for model
    test_size=0.66
    n_hidden_states=i 
    n_latency_days=10
    n_steps_frac_change=50
    n_steps_frac_high=5
    n_steps_frac_low=15
    days = 100
    #Construct logging 
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    
    #Set up the model
    hmm_model = GaussianHMM(n_components=n_hidden_states)
    
    #Split the data for training and testing
    train_data, test_data = train_test_split(
        data, test_size=test_size, shuffle=False)
    
    #Report the log
    logger.info('>>> Extracting Features')
    #Prepare model input
    feature_vector = np.column_stack((frac_change, frac_high, frac_low))
    #Report the log
    logger.info('Features extraction Completed <<<')
    #Fit the model
    hmm_model.fit(feature_vector)
    
    #Output the states 
    lpa, hsa = hmm_model.decode(feature_vector)
    df = pd.DataFrame()
    df['hidden states'] = hsa
    df.to_csv("hidden states.csv")
    
    #Compute all possible outcomes
    frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
    frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
    frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
     
    possible_outcomes = np.array(list(itertools.product(
        frac_change_range, frac_high_range, frac_low_range)))
    
    
    #predict close prices for days
    predicted_close_prices = []
    for day_index in tqdm(range(days)):
        #get most probable outcome
        previous_data_start_index = max(0, day_index - n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = test_data.iloc[previous_data_end_index: previous_data_start_index]
        
        popen_price = np.array(previous_data['Open'])
        pclose_price = np.array(previous_data['Close'])
        phigh_price = np.array(previous_data['High'])
        plow_price = np.array(previous_data['Low'])
        
        pfrac_change = (pclose_price - popen_price) / popen_price
        pfrac_high = (phigh_price - popen_price) / popen_price
        pfrac_low = (popen_price - plow_price) / popen_price
        
        previous_data_features = np.column_stack((pfrac_change, pfrac_high, pfrac_low))
         
        outcome_score = []
        for possible_outcome in possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(hmm_model.score(total_data))
        most_probable_outcome = possible_outcomes[np.argmax(
            outcome_score)]
        
        #Predict close price
        open_price = test_data.iloc[day_index]['Open']
        predicted_frac_change, _, _ = most_probable_outcome
        predicted_close_price= open_price * (1 + predicted_frac_change)
        
        predicted_close_prices.append(predicted_close_price)
        
        
    test_data = test_data[0: days]
    plotdays = test_data['Date']
    actual_close_prices = test_data['Close']    
    err = abs( np.subtract(np.array(predicted_close_prices), np.array(actual_close_prices)))
      

    mean_err = np.mean(err)
    sd_err = np.std(err)
    
    mean_price = np.mean(actual_close_prices)
    
    rmse = np.sqrt((sum(err**2) )/len(err))
    
    nrmse = rmse/mean_price
    

    rmse_arr.append(rmse)
    nrmse_err_arr.append(nrmse)
    mean_err_arr.append(mean_err)
    nmean_err_arr.append(mean_err/mean_price)
    
    
#    print("Root Mean Square Error: ", rmse)
#    print("Normalised Root Mean Square Error: ", nrmse)
#    print("Mean Error: ", mean_err)
#    print("Normalised Mean Error: ", mean_err/mean_price)
    
    
    
states = np.array([2,3,4,5])
dct = {'rmse' : rmse_arr,
       'nrmse' : nrmse_err_arr,
       'mean_err' : mean_err_arr,
       'nmean_err' : nmean_err_arr}

df = pd.DataFrame(dct, columns = ['rmse', 'nrmse', 'mean_err', 'nmean_err'], index = [2,3,4,5])

plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set1')


plt.xticks(np.array([2,3,4,5]))
for a,b in zip([2,3,4,5],rmse_arr):
    plt.text(a, b, '%.2f' % b,fontsize=10)
for a,b in zip([2,3,4,5],mean_err_arr):
    plt.text(a, b, '%.2f' % b,fontsize=10)
plt.plot(states, df['rmse'], marker = '', linewidth = 1, alpha = 0.9, label = 'RMSE')
plt.plot(states, df['mean_err'], marker = '', linewidth = 1, alpha = 0.9, label = 'MAE')
plt.legend(loc =2, ncol = 1)
plt.xlabel('Number of Hidden States')
plt.ylabel('Normalised Errors')



plt.xticks(np.array([2,3,4,5]))
for a,b in zip([2,3,4,5],nrmse_err_arr):
    plt.text(a, b, '%.2f' % b,fontsize=10)
for a,b in zip([2,3,4,5],nmean_err_arr):
    plt.text(a, b, '%.2f' % b,fontsize=10)
plt.plot(states, df['nrmse'], marker = '', linewidth = 1, alpha = 0.9, label = 'RMSE')
plt.plot(states, df['nmean_err'], marker = '', linewidth = 1, alpha = 0.9, label = 'MAE')
plt.legend(loc =2, ncol = 1)
plt.xlabel('Number of Hidden States')
plt.ylabel('Normalised Errors')









```

automatically created on 2021-04-15