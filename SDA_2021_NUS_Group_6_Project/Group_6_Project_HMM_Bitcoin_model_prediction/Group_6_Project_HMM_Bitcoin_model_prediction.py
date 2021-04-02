# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:03:12 2021

@author: Zhang Yuxi
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
n_hidden_states=3 
n_latency_days=10
n_steps_frac_change=50
n_steps_frac_high=5
n_steps_frac_low=15
days = 200
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
 
#Plot the predicted prices
test_data = test_data[0: days]
plotdays = test_data['Date']
actual_close_prices = test_data['Close']
 
fig = plt.figure()
 
axes = fig.add_subplot(111)
axes.plot(plotdays, actual_close_prices, 'bo-', label="actual")
axes.plot(plotdays, predicted_close_prices, 'r+-', label="predicted")
axes.set_title('BTC-USD')
 
fig.autofmt_xdate()
 
plt.legend()
plt.show()

