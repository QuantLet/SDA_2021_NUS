[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **Group_6_Project_HMM_Bitcoin_model** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'Group_6_Project_HMM_Bitcoin_model'

Published in: 'SDA_2021_NUS'

Description: 'Construction, State analysis and back test of HMM model'

Keywords: 'HMM, Bitcoin, backtest, states, trading' 

Author: 'Changjie Jin, Ng Zhi Wei Aaron, You Pengxin, Chio Qi Jun, Zhang Yuxi'

Submitted:  '3 April 2021'

Datafile: 'BTC-USD.xlsx'

```

### PYTHON Code
```python

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:52:54 2021

@author: ngzwa
"""
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import warnings
warnings.filterwarnings("ignore")

#data from excel
dataset = pd.read_excel('BTC-USD.xlsx')
column_price = 'Close'
column_volume = 'Volume'
column_high = 'High'
column_low = 'Low'
column_open = 'Open'
asset = 'BTCUSD'
dataset = dataset.shift(1)

#plot initial bitcoin prices
plt.figure(figsize=(10,10))
plt.plot(dataset[column_price])
plt.title(asset)
plt.show()

#hmm brute force method to get the best model using diag covariance (default) and full covariance
def get_best_hmm_model(X, max_states, max_iter = 10000):
    best_score = -(10 ** 10)
    best_state = 0
    
    best_score2 = -(10 ** 10)
    best_state2 = 0
    
    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components = state, covariance_type = "diag", n_iter = max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state

        hmm_model2 = GaussianHMM(n_components = state, covariance_type = "full", n_iter = max_iter).fit(X)
        if hmm_model2.score(X) > best_score2:
            best_score2 = hmm_model2.score(X)
            best_state2 = state        
    
    if best_score > best_score2:
          best_model = GaussianHMM(n_components = best_state, covariance_type = "diag", n_iter = max_iter).fit(X)
    else:
          best_model = GaussianHMM(n_components = best_state2, covariance_type = "full", n_iter = max_iter).fit(X)    
  
    
    return best_model

# General plots of hidden states
def plot_hidden_states(hmm_model, data, X, column_price):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(hmm_model.n_components, 3, figsize = (15, 15))
    colours = cm.flag(np.linspace(0, 1, hmm_model.n_components))
    hidden_states = hmm_model.predict(X)
    
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c = 'grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c = colour)
        ax[0].set_title("#{0} hidden state".format(i))
        ax[0].grid(True)
        
        #futures returns plot i+1
        ax[1].hist(data["future_returns"][mask], bins = 30)
        ax[1].set_xlim([-0.2, 0.2])
        ax[1].set_title("Future return distrbution at #{0} hidden state".format(i))
        ax[1].grid(True)
        
        # #futures return cumm sum
        ax[2].plot(data["future_returns"][mask].cumsum(), c = colour)
        ax[2].set_title("Cumulative future returns at #{0} hidden state".format(i))
        ax[2].grid(True)
    
    plt.tight_layout()
    
    
# Statistical features of states
def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m, m + h

def compare_hidden_states(hmm_model, cols_features, conf_interval, iters = 10000):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize = (15, 15))
    colours = cm.flag(np.linspace(0, 1, hmm_model.n_components))
    
    for i in range(0, hmm_model.n_components):
        mc_df = pd.DataFrame()
    
        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features
    
        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color = colours[i])
            axs[k][i].set_title(cols_features[k] + " (state " + str(i) + "): " + str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)
            
    plt.tight_layout()
    

# Feature params
future_period = 1

#Statistical features

# Create features
cols_features = ['frac_change','frac_high','frac_low']
dataset['last_return'] = dataset[column_price].pct_change()
dataset['frac_change'] = (dataset[column_price] - dataset[column_open]) /dataset[column_open]
dataset['frac_high'] = (dataset[column_high] - dataset[column_open]) /dataset[column_open]
dataset['frac_low'] = (dataset[column_open] - dataset[column_low]) /dataset[column_open]

dataset["future_returns"] = dataset[column_price].pct_change(future_period).shift(-future_period)

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna()

# Plot features
plt.figure(figsize=(20,10))
fig, axs = plt.subplots(len(cols_features), 1, figsize = (15, 15))
colours = cm.rainbow(np.linspace(0, 1, len(cols_features)))
for i in range(0, len(cols_features)):
    axs[i].plot(dataset.reset_index()[cols_features[i]], color = colours[i])
    axs[i].set_title(cols_features[i])
    axs[i].grid(True)

plt.tight_layout()

# Split the data on sets
train_ind = int(np.where(dataset['Date'] == '2017-01-01')[0])
train_set = dataset[cols_features].values[:train_ind]
test_set = dataset[cols_features].values[train_ind:]

#price data
train_set_price = dataset[column_price].values[:train_ind]
test_set_price = dataset[column_price].values[train_ind:]

model = get_best_hmm_model(X = train_set, max_states = 3, max_iter = 10000)
print("Best model with {0} states ".format(str(model.n_components)))

#plot and visualise the hidden states
plot_hidden_states(model, dataset[:train_ind].reset_index(), train_set, column_price)

#compare the hidden states
compare_hidden_states(model, cols_features, conf_interval=0.95)

hidden_states = model.predict(train_set)
size = len(hidden_states)
data_train = dataset[:train_ind]

count0 = 0
count1 = 0
count2 = 0

sum_ret_0 = 0
sum_ret_1 = 0
sum_ret_2 = 0
    
for i in range(0, size-1):

    if hidden_states[i] == 0:
        sum_ret_0 += data_train["future_returns"].values[i]
        count0 += 1
            
    elif hidden_states[i] == 1:
        sum_ret_1 += data_train["future_returns"].values[i]
        count1 += 1
    else:
        sum_ret_2 += data_train["future_returns"].values[i]
        count2 += 1
       

best_avg = max(sum_ret_0 , sum_ret_1, sum_ret_2)
worst_avg = min(sum_ret_0 , sum_ret_1, sum_ret_2)

if (best_avg == sum_ret_0):
    best_state = 0
elif (best_avg == sum_ret_1):
    best_state = 1
else:
    best_state = 2
    
if (worst_avg == sum_ret_0):
    worst_state = 0
elif (worst_avg == sum_ret_1):
    worst_state = 1
else:
    worst_state = 2

if(best_state + worst_state == 3):
    ran_state = 0
elif( best_state+ worst_state == 1):
    ran_state = 2
elif(best_state + worst_state == 2):
    ran_state = 1
        
#Run backtest
#exit at rand state
def backtest(hmm_model, testdata, testset):
    #init portfolio
    size = len(testdata)
    Portfolio = 0
    Port_pnl = np.zeros(size)
    states = hmm_model.predict(testset)
    long_pos = 0
    short_pos = 0
    pos_val = 100000
    
    #states definition
    long_state = best_state
    short_state = worst_state
    rand_state = ran_state
    
    #conditions
    for t in range(0, size):
        if( states[t] == short_state and Portfolio == 0 and long_pos == 0 and short_pos == 0):
            Portfolio = testdata[t]
            short_pos = 1
        elif (states[t] == long_state and Portfolio == 0 and short_pos == 0 and long_pos == 0):
            Portfolio = testdata[t]
            long_pos = 1
            
        #calculation of PnL
        elif (states[t] == rand_state and Portfolio != 0 and  short_pos == 1):
            Port_pnl[t] = (Portfolio - testdata[t]) / testdata[t] * pos_val
            Portfolio = 0
            short_pos = 0
        elif(states[t] == rand_state and Portfolio != 0 and long_pos == 1):
            Port_pnl[t] = (testdata[t] - Portfolio) / testdata[t] * pos_val
            Portfolio = 0
            long_pos = 0
        
        total_pnl = Port_pnl.cumsum() 
    
    return total_pnl

totalpnl = backtest(model,test_set_price, test_set)

#run model on test set
test_states = model.predict(test_set)

#plot OOS states
plt.figure(figsize=(10,10))
plt.plot(test_states)
plt.title('Out of sample states')
plt.show()

#Plot states plot for test set data
plot_hidden_states(model, dataset[train_ind:].reset_index(), test_set, column_price)

#plot initial bitcoin prices
plt.figure(figsize=(10,10))
plt.plot(totalpnl)
plt.title('Backtest Pnl')
plt.show()

```

automatically created on 2021-04-15