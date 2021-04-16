# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:03:12 1911

@author: Jin Changjie
"""
#pip install pystan
#conda install -c conda-forge fbprophet -y
# pip install keras
# pip install tensorflow
#pip install statsmodels
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

days = 100
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

test_size=0.66
train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

pro_train_data = train_data;
pro_train_data["frac_change"]= (train_data["Close"] - train_data["Open"]) / train_data["Open"];
pro_train_data = pro_train_data[["Date","frac_change"]]
lstm_train_data = pro_train_data;
arima_train = lstm_train_data
pro_test_data = test_data;
pro_test_data["frac_change"]= (test_data["Close"] - test_data["Open"]) / test_data["Open"];
pro_test_data = pro_test_data[["Date","frac_change","Open","Close"]][0:days]
lstm_test_data = pro_test_data ;
arima_test = lstm_test_data





##################################################################################
# Prophet Part
#pip install pystan
#conda install -c conda-forge fbprophet -y
#Prophet
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import warnings
days = 100
warnings.filterwarnings('ignore')

def FB(data: pd.DataFrame) -> pd.DataFrame:
    
    df = pd.DataFrame({
    'ds': data.Date,
    'y': data.frac_change,
    })
    
    df['cap'] = data.frac_change.values.max()
    df['floor'] = data.frac_change.values.min()

    m = Prophet(
        changepoint_prior_scale=0.05, 
        daily_seasonality=False,
        yearly_seasonality=True, #年周期性
        weekly_seasonality=True, #周周期性
        growth="logistic",
    )
    
#     m.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.1)#月周期性
    #m.add_country_holidays(country_name='CN')#中国所有的节假日    
    
    m.fit(df)
    
    future = m.make_future_dataframe(periods=days, freq='D')#预测时长
    future['cap'] = data.frac_change.values.max()
    future['floor'] = data.frac_change.values.min()

    forecast = m.predict(future)
    
    fig = m.plot_components(forecast)
    fig1 = m.plot(forecast)
    
    return forecast

result_frac = FB(pro_train_data)

#Plot the predicted pricespro_
temp = result_frac.iloc[-len(pro_test_data["Date"]):]
temp["Pre_Close"] = pro_test_data['Open'].values*(1+result_frac.iloc[-len(pro_test_data["Date"]):]["yhat"].values)
result_frac["Pre_Close"] = temp["Pre_Close"]
pro_predicted_close_prices = result_frac.iloc[-len(pro_test_data["Date"]):]["Pre_Close"]

actual_close_prices = pro_test_data['Close']
 

# fig = plt.figure()
# axes = fig.add_subplot(111)
# axes.plot(plotdays, actual_close_prices, 'bo-', label="actual")
# axes.plot(plotdays, pro_predicted_close_prices, 'r+-', label="predicted")
# axes.set_title('BTC-USD')
 
# fig.autofmt_xdate()
 
# plt.legend()
# plt.show()

# compare RMSE & ME
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rms_prophet = mean_squared_error(actual_close_prices, pro_predicted_close_prices, squared=False)
me_prophet = mean_absolute_error(actual_close_prices, pro_predicted_close_prices)


####################################################################################
#LSTM Part
# pip install keras
# pip install tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

n_in = 10 #历史数量
n_out = 1 #预测数量
n_features = 1
# n_test = 1
n_val = days
n_epochs = 300

def split_data(x, y, n_test: int):
    x_train = x[:-n_val-n_out+1]
    x_val = x[-n_val:]
    y_train = y[:-n_val-n_out+1]
    y_val = y[-n_val:]
    return x_train, y_train, x_val, y_val

def build_train(train, n_in, n_out): 
    train = train.drop(["Date"], axis=1)
    X_train, Y_train = [], []
    for i in range(train.shape[0]-n_in-n_out+1):
        X_train.append(np.array(train.iloc[i:i+n_in]))
        Y_train.append(np.array(train.iloc[i+n_in:i+n_in+n_out]["frac_change"]))     
    return np.array(X_train), np.array(Y_train)

def build_lstm(n_in: int, n_features: int):   
    model = Sequential()
    model.add(LSTM(12, activation='relu', input_shape=(n_in, n_features)))
    model.add(Dropout(0.3))
    model.add(Dense(n_out))
    model.compile(optimizer='adam', loss='mae')
    return model

def model_fit(x_train, y_train, x_val, y_val, n_features):
    model = build_lstm(
        n_in   = n_in,
        n_features= 1
    )
    model.compile(loss='mae', optimizer='adam')
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=180, verbose=1,  validation_data=(x_val, y_val))
    m = model.evaluate(x_val, y_val)
    print(m)
    return model

def minmaxscaler(data: pd.DataFrame) -> pd.DataFrame:
    volume = data.frac_change.values
    volume = volume.reshape(len(volume), 1)
    volume = scaler.fit_transform(volume)
    volume = volume.reshape(len(volume),)    
    data['frac_change'] = volume  
    return data

scaler = MinMaxScaler(feature_range=(0, 1))


lstm_data = data[["Date"]]
lstm_data["frac_change"]= (data["Close"] - data["Open"]) / data["Open"]
lstm_test = data[["Date","Open","Close"]]
lstm_test["frac_change"]=lstm_data["frac_change"]

x, y = build_train(lstm_data, n_in, n_out)
x_train, y_train, x_val, y_val = split_data(x, y, n_val)
model = build_lstm(n_in, 1)
model = model_fit(x_train, y_train, x_val, y_val, 1)
predict = model.predict(x_val)


scaler = MinMaxScaler(feature_range=(0, 1))
lstm_data = minmaxscaler(lstm_data)

validation = scaler.inverse_transform(predict)
actual = scaler.inverse_transform(y_val)

predict =[]
predict = pd.DataFrame(validation)
predict["actual"] = actual
predict.columns = ["predict_frac_change","Actual_frac"]

predict["Predict"] = lstm_test_data["Open"].values*(1.39+predict["predict_frac_change"].values)


# fig = plt.figure()
# plt.plot(predict["Predict"] )
# plt.plot(predict["Actual"] )

# compare RMSE & ME
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rms_lstm = mean_squared_error(predict["Predict"], actual_close_prices, squared=False)
me_lstm = mean_absolute_error(predict["Predict"], actual_close_prices)


#################################################################################
# ARIMA Part
#pip install statsmodels

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(arima_train["frac_change"])
plot_pacf(arima_train["frac_change"])
plt.show()
 
#平稳性检测
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

model = ARIMA(arima_train["frac_change"], order=(1,0,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


history = [x for x in arima_train["frac_change"]]
predictions = list()
for t in range(days):
    model = ARIMA(history, order=(1,0,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = arima_test["frac_change"].values[t]
    history.append(obs)

predictions = pd.DataFrame(predictions)
predictions.columns = ["frac"]
arima_result = pd.DataFrame()
Arima_Predict = arima_test["Open"].values*(1+predictions["frac"].values)
Arima_Actual = arima_test["Close"].values





# pyplot.plot(Arima_Predict)
# pyplot.plot(Arima_Actual )
# pyplot.show()


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
rms_arima = mean_squared_error(Arima_Actual , Arima_Predict , squared=False)
me_arima = mean_absolute_error(Arima_Actual , Arima_Predict)






#################################################################################
# HMM model

#Set constants for model
test_size=0.66
n_hidden_states=3 
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
for day_index in  tqdm(range(days)):
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
axes.plot(plotdays, predicted_close_prices, 'r+-', label="HMM predicted")
axes.plot(plotdays, Arima_Predict, label="ARIMA predicted")
axes.plot(plotdays, predict["Predict"], label="LSTM predicted")
axes.plot(plotdays, pro_predicted_close_prices, label="Prophet predicted")

axes.set_title('BTC-USD')
fig.autofmt_xdate()
plt.legend()
plt.show()

rms_HMM = mean_squared_error(actual_close_prices,predicted_close_prices, squared=False)
me_HMM = mean_absolute_error(actual_close_prices, predicted_close_prices)


compare = pd.DataFrame()
compare["Model"] = ["HMM(3 States)","Prophet","Simple LSTM","ARIMA"]
compare["RMSE"] = [rms_HMM,rms_prophet,rms_lstm,rms_arima]
compare["ME"] = [me_HMM,me_prophet,me_lstm,me_arima]
fig = plt.figure()
plt.title('Model Comparison')
scale_ls = range(4)
index_ls = compare["Model"]
plt.xticks(scale_ls,index_ls) 
p1,=plt.plot(compare["RMSE"])
p2,=plt.plot(compare["ME"])
for a,b in zip([0.0,1.0,2.0,3.0],compare["RMSE"]):
    plt.text(a, b, '%.2f' % b,fontsize=10)
for a,b in zip([0.0,1.0,2.0,3.0],compare["ME"]):
    plt.text(a, b, '%.2f' % b,fontsize=10)
plt.legend([p2, p1], ["MAE", "RMSE"], loc='best')
plt.show()



    


