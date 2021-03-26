from astropy.time.utils import split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv')

data.fillna(method = 'ffill', inplace = True)
data.fillna(data.mean(), inplace = True)

x = data.drop(['id', 'y'], axis=1)
y = data.y

##### split by part and predict on next split #####
### block time series split
gap=100000
test_day = 100
for i in range(gap, x.shape[0], gap):
     train_index = list(range(i-gap, i-test_day))
     test_index = list(range(i-test_day, i))
     print("TRAIN:", i-test_day, "TEST:", i)
     x_train, x_test = x.loc[train_index], x.loc[test_index]
     y_train, y_test = y[train_index], y[test_index]

     '''lr = LinearRegression(normalize=True)
     lr.fit(x_train, y_train)

     train_pred = lr.predict(x_train).reshape(-1, 1)
     binarizer = Binarizer(threshold=0.5).fit(train_pred)
     train_pred = binarizer.transform(train_pred)
     print(accuracy_score(y_train, train_pred))

     test_pred = lr.predict(x_test).reshape(-1, 1)
     binarizer = Binarizer(threshold=0.5).fit(test_pred)
     test_pred = binarizer.transform(test_pred)
     print(accuracy_score(y_test, test_pred))'''

     ### train xgb model
     from xgboost import XGBClassifier

     model_xgb = XGBClassifier(
          learning_rate=0.1,
          n_estimators=50,
          max_depth=5,
          min_child_weight=1,
          gamma=0,
          subsample=0.8,
          colsample_bytree=0.8,
          objective='binary:logistic',
          scale_pos_weight=1,
          seed=27,
          verbosity=0
     )
     model_xgb.fit(x_train, y_train)

     print('In sample acc: ', accuracy_score(y_train, model_xgb.predict(x_train)))
     print('Out sample acc: ', accuracy_score(y_test, model_xgb.predict(x_test)))

##### predict one day forward #####
### train xgb model

     # train test split
     len = data.shape[0]
     test_len = len * 4 // 5
     train = data.iloc[:test_len, :]
     test = data.iloc[test_len:, :]

     x_train = train.drop(['id','y'], axis=1)
     x_test = test.drop(['id','y'], axis=1)

     pred_start = 50000
     pred_end = 50100

     x = x_train.loc[:pred_start-1,:]
     y = x.last_price.shift(1)
     y.fillna(method='bfill', inplace=True)


     from xgboost import XGBClassifier
     model_xgb = XGBClassifier(
          learning_rate=0.1,
          n_estimators=50,
          max_depth=5,
          min_child_weight=1,
          gamma=0,
          subsample=0.8,
          colsample_bytree=0.8,
          scale_pos_weight=1,
          seed=27,
          verbosity=2
     )
     model_xgb.fit(x, y)

     real = x_train.last_price.loc[pred_start:pred_end]
     pred_xgb = model_xgb.predict(x_train.loc[pred_start-1:pred_end-1,:])

### train neural network mlp model
     from tensorflow.keras.models import Sequential
     from tensorflow.python.keras.layers import Dense, Dropout
     import tensorflow.keras.optimizers as optimizers

     n_input = x.shape[1]
     optimizer = optimizers.Adam(lr=0.0001)

     model_mlp = Sequential()
     model_mlp.add(Dense(10, input_dim=n_input, activation='relu'))
     model_mlp.add(Dense(1))
     model_mlp.compile(loss='mse', optimizer=optimizer)

     history = model_mlp.fit(x, y, batch_size=32, epochs=50)

     real = x_train.last_price.loc[pred_start:pred_end]
     pred_mlp = model_mlp.predict(x_train.loc[pred_start - 1:pred_end - 1, :])

     plt.plot(real.values)
     plt.plot(pred_xgb)
     plt.plot(pred_mlp)
     plt.legend(['Real', 'XGBoost', 'NN-MLP'])
