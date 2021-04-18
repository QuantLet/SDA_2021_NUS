[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **High_Frequency_Xgboost_MLP** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'High_Frequency_Xgboost_MLP'

Published in: 'SDA_2021_NUS/SDA_2021_NUS_High_Xgboost_MLP'

Description: 'Using Xgboost and MLP to do high frequency prediction of stock index futures'

Keywords: 'High frequency, CSI 300, TFT, Xgboost, MLP' 

Author: 'Cheng Tuoyuan, Wang Duyue, Wang Wenbo, Zheng Zhongyi'

Submitted:  '07.April 2021'

Datafile: 'CSI 300 index futures 500ms data'

```

### PYTHON Code
```python

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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('201902_IF1903.csv')

time = data.TDATETIME
data.drop(['IFCD', 'TDATETIME'], axis=1, inplace=True)
price = data.LASTPRICE

data = (data - data.mean()) / data.std()

##### predict one day forward #####
### train xgb model

# train test split
len = data.shape[0]
test_len = len * 9 // 10
X = data.iloc[:test_len, :]

window = 10
pred_start = 100000
pred_end = pred_start + window

x = X.iloc[:pred_start,:]
y = price.loc[x.index].shift(window)
y.fillna(method='bfill', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=False, random_state=1, test_size=0.1)

from xgboost import XGBRegressor
model_xgb = XGBRegressor(
     silent=0,
     learning_rate=0.3,
     min_child_weight=1,
     max_depth=10,  #
     gamma=10,
     subsample=1,
     max_delta_step=0,
     colsample_bytree=0.5,
     colsample_bylevel=0.5,
     reg_lambda=0.5,
     n_estimators=50,
     seed=1000
)
model_xgb.fit(X_train, y_train)
print(r2_score(y_test, model_xgb.predict(X_test)))

pred_xgb = model_xgb.predict(X_test.iloc[:10,:])

### train neural network mlp model
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

tf.random.set_seed(1)
optimizer = tf.optimizers.Adam(lr=0.005)

model = Sequential()
model.add(Dense(30, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=64, epochs=50,verbose=1)
print(r2_score(y_test, model.predict(X_test)))

pred_mlp = model.predict(X_test.iloc[:10,:])

real = y_test.iloc[:10]
real_before = y_train.iloc[-25:]

plt.plot(np.concatenate((real_before.values, real.values)))
plt.plot(np.concatenate((real_before.values, pred_xgb)))
plt.plot(np.concatenate((real_before.values, pred_mlp.reshape(1,-1)[0])))
plt.legend(['Real', 'XGBoost', 'NN-MLP'])

```

automatically created on 2021-04-18