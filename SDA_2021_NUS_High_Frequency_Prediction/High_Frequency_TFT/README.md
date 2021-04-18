[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **High_Frequency_TFT** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'High_Frequency_TFT'

Published in: 'SDA_2021_NUS/SDA_2021_NUS_High_Frequency_Prediction'

Description: 'Using TFT to do high frequency prediction of stock index futures'

Keywords: 'High frequency, CSI 300, TFT, Xgboost, MLP' 

Author: 'Cheng Tuoyuan, Wang Duyue, Wang Wenbo, Zheng Zhongyi'

Submitted:  '07.April 2021'

```

### PYTHON Code
```python


import pandas as pd

from sklearn.metrics import roc_auc_score as auc
# load data
data = pd.read_csv('201902_IF1903.csv')

time = data.TDATETIME
data.drop(['IFCD', 'TDATETIME'], axis=1, inplace=True)
data.loc[:, 'const'] = ' '
data.loc[:, 'idx'] = data.index

import copy
from pathlib import Path
import warnings

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

max_prediction_length = 15
max_encoder_length = 24

training = TimeSeriesDataSet(
    data,
    time_idx="idx",
    target="LASTPRICE",
    group_ids=['const'],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["idx"],
    static_categoricals=["const"],
    time_varying_unknown_reals=[
        'CP', 'CQ', 'CM', 'S5', 'S4', 'S3', 'S2',
       'S1', 'B1', 'B2', 'B3', 'B4', 'B5', 'SV5', 'SV4', 'SV3', 'SV2', 'SV1',
       'BV1', 'BV2', 'BV3', 'BV4', 'BV5', 'OPNINTRST', 'UPPERLMTPRC',
       'LOWERLMTPRC', 'HIGHPRICE', 'LOWPRICE', 'PRESTLMTPRC'
    ],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missings=True
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.02,
    hidden_size=20,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=10,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=30,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)



# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()


# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

for idx in range(15):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)


'''data = pd.read_csv('train.csv')
# train test split
len  = data.shape[0]
test_len = len * 4 // 5
train = data.iloc[:test_len, :]
test = data.iloc[test_len:, :]

x_train = train.drop('y',axis=1)
y_train = train.y

x_test = test.drop('y',axis=1)
y_test = test.y
# fill missing value
data.drop('id', axis = 1, inplace= True)
data.fillna(data.mean(), inplace = True)



### train xgb model
from xgboost import XGBClassifier
model_xgb = XGBClassifier(
    verbosity=2,
    booster='gblinear',
    max_depth=3
)
model_xgb.fit(x_train, y_train)

print('In sample auc: ', auc(y_train, model_xgb.predict(x_train)))
print('Out sample auc: ', auc(y_test, model_xgb.predict(x_test)))


### train neural network mlp model
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import tensorflow.keras.optimizers as optimizers
n_input = train.shape[1]-1
optimizer = optimizers.Adam(lr=0.001)

model_mlp = Sequential()
model_mlp.add(Dense(100, input_dim = n_input, activation = 'relu'))
model_mlp.add(Dropout(0.15))
model_mlp.add(Dense(1, activation= 'sigmoid'))
model_mlp.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = 'accuracy')

history = model_mlp.fit(x_train, y_train, batch_size=16, epochs=1, validation_data = (x_test,y_test))

test_pred = model_mlp.predict(x_test)
train_pred = model_mlp.predict(x_train)

trans_binary = lambda x: 1 if x>=0.5 else 0
test_pred = list(map(trans_binary, test_pred))
train_pred = list(map(trans_binary, train_pred))


print('In sample auc: ', auc(y_train, train_pred))
print('Out sample auc: ', auc(y_test, test_pred))'''


```

automatically created on 2021-04-18