# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:25:11 2021

@author: Group 9
"""

import numpy as np
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score as acc_rate
import pandas as pd
# use this line in terminal if graphviz does not work: conda install python-graphviz 


#Read Data
p2p=pd.read_csv('./p2p.csv') 

all_inputs = p2p[['ratio001','ratio003','ratio005','ratio006','ratio011','ratio017','ratio018','ratio019','DPO','DSO','turnover','ratio037']].values
all_outputs= p2p[['status']].values


# randomly choose 1/3 of samples as testing data
np.random.seed(123)
test_idx = np.random.randint(0, len(all_outputs), len(all_outputs) // 3)


# training data
train_data_1 = np.delete(all_inputs, test_idx, axis=0)
train_target_1 = np.delete(all_outputs, test_idx)


# testing data
test_data_1 = all_inputs[test_idx]
test_target_1 = all_outputs[test_idx]


# train the model
# initial an decision tree classifier object with given arguments
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  splitter='best')
clf.fit(X=train_data_1, y=train_target_1)

print('\nThe target test data set is:\n', test_target_1)
print('\nThe predicted result is:\n', clf.predict(test_data_1))
print('\nAccuracy rate is:\n', acc_rate(test_target_1, clf.predict(test_data_1)))


# visualizing the tree
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=['ratio001','ratio003','ratio005','ratio006','ratio011','ratio017','ratio018','ratio019','DPO','DSO','turnover','ratio037'],
                                class_names=['0','1'],
                                filled=True,
                                rounded=True,
                                impurity=False,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("p2p", view = True)