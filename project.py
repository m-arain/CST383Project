# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:13:29 2021

@author: Robert Meis
@team Members: Jason Contreras, Mohammad Arain
CST-383 Project: Spam vs. Ham Classifier 
"""

import pandas as pd
import numpy as np

df = pd.read_csv('spam_ham_dataset.csv') #keep both the .csv file and this file in the same folder or update file path
df0 = df; #copy df in case df gets modified incorrectly

print(df['label_num'].value_counts())
print(df['label_num'].value_counts()[1] / df.shape[0])
df.columns = ['number', 'label', 'subject', 'label_num']
print(df.columns)
df.drop('number', axis=1, inplace=True) #we may want to drop the 'number' column in case it biases the ML algorithm to think higher numbered emails are more significant
print(df.head(20))

#Split data set into a training set (80% of data) and a test set (20% of data)
print(df.shape[0] * 0.80)
training_set = df[: 4100]
test_set = df[4101:]
print(training_set.shape[0], test_set.shape[0])

#Extract features array and target (labels) array
x_features = training_set.drop('label_num', axis=1)
y_target = training_set['label_num']
print(x_features.head(10))
print(x_features.columns)




