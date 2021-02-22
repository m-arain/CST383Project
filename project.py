# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:13:29 2021

@author: Robert Meis
@team Members: Jason Contreras, Mohammad Arain
CST-383 Project: Spam vs. Ham Classifier 
References: https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html
https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
https://pandas.pydata.org/ (multiple pages)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

df = pd.read_csv('spam_ham_dataset.csv') #keep both the .csv file and this file in the same folder or update file path
df0 = df; #copy df in case df gets modified incorrectly

df.columns = ['number', 'label', 'subject', 'label_num'] #drop 'number' (= bias) and 'label' (not needed) columns
df.drop(['number', 'label'], axis=1, inplace=True) #

#extract label and target vectors
X = df['subject'].str.strip('Subject: ') #X = labels vector. Strip word 'Subject: ' which appears in front of each email subject (not needed/biasing)
y = df['label_num'] #target vector

#weight words using tfidVectorizer for X_train and X_test
vec = TfidfVectorizer()
X = vec.fit_transform(X)
X = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

#separate X and y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#use kNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

'''
#use Naive-Bayes (testing)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
predictions = model.predict(y_test)
'''


