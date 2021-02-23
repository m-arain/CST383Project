# -*- coding: utf-8 -*-
"""
@author: Robert Meis
@team Members: Jason Contreras, Mohammad Arain
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
from sklearn.tree import DecisionTreeClassifier
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

#Decision Tree Model
clf = DecisionTreeClassifier(max_depth=15, random_state=0)
clf.fit(X_train, y_train)
dt_predict = clf.predict(X_test)

#print accuracy
d_tree_accuracy = (dt_predict == y_test).mean()
print("Decision Tree Accuracy {:.2f}".format(d_tree_accuracy))

#Naive-Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)
nb_predict = clf.predict(X_test)

#print accuracy
nb_accuracy = (nb_predict == y_test).mean()
print('Naive-Bayes Accuracy {:.2f}'.format(nb_accuracy))


#Notes

'''kNN was tested but omitted due to limited accuracy and significant run time
#use kNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test[:50])
print(predictions, y_test[:50])
'''


