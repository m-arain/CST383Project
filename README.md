# CST 383 Final Project
#### Spam/Ham Machine Learning Classifier
#### Jason Contreras, Mohammad Arain, Robert Meis
#### CST 383 Data Science
#### Spring 2021
#### Professor Ergezer 

### Video Presentation: https://www.youtube.com/watch?v=9QYtnXWecbg

@Team Members: Robert Meis, Jason Contreras, Mohammad Arain


### Introduction
The purpose of this project is to create a Spam/Ham Classifier with high accuracy. Spam/Ham classifiers are a widely used technique for e-mail filtering. This project was undertaken because it piqued our interest, is popular in our field of computer science and is applicable towards our class curriculum.


### Selection of Data
Original source of the dataset: https://www.kaggle.com/venky73/spam-mails-dataset. <br/>
Data consists of a publicly available dataset containing a total of 5171(~5000) Enron emails.<br/>
Before data munging, original data consists of four columns: number, label, text, label_num

#### Data Fields
1. Unamed:0 (Number) - Unique id of Message
2. Label - Type of message (Spam or Ham)
3. Text - Actual Message
4. label_num - Type of message (0 = ham, 1 = Spam)

![Dataset](https://github.com/m-arain/CST383Project/raw/main/Dataset.png 'Dataset')


### Machine Learning
#### Feature Engineering 
The 'email number' column is dropped to avoid bias and 'label' column is dropped because it is not necessary as the machine learning algorithmn will use numeric values.

Each email subject began with "Subject: ". This was stripped from each email subject using Python string processing as it was both irrelevant and potentially biasing.


### Methods
Tools used throughout the entire process include  <br/>
* Numpy, Pandas, Matplotlib, Scikit-Learn and Seaborn fo data analysis and visualization
* Github was used for version control
* Spyder as the IDE
* Google Collab for python notebook 
<br/> <br/>
Decision Tree, Naive-Bayes, and Random Forest machine learning algorithmns were all used to model the classifier.<br /><br />
kNN was also initially included but proved to be inefficient and less accurate than the other models.

TFIDF Vectorizer was used to transform each word or "term" in the email subject into a weighted value reflecting its frequency within a document type (spam or ham). TFIDF Vectorizer assigns an inverse weighting to terms occurring frequently across documents, such as "the" and "and," so words weighted highly are more meaningful.


### Results
With default values for each model, Naive-Bayes showed the highest accuracy at 87% vs. 80% for Decision Trees.<br/><br/>
Tuning Decision Trees to have a max leaf depth of greater than or equal to 15 led to a higher accuracy on untrained test data, with an optimal accuracy of 0.94 at a max_depth = 15. Furthermore, higher max_depth values provided minimal or no additional improvement. <br/><br/>
Random Forest was then tested to determine whether it would provide a higher accuracy level vs. Decision Trees. Using default params, Random Forest had lower accuracy than Decision Trees (around 70%). Experimental tuning of parameters increased accuracy slightly, such as setting a min_samples_split to 10. However, expanding the max_depth parameter of Random Forest to 100 increased its accuracy to 97%, the highest of any of the three models.

![Results](https://github.com/m-arain/CST383Project/raw/main/Results.png 'Results')


### Discussion 
The results do imply that the three linear ML classifiers tested (Decision Tree, Naive-Bayes and Random Forest) are all viable approaches for a spam/ham classifier; which aligns with what researchers say according to ScienceDirect. The 97% accuracy on Random Forest also does reinforce the fact that Random Forest is capable of high classification accuracies and worked best for our goal of a high accuracy. Other potential factors to extend this project in the future could include grid search for decision trees, boosting, support vector machines(SVS) and Artificial Nueral networks (ANN).


### Summary
Based on the simulation results Decision Tree, Naive-Bayes and Random Forest have all proven to be capable machine learning algorithmns to base a spam/ham classifier off. In this case Random Forest was the most capable with 97% accuracy with minimal tuning. kNN Neighborsclassifier proved to be inefficient for our large dataset as the TFDIF Vectorizer produced too many columns for kNN to handle. The goal of building a spam/ham classifier with high accuracy was achieved.


### References/Bibliography

References/Bibliography: <br/>
https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html <br/>
https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html <br/>
https://pandas.pydata.org/ (multiple pages) <br/>
https://scikit-learn.org/ (multiple pages) <br/>
https://seaborn.pydata.org/ (multiple pages) <br/>
https://matplotlib.org/ (multiple pages) <br/>
https://www.kite.com/python/answers/how-to-plot-a-bar-chart-using-a-dictionary-in-matplotlib-in-python <br/>
https://python-graph-gallery.com/3-control-color-of-barplots/ <br/>
https://www.sciencedirect.com/science/article/pii/S2405844018353404 <br/>

Spam/Ham Dataset is obtained here: https://www.kaggle.com/venky73/spam-mails-dataset

Data was downloaded on Feb 7 2021
