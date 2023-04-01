#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Coursework 2
# 
# For coursework 2 you will be asked to train and evalute several different classifiers: Naïve Bayes classifier, Random Forest classifier, and kNN classifier using the iris dataset. You will be asked to answer a series of questions relating to each individual model and questions comparing each model. 
# 
# #### You are free to use the sklearn library. 
# 
# 
# Notes:
# - Remember to comment all of your code (see here for tips: https://stackabuse.com/commenting-python-code/). You can also make use of Jupyter Markdown, where appropriate, to improve the layout of your code and documentation.
# - Please add docstrings to all of your functions (so that users can get information on inputs/outputs and what each function does by typing SHIFT+TAB over the function name. For more detail on python docstrings, see here: https://numpydoc.readthedocs.io/en/latest/format.html)
# - When a question allows a free-form answer (e.g. what do you observe?), create a new markdown cell below and answer the question in the notebook. 
# - Always save your notebook when you are done (this is not automatic)!
# - Upload your completed notebook using the VLE
# 
# Plagiarism: please make sure that the material you submit has been created by you. Any sources you use for code should be properly referenced. Your code will be checked for plagiarism using appropriate software.
# 
# ### Marking 
# 
# The grades in this coursework are allocated approximately as follows:
# 
# |                                                    | mark  |  
# |----------------------------------------------------|-------|
# | Code                                               | 7     |
# | Code Report/comments                               | 6     |
# | Model questions                                    | 14    |  
# | Model comparision questions                        | 18    |
# | Total available                                    |**45** |  
# 
# ##### Remember to save your notebook as “CW2.ipynb”. It is a good idea to re-run the whole thing before saving and submitting. 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ## 1. Classifiers [7 marks total]
# Code and train your three classifiers in the cells below the corresponding header. You do not need to implement cross-validation in this coursework, simply fit the data. You are free to use sklearn and other packages where necessary.  
# 
# 

# In[1]:


# import datasets, classifiers, etc. 
from sklearn import datasets 

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random_state = 2023 # set a global random state number for all


# In[7]:


#load data, 50:50 split for train:test
'''
TO-DO: change the test_size parameter, which can greatly affect the performance of the models!!!!, 
0.75, 0.5, 0.25, ore even more, check how the models would be affected!!!! 

Take good look of the Summary Statistics!!!!!
'''
iris = datasets.load_iris()
#X, y = iris

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
print(iris.DESCR) # print dataset description


# In[52]:


# split the dataset into train/test sets
# set the test_size into [0.25, 0.5, 0.75], and see how the size of the training set affect each model. 
X, y = iris.data, iris.target

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.75, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y, test_size=0.25, random_state=random_state)


# ### 1.1 Naïve Bayes Classifier [2]
# Train a naïve bayes classifier in python. 
# 
# Use your code to fit the data given above. 
# 

# In[10]:


len(X_train), len(X_train_s), len(X_train_l)


# In[11]:


len(X_test), len(X_test_s), len(X_test_l)


# In[12]:


# Source : https://scikit-learn.org/stable/modules/naive_bayes.html
'''
Train a naïve bayes classifier
'''

clf_NB = MultinomialNB()
clf_NB.fit(X_train, y_train)
clf_NB.predict(X_test)


# In[13]:


y_pred = MultinomialNB().fit(X_train, y_train).predict(X_test)


# In[14]:


# Accuracy score at .6, how would increase/decrease the training dataset size affect the accuracy score? 
acc_NB = accuracy_score(y_test, y_pred)
print(acc_NB)


# In[16]:


# Plot confusion matrix to find out what is being confused!!! 
# compare the cm with other models
# compare the cm when the mode is run on X_train_s, X_train_l

cm = confusion_matrix(y_test, y_pred)

labels = iris.target_names
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8,4))
sns.heatmap(cm_df, annot=True)
plt.title('Multinomial Naïve Bayes')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[17]:


# get probability of each class for every X_test
# VERY IMPORTANT TO compare this probability with other models. 
# keep the array in proba_NB and compare it with proba_KNN, proba_RF

proba_NB = MultinomialNB().fit(X_train, y_train).predict_proba(X_test)


# ### 1.2 Random Forest Classifier [3]
# Train a random forest classifier in python. Use your code to fit the data given above. 
# 
# Evaluate feature performance of the model. 
# 
# Visualise the feature importance. 
# 

# In[26]:


#Train a random forest classifier in python

clf_RF = RandomForestClassifier(random_state=random_state)
clf_RF.fit(X_train, y_train)
clf_RF.predict(X_test)


# In[27]:


# Alternate code to above

y_pred = RandomForestClassifier().fit(X_train, y_train).predict(X_test)
y_pred


# In[28]:


'''
Very high accuracy_score
'''
acc_RF = accuracy_score(y_test, y_pred)
print(acc_RF)


# In[38]:


'''
The confused datapoints are different from BN!!!
Virginica has no confusion at all!
'''
cm_RF = confusion_matrix(y_test, y_pred)

RF = pd.DataFrame(cm_RF, index=labels, columns=labels)
plt.figure(figsize=(8,4))
sns.heatmap(RF, annot=True)
plt.title('Random Forest Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[31]:


# much higher confidence with its predictions compare with NB
proba_RF = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
proba_RF


# ### 1.3 kNN Classifier [2]
# Train a kNN classifier in python. 
# 
# Use your code to fit the data given above. 

# In[39]:


#Write your code here

clf_KNN = KNeighborsClassifier(n_neighbors=3)
clf_KNN.fit(X_train, y_train)


# In[40]:


clf_KNN.predict(X_test)


# In[41]:


y_pred = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train).predict(X_test)


# In[42]:


y_pred


# In[43]:


acc_KNN = accuracy_score(y_test, y_pred)
print(acc_KNN)


# In[44]:


# cm looks just like the cm of RF
cm_KNN = confusion_matrix(y_test, y_pred)

KNN = pd.DataFrame(cm_KNN, index=labels, columns=labels)

plt.figure(figsize=(8,4))
sns.heatmap(KNN, annot=True)
plt.title('KNeighborsClassifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[48]:


# the confidence is even higher than the RF model
# Explain!!!
proba_KNN = clf_KNN.predict_proba(X_test)
proba_KNN


# ## 2 Code Report [6 marks total]
# In a markdown box, write a short report (no more than 500 words) that describes the workings of your code. 

# In[13]:


#Write your answer here


# ## 3 Model Questions [14 marks total]
# Please answer the following questions relating to your classifiers. 

# ### 3.1 Naïves Bayes Questions [4]
# Why do zero probabilities in our Naïve Bayes model cause problems? 
# 
# How can we avoid the problem of zero probabilities in our Naïve Bayes model? 
# 
# Please answer in the cell below.

# **Why do zero probabilities in our Naïve Bayes model cause problems?**
# 
# In a Naive Bayes classifier, the probability of each feature belonging to a particular class is multiplied to calculate the likelihood of a given example belonging to that class. If one of these probabilities is zero, it means that the feature has never been seen in the training data for that class, which can cause problems when making predictions.
# 
# When we multiply a zero probability by any other value, the result will always be zero, which effectively eliminates the influence of that feature for that class. This can lead to incorrect predictions, as well as cause the denominator in Bayes' theorem to be zero, resulting in undefined probabilities and errors.
# 
# To resolve this issue, one common approach is to use a smoothing technique such as Laplace smoothing, which adds a small positive value to each count in the probability calculation to avoid getting a zero probability. This effectively replaces the maximum likelihood estimate of the probabilities with a smoothed estimate that takes into account the uncertainty in the data.
# 
# **How can we avoid the problem of zero probabilities in our Naïve Bayes model?**
# 
# One way to avoid the problem of zero probabilities in a Naive Bayes model is to use smoothing techniques. Smoothing can help to mitigate the issue of zero probabilities by adding a small positive value to the count of each feature in each class, which effectively replaces the maximum likelihood estimate of the probabilities with a smoothed estimate.
# 
# The most common smoothing technique used in Naive Bayes models is Laplace smoothing, which involves adding a small constant value (often 1) to each count in the probability calculation. This ensures that even if a feature has never been seen in the training data for a particular class, its probability is not zero, but rather a small positive value.
# 
# Another smoothing technique is called Bayesian smoothing, which can be more flexible and sophisticated than Laplace smoothing, but also more complex to implement.
# 
# In addition to smoothing, it may also be helpful to gather more training data to reduce the impact of rare events. Having more training examples of each class will reduce the impact of rare events and increase the accuracy of the model.
# 
# https://www.atoti.io/articles/how-to-solve-the-zero-frequency-problem-in-naive-bayes/
# 

# ### 3.2 Random Forest Questions [6]
# Which feature is the most important from your random forest classifier? 
# 
# Can any features be removed to increase accuracy of the model, if so which features? 
# 
# Explain why it would be useful to remove these features. 
# 
# Please answer in the cell below.

# In[15]:


#Write your answer here
'''
Which feature is the most important from your random forest classifier?
'''
petal length (cm) 0.441030464364
petal width (cm) 0.423357996355

They both have very high class correlation:
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        

'''
Can any features be removed to increase accuracy of the model, if so which features?
'''
from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
>>> rnd_clf.fit(iris["data"], iris["target"])
>>> for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
... print(name, score)
...
sepal length (cm) 0.112492250999
sepal width (cm) 0.0231192882825 !!!!!!!!
petal length (cm) 0.441030464364
petal width (cm) 0.423357996355


get_ipython().getoutput(' sepal width (cm) 0.0231192882825 -> least important')

'''
Explain why it would be useful to remove these features.
'''
if run iris.DESCR, 
part of the printout of iris.DESCR

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194 !!!!!!!!!! negative correlation 
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        
sepal width is not just least important, it has negative class correlation. 


# ### 3.3 kNN Questions [4]
# Do you think the kNN classifier is best suited to the iris dataset? 
# 
# What ideal qualities would the most appropriate dataset display?  
# 
# Please answer in the cell below.

# **Do you think the kNN classifier is best suited to the iris dataset?**
# 
# Hands-on 238-9
# This is where clustering algorithms step in: many of them can easily detect the
# top left cluster. It is also quite easy to see with our own eyes, but it is not so obvious
# that the lower right cluster is actually composed of two distinct sub-clusters. That
# said, the dataset actually has two additional features (sepal length and width), 
# not represented here, and clustering algorithms can make good use of all features, so in
# fact they identify the three clusters fairly well
# 
# 
# **What ideal qualities would the most appropriate dataset display?**
# 
# There are several factors that make a dataset most suitable for use in a KNN (k-nearest neighbors) classifier analysis:
# 
# - Dimensionality: KNN is a distance-based algorithm, so high-dimensional data can lead to the curse of dimensionality, where the distance between data points becomes less meaningful as the number of dimensions increases. A suitable dataset for KNN would have a low number of dimensions, ideally two or three.
# 
# - Scale: The scale of the features in the dataset should be consistent. Features with different scales can impact the distance calculation and affect the performance of the KNN algorithm.
# 
# - Class distribution: The class distribution should be roughly balanced, meaning that the number of examples for each class should be similar. If the class distribution is imbalanced, the KNN algorithm may be biased towards the majority class.
# 
# - Noise: The dataset should be free of noise and irrelevant features, as these can impact the performance of the KNN algorithm by affecting the distance calculation and leading to incorrect predictions.
# 
# - Separability: The classes in the dataset should be separable, meaning that they should be distinct and have clear boundaries. If the classes are highly overlapping, the KNN algorithm may have difficulty in determining the correct class for a given example.
# 
# By considering these factors, you can ensure that your dataset is suitable for use in a KNN classifier analysis and achieve the best possible results.
# 

# ## 4 Comparing Models [18 marks total]
# Please answer the following questions comparing your classifiers.

# ### 4.1 Compare each model [3]
# What differences do you see between your Naïve Bayes classifier, your random forest classifier, and your kNN classifier? 

# In[17]:


#Write your answer here
Multiclass classifiers
Random Forest classifiers or naive Bayes classifiers are capable of handling multiple classes


# ### 4.2 Accuracy [6]
# Can you explain why there are differences in accuracy between the three classifiers? 

# In[50]:


'''
Print accuracy scores of each model with the 50/50 split dataset
'''
for clf in (clf_NB, clf_RF, clf_KNN):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[53]:


'''
Print accuracy scores of each model with the 25% train / 75% test split dataset
NB accuracy scores is actually higher with less training data!!!
So NB is good model when data is very limited. 
'''
for clf in (clf_NB, clf_RF, clf_KNN):
    clf.fit(X_train_s, y_train_s)
    y_pred_s = clf.predict(X_test_s)
    print(clf.__class__.__name__, accuracy_score(y_test_s, y_pred_s))


# In[54]:


'''
Print accuracy scores of each model with the 75% train / 25% test split dataset
NB accuracy scores is much higher with 75% training data!!!
'''
for clf in (clf_NB, clf_RF, clf_KNN):
    clf.fit(X_train_l, y_train_l)
    y_pred_l = clf.predict(X_test_l)
    print(clf.__class__.__name__, accuracy_score(y_test_l, y_pred_l))


# ### 4.3 Appropriate Use [9]
# When would it be appropriate to use each different classifier? 
# 
# Reference real-world situations and examples of specific data sets and explain why that classifier would be most appropriate for that use-case.

# In[19]:


#Write your answer here
'''
Reference real-world situations and examples of specific data sets and 
explain why that classifier would be most appropriate for that use-case.
'''
* They require a small amount of training data to estimate the necessary parameters. source: https://scikit-learn.org/stable/modules/naive_bayes.html
In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. 
source: https://scikit-learn.org/stable/modules/naive_bayes.html
        
* good for dataset with highly indepedent features
all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, 
given the class variable. 
For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. 
A naive Bayes classifier considers each of these features to contribute independently to the probability that 
this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.
source : https://en.wikipedia.org/wiki/Naive_Bayes_classifier

Naive Bayes spam filtering -> https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
    
* naive Bayes model is well suited to data in the form of raw frequency counts

* live data stream, 
_______________________________________________________________________________________________________
Random Forest classifier

* The Random Forest algorithm introduces extra randomness when growing trees;
instead of searching for the very best feature when splitting a node (see Chapter 6), it
searches for the best feature among a random subset of features. This results in a
greater tree diversity, which (once again) trades a higher bias for a lower variance,
generally yielding an overall better model. (Hands-on p.199)

* easy to measure the relative importance of each feature
Random Forests are very handy to get a quick understanding of what features
actually matter, in particular if you need to perform feature selection.

*  curse of dimensionality, dimensionality reduction
_________________________________________________________________________________________________________

kNN classifier

* unsupervised learning

