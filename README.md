# Breast-cancer-detection
Breast cancer detection using 4 different models i.e. Logistic Regression, KNN, SVM and Decision Tree Machine Learning models and optimising them for even a better accuracy. This project is started with the goal use machine learning algorithms and learn how to optimize the tuning params and also and hopefully to help some diagnoses.




### Libraries used
```python
import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
%matplotlib inline 
import matplotlib.pyplot as plt #for plotting the graphs
from scipy import stats #for statistical info
from time import time

from sklearn import tree
from sklearn.model_selection import train_test_split # to split the data in train and test
from sklearn.model_selection import KFold # for cross validation
from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn import metrics  # for checking the accuracy 

#Classifiers 

from sklearn import svm #for Support Vector Machines
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix #for Logistic regression
from sklearn.svm import SVC # for support vector classifier
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier # for K neighbor classifier
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier

```

Name - Guttikonda Yoganand 

Batch - DW52DW53


