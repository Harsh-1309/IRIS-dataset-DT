#Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
filepath = r"D:\IEEE PROJECT\iris DATASET\Iris.csv"
df = pd.read_csv(filepath) 
df.isnull().any()

#Loading the iris data
data = load_iris()
print('Classes to predict: ', data.target_names)

#Extracting data attributes
X = data.data
### Extracting target/ class labels
y = data.target
print('Number of examples in the data:', X.shape[0])

#First four rows in the variable 'X'
X[:4]
sns.pairplot(df, hue='Species')

#Using the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

#Importing the Decision tree classifier from the sklearn library.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy')

#Training the decision tree classifier. 
clf.fit(X_train, y_train)

#Predicting labels on the test set.
y_pred =  clf.predict(X_test)

#Importing the accuracy metric from sklearn.metrics library
from sklearn.metrics import accuracy_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))

#Tuning the module for increased accuracy
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
clf.fit(X_train, y_train)
print('Accuracy Score on train data after tuning: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data after tuning: ', accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
