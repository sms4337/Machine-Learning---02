"""
Created on Wed Nov 30 2022

@author: Syed Misba Shahriyaar
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt

# Step 1 - Load Data
dataset = pd.read_csv("iphone_purchase_dataset.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# Optional - if you want to convert X to float data type
# X = np.vstack(X[:, :]).astype(np.float)

# Step 3 - Split Data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4 - Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Step 5 - Logistic Regression Classifier
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)


# Step 6 - Predict
y_pred = classifier.predict(X_test)

fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (8,8), dpi=1000)
plt.scatter(dataset["Age"], dataset["Purchase Iphone"])
plt.xlabel("Age")
plt.ylabel("People Who bought Iphone")
fig.savefig('logistic_regression.png')

# Step 7 - Confusion Matrix
print('Confusion Matrix of Logistic Regression Algorithm: ')
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)

# Step 8 - Make New Predictions
x1 = sc.transform([[1,21,40000]])
x2 = sc.transform([[1,21,80000]])
x3 = sc.transform([[0,21,40000]])
x4 = sc.transform([[0,21,80000]])
x5 = sc.transform([[1,55,90000]])
x6 = sc.transform([[1,41,80000]])
x7 = sc.transform([[0,41,40000]])
x8 = sc.transform([[0,41,80000]])

print("Male aged 21 making $40k will buy iPhone:", classifier.predict(x1))
print("Male aged 21 making $80k will buy iPhone:", classifier.predict(x2))
print("Female aged 21 making $40k will buy iPhone:", classifier.predict(x3))
print("Female aged 21 making $80k will buy iPhone:", classifier.predict(x4))
print("Male aged 55 making $90k will buy iPhone:", classifier.predict(x5))
print("Male aged 41 making $80k will buy iPhone:", classifier.predict(x6))
print("Female aged 41 making $40k will buy iPhone:", classifier.predict(x7))
print("Female aged 41 making $80k will buy iPhone:", classifier.predict(x8))



