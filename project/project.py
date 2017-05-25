# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:48:19 2017

@author: cemalper
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
data = pd.read_csv('bank-additional-full.csv', sep = ';')
df_with_dummies = data
data["y"] = data["y"].astype('category')
df_with_dummies['y'] = data["y"].cat.codes
y = np.array(df_with_dummies['y'])
X = np.array(data.drop('y', axis=1))
# data exploration
#print(data.head())

le = LabelEncoder()
X = le.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#clf = Pipeline([('transform',LabelEncoder()),('tree',DecisionTreeClassifier())])

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


