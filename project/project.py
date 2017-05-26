# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:48:19 2017

@author: cemalper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,classification_report
plt.style.use('ggplot')

data = pd.read_csv('bank-additional-full.csv', sep = ';')
data["y"] = data["y"].astype('category')
data['y'] = data["y"].cat.codes.astype('int8')
df_with_dummies = pd.get_dummies(data) # get the data ready for sklearn
X = df_with_dummies.drop('y', axis=1)
y = df_with_dummies['y']

#to get whether the classes are balanced or not!
print(y.value_counts())
# very high imbalance between classes

#le = LabelEncoder()
#enc = OneHotEncoder()
#
#y = np.array(df_with_dummies['y'])
#X = np.array(data.drop('y', axis=1))
#
## data exploration
##print(data.head())
#
#df_by_type = data.columns.to_series().groupby(data.dtypes).groups
#hed = data.select_dtypes(include=['object'])
#for col in hed: # for each categorical column
#    print(hed[col].unique())
#    #print(col)
#    hed[col] = hed[col].astype('category').cat.codes # convert to integer codes 
#hed = np.array(hed) # convert integers to one-hot columns
#hed = enc.fit_transform(hed).todense()
## data processing
##for x in df_by_type[np.dtype('object')]:
##    #print(type(x))
##    print(x)
##le = LabelEncoder()
###X = le.fit_transform(X)
## CODE BEGIN
#norm = StandardScaler()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
####clf = Pipeline([('transform',LabelEncoder()),('tree',DecisionTreeClassifier())])
##
#clf_list = [DecisionTreeClassifier(class_weight='balanced'), 
#            RandomForestClassifier(class_weight='balanced',n_estimators=40,bootstrap= False), 
#            GaussianNB(),
#            Pipeline([('scaler', MinMaxScaler()), ('svc', SVC(class_weight='balanced'))])]
#[x.fit(X_train,y_train) for x in clf_list]
#importances_Tree = clf_list[0].feature_importances_ 
#importances_RF = clf_list[1].feature_importances_
##components_PCA = clf_list[-1].explained_variance_
#acc_scores = [x.score(X_test,y_test) for x in clf_list]
#reports = [classification_report(x.predict(X_test),y_test) for x in clf_list]
#fig, ax = plt.subplots()
##fig.canvas.draw()
#col_labels = [col for col in X]
##order = sorted(range(len(col_labels)), key=lambda k: importances_Tree[k])[::-1]
##col_labels = [col_labels[x] for x in order]
##importances_Tree = [importances_Tree[x] for x in order]
## CODE END

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters, cv = 5)
clf.fit(X, y)
print(sorted(clf.cv_results_.keys()))




plt.xticks(np.arange(0, len(importances_Tree), 1.0))
plt.plot(range(len(importances_Tree)),importances_Tree)
plt.plot(range(len(importances_Tree)),importances_RF)
plt.legend(['Decision Tree','Random Forest'])
ax.set_xticklabels(col_labels)
fig= plt.figure()
ax = fig.add_subplot(121)
plt.scatter(X[col_labels[0]],X[col_labels[1]], c = y,alpha=0.5)
plt.xlabel(col_labels[0])
plt.ylabel(col_labels[1])
fig.add_subplot(122)
plt.scatter(X[col_labels[0]],X[col_labels[3]], c = y,alpha=0.5)
plt.xlabel(col_labels[0])
plt.ylabel(col_labels[3])
print(acc_scores)
#
#
