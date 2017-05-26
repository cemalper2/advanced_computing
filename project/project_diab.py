# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:02:46 2017

@author: alper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#plt.rcParams["figure.figsize"] = (8, 8)
#plt.rcParams["font.size"] = 14
data = pd.read_csv('diabetic_data.csv')
#get feature names
print(data.keys());

#our target is the 'readmitted' field, which is a categorical variable 
# Possible values: 'NO' no readmission, '>30' patient readmitted in more than 30 days,
# '<30' patient readmitted in less than 30 days
print(data['readmitted'].unique())
hede = data['readmitted'].value_counts()
plt.bar(range(len(hede.values)),[hede[x] for x in hede.keys()])
plt.xticks(range(len(hede.values)), hede.keys())
plt.xlabel('Readmitted')
plt.ylabel('# of Occurences')
#it is evident that we have an 'imbalanced' data set, therefore we should be careful in choosing our accuracy metric.
#Using the simple accuracy is not adequate!!


#for (ind,x) in enumerate(data.groupby('age')):
#    print(''.join([str(x[0]),':']))
#    plt.bar(range(3)+0.5*ind,)
    
colors = ['red','green','blue','purple']

fig = plt.figure(figsize=(8,8))
scat_plt = plt.scatter(data['time_in_hospital'], data['number_inpatient'], c=pd.get_dummies(data['readmitted']))
#cbar = plt.colorbar(scat_plt)
plt.xlabel('Time in Hospital')
plt.ylabel('# Inpatient Visits')



fig = plt.figure(figsize=(8,8))
scat_plt = plt.scatter(data['time_in_hospital'], data['number_outpatient'], c=pd.get_dummies(data['readmitted']))
#cbar = plt.colorbar(scat_plt)
plt.xlabel('Time in Hospital')
plt.ylabel('# Outpatient Visits')


fig = plt.figure(figsize=(8,8))
scat_plt = plt.scatter(data['number_inpatient'], data['number_outpatient'], c=pd.get_dummies(data['readmitted']))
#cbar = plt.colorbar(scat_plt)
plt.xlabel('# Inpatient Visits')
plt.ylabel('# Outpatient Visits')
