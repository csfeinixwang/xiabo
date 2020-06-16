# -*- coding: utf-8 -*-
"""
Created on Sat May  5 19:30:28 2018

@author: Administrator
"""

import pandas as pd

from io import StringIO

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor




df=pd.read_csv("data.csv")



regr = RandomForestRegressor(max_depth=2, random_state=0)


# In[11]:

regr.fit(X, y)


# In[12]:

print(regr.feature_importances_)


# In[13]:

print(regr.predict([[0.56645,0.87683
]]))


# In[16]:

print(regr.predict([[0.54857,0.9172
]]))


# In[19]:

print(regr.predict([[0.55039,0.90304


]]))
