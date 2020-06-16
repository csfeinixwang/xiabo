
# coding: utf-8

# In[1]:

import pandas as pd

from io import StringIO

df=pd.read_csv("data.csv")


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid",context="notebook")
cols=['x1','x2','y']
sns.pairplot(df[cols],size=2.5)
plt.show()


# In[3]:

import numpy as np
cm=np.corrcoef(df[cols].values.T)


# In[4]:

sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)


# In[5]:

plt.show()


# In[6]:

X,y=df.iloc[:,0:2].values,df.iloc[:,2].values


# In[7]:

print(X)


# In[8]:

print(y)


# In[9]:

from sklearn.ensemble import RandomForestRegressor


# In[10]:

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


# In[24]:

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)


# In[25]:

regr.fit(X_train,y_train)
y_train_pred=regr.predict(X_train)
y_test_pred=regr.predict(X_test)


# In[27]:

from sklearn.metrics import mean_squared_error,r2_score

print('MSE train: %.3f, test: %.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
print('R^2 train: %.3f, test: %.3f'%(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))


# In[28]:

plt.scatter(y_train_pred,y_train_pred- y_train, c='black',marker='o',s=35,alpha=0.5,label='Training data')
plt.scatter(y_test_pred,y_test_pred- y_test, c='lightgreen',marker='s',s=35,alpha=0.7,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()


# In[43]:

print(regr.get_params(deep=True))


# In[42]:





# In[ ]:



