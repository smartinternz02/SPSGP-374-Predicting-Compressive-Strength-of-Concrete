#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from collections import Counter as c
import pickle


# In[2]:


data = pd.read_csv('concrete.csv')


# In[3]:


req_col_names = ["cement", "slag", "flyAsh", "water", "superplasticizer",
                 "coarseAggregate", "fineAggregare", "age", "csMPa"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().any()


# In[9]:


data.isnull().sum()


# In[10]:


data.boxplot(figsize=(18,7))


# In[11]:


sns.pairplot(data)
plt.show()


# In[12]:


corr = data.corr()

sns.heatmap(corr, annot=True, cmap='Blues')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[13]:


ax = sns.distplot(data.csMPa)
ax.set_title("Compressive Strength Distribution")


# In[14]:


x = pd.DataFrame(data,columns = data.columns[:8])
y = pd.DataFrame(data,columns = data.columns[8:])


# In[15]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 1)


# In[16]:


from sklearn.linear_model import Lasso, Ridge


# In[17]:


lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
print(y_pred_lr)


# In[18]:


lasso = Lasso()
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)
print(y_pred_lasso)


# In[19]:


ridge = Ridge()
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)
print(y_pred_ridge)


# In[20]:


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train,y_train)


# In[21]:


y_pred = gbr.predict(x_test)
print(" Prediction made by Gradient Boosting model:",y_pred)


# In[22]:


score = gbr.score(x_test,y_test)
print("Score of Gradient Boosting Model:",score)


# In[23]:


print(" MAE:",mean_absolute_error(y_test,y_pred))
print(" MSE:",mean_squared_error(y_test,y_pred))
print(" RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[24]:


pickle.dump(gbr,open('cement.pkl','wb'))


# In[ ]:




