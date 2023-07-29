#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<h1>Importing Libraries</h1>\n')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().run_cell_magic('HTML', '', '<h2>Importing Data and Checking out</h2>\n')


# In[4]:


HouseDF = pd.read_csv(r"F:\machine learning\USA_Housing.csv")


# In[5]:


HouseDF.head()


# In[6]:


HouseDF.info()


# In[7]:


HouseDF.describe()


# In[8]:


HouseDF.columns()


# In[9]:


HouseDF.columns


# In[10]:


get_ipython().run_cell_magic('HTML', '', '<h2>Exploratory Data Analysis for House Price Prediction</h2>\n')


# In[11]:


sns.pairplot(HouseDF)


# In[12]:


sns.distplot(HouseDF['Price'])


# In[13]:


sns.heatmap(HouseDF.corr(), annot=True)


# In[14]:


get_ipython().run_cell_magic('HTML', '', '<h2>Training a Linear regression Model</h2>\n<h3>X and Y list</h3>\n')


# In[15]:


X = HouseDF[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
Y = HouseDF['Price']


# In[16]:


get_ipython().run_cell_magic('HTML', '', '<h4>Split Data into Train, Test</h4>\n')


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=10)


# In[19]:


get_ipython().run_cell_magic('HTML', '', '<h2>Creating and training the Linear Regression Model</h2>\n')


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()


# In[22]:


lm.fir(X_train, Y_train)


# In[23]:


lm.fit(X_train, Y_train)


# In[24]:


get_ipython().run_cell_magic('HTML', '', '<h2>Linear Regression Model Evaluation</h2>\n')


# In[25]:


print(lm.intercept_)


# In[26]:


coeff_df = pd.DataFrame(lm.coef,X.columns,columns=['Coefficient'])
coeff_df


# In[27]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[28]:


get_ipython().run_cell_magic('HTML', '', '<h2>Prediction from our Linear Regression Model</h2>\n')


# In[29]:


predictions= lm.predict(X_test)


# In[30]:


plt.scatter(Y_test,predictions)


# In[31]:


get_ipython().run_cell_magic('HTML', '', '<h9>In the above scatter plot, we see data is in line shape, which means our model has done good predictions</h9>\n')


# In[32]:


sns.distplot((Y_test-predictions),bins=50)


# In[33]:


get_ipython().run_cell_magic('HTML', '', '<h2>Regression Evaluation Metrics</h2>\n')


# In[34]:


from sklearn import metrics


# In[35]:


print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


# In[36]:


get_ipython().run_cell_magic('HTML', '', '<h1>Thank You</h1>\n')


# In[ ]:




