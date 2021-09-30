#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import selenium
import pandas as pd
import time
from bs4 import BeautifulSoup
import csv

# Importing selenium webdriver 
from selenium import webdriver

# Importing required Exceptions which needs to handled
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

#Importing requests
import requests

# importing regex
import re


# In[2]:


driver=webdriver.Chrome("chromedriver.exe") 
time.sleep(4)

url = "https://www.cardekho.com"
driver.get(url)
time.sleep(6)


# In[3]:


search_cars = driver.find_element_by_id('cardekhosearchtext')
search_cars


# In[4]:


# write on search bar
search_cars.send_keys("Used Cars")
time.sleep(2)

# do click using class_name function
search_btn = driver.find_element_by_class_name('searchbtn')
search_btn.click()
time.sleep(6)


# In[5]:


# so lets extract all the tags having the cars
car_tags=driver.find_elements_by_xpath("//div[@class='gsc_col-xs-7 carsName']")
car_tags


# In[6]:


# Now the text of the car is inside the tags extracted above

# so we will run a loop to iterate over the tags extracted above and extract the text inside them.
car_titles=[]
for i in car_tags:

    car_titles.append(i.text)
car_titles


# In[22]:


price_tags=driver.find_elements_by_xpath("//span[@class='amnt ']")
price_tags


# In[20]:


# so we will run a loop to iterate over the tags extracted above and extract the text inside them.
car_price=[]
for i in price_tags:
    
    car_price.append(i.text)
car_price


# In[7]:


driver=webdriver.Chrome("chromedriver.exe") 
time.sleep(4)
url = "https://www.cars24.com/"
driver.get(url)
time.sleep(6)


# In[8]:


# scrapping car brands
car_tags=driver.find_elements_by_xpath("//div[@class='_1l4fi']")
car_tags


# In[9]:


# Now the text of the car is inside the tags extracted above

# so we will run a loop to iterate over the tags extracted above and extract the text inside them.
car_titles=[]
for i in car_tags:

    car_titles.append(i.text)
car_titles


# In[24]:


price_tags=driver.find_elements_by_xpath("//div[@class='_7udZZ']")
price_tags


# In[25]:


# so we will run a loop to iterate over the tags extracted above and extract the text inside them.
car_price=[]
for i in price_tags:
    
    car_price.append(i.text)
car_price


# In[10]:


driver=webdriver.Chrome("chromedriver.exe") 
time.sleep(4)
url = "https://www.olx.in/cars_c84/q-old-car"
driver.get(url)
time.sleep(6)


# In[11]:


# scrapping car brands
car_tags=driver.find_elements_by_xpath("//div[@class='IKo3_']")
car_tags


# In[12]:


# so we will run a loop to iterate over the tags extracted above and extract the text inside them.
car_brands=[]
for i in car_tags:
    
    car_brands.append(i.text)
car_brands


# In[26]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt


# In[56]:


#uploading csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\CAR Project.csv")
orgdata = data


# In[57]:


# understanding the data
data.head()


# In[58]:


data.shape


# In[59]:


data.tail()


# In[60]:


data.columns


# In[61]:


data.info()


# In[62]:


# summing up the nissing values (column wise)
data.isnull()


# In[63]:


data.isnull().sum(axis=0).sort_values(ascending=False)


# In[64]:


data_num = data.select_dtypes(include = ['float64', 'int64', 'object'])
data_num.head()


# In[65]:


data_num.hist(figsize=(18, 22), bins=55, xlabelsize=10, ylabelsize=10); 


# In[99]:


data.drop(['Brand', 'Model', 'Location'], axis = 1)


# In[162]:


#uploading csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\CAR01.csv")
orgdata = data


# In[163]:


# understanding the data
data.head()


# In[164]:


data.columns


# In[165]:


# importing required library
import seaborn as sns
import os
import csv
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
plt.style.use('bmh')


# In[166]:


Price= pd.pivot_table(data,index = 'Mfg_Year', values='Price')


# In[167]:


Price


# In[168]:


Price.plot(kind='bar')


# In[169]:


Price= pd.pivot_table(data,index = 'Kms', values='Price')


# In[170]:


Price


# In[171]:


Price.plot(kind='bar')


# In[172]:


Price= pd.pivot_table(data,index = 'Fuel_Type', values='Price')


# In[173]:


Price


# In[174]:


Price.plot(kind='bar')


# In[175]:


Price= pd.pivot_table(data,index = 'Transmission', values='Price')


# In[176]:


Price


# In[177]:


Price.plot(kind='bar')


# In[178]:


corelation = data.corr() 


# In[179]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[180]:


sns.boxplot


# In[181]:


sns.pairplot


# In[182]:


y = np.array(data['Price'])
y.shape


# In[183]:


x = np.array(data.loc[:, 'Mfg_Year' : 'Transmission'])
x.shape


# In[184]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,)


# In[185]:


x_train.shape


# In[186]:


x_test.shape


# In[187]:


y_train.shape


# In[188]:


y_test.shape


# In[189]:


from sklearn.model_selection import KFold
folds = (KFold(n_splits = 10, shuffle = True, random_state = 100))


# In[190]:


hyper_params = [{'n_features_to_select':list(range(1,4))}]


# In[191]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[192]:


from sklearn.feature_selection import RFE
rfe = RFE(lm)
from sklearn.model_selection import GridSearchCV
modelcv = GridSearchCV(estimator = rfe,
                      param_grid = hyper_params,
                      scoring = 'r2',
                      cv = folds,
                      verbose = 1,
                      return_train_score = True)
modelcv.fit(x_train, y_train)


# In[193]:


cvresults = pd.DataFrame(modelcv.cv_results_)
cvresults


# In[194]:


data.shape


# In[195]:


print(np.mean(cvresults))


# In[196]:


plt.figure(figsize = (20,17))


# In[197]:


plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_test_score'])
plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_train_score'])
plt.xlabel('Number of features')
plt.ylabel('Optimal number of features')


# In[198]:


n_features_optimal = 6


# In[199]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[200]:


rfe = RFE(lm, n_features_to_select = n_features_optimal)


# In[201]:


rfe.fit(x_train, y_train)


# In[202]:


y_pred = lm.predict(x_test)
y_pred


# In[203]:


r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# In[ ]:




