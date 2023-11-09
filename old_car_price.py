#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[3]:


# loading data set from sv to data frames
car_dataset = pd.read_csv('car data.csv')


# In[4]:


# displays the top rows of a DataFrame
car_dataset.head()


# In[5]:


# checking the number of rows and columns
car_dataset.shape


# In[6]:


# getting some information about the dataset
car_dataset.info()


# In[7]:


# checking the number of missing values
car_dataset.isnull().sum()


# In[8]:


# checking the distribution of categorical data
print(car_dataset.fuel.value_counts())
print(car_dataset.seller_type.value_counts())
print(car_dataset.transmission.value_counts())
print(car_dataset.owner.value_counts())


# In[9]:


# converting the text data into numerical data

# encoding "fuel" Column
car_dataset.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)

# encoding "seller_type" Column
car_dataset.replace({'seller_type':{'Dealer':0,'Individual':1,'Trustmark Dealer':2}},inplace=True)

# encoding "transmission" Column
car_dataset.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)

# encoding "owner" Column
car_dataset.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}},inplace=True)




# In[10]:


car_dataset.head()


# In[11]:


X = car_dataset.drop(['name','selling_price'],axis=1)
Y = car_dataset['selling_price']


# In[12]:


print(X)


# In[13]:


print(Y)


# In[14]:


# splitting the trainnig and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)


# In[15]:


# loading the linear regression model
lin_reg_model = LinearRegression()


# In[16]:


lin_reg_model.fit(X_train,Y_train)


# In[17]:


# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)


# In[18]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[19]:


# VISUALISES THE ACTUAL AND PRIDICTED PRICES
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[20]:


# prediction on Test data
test_data_prediction = lin_reg_model.predict(X_test)


# In[21]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[22]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[23]:


# loading the LASSO regression model
lass_reg_model = Lasso()


# In[24]:


lass_reg_model.fit(X_train,Y_train)


# In[25]:


# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)


# In[26]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[27]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[28]:


# prediction on Test data
test_data_prediction = lass_reg_model.predict(X_test)


# In[29]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[30]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[ ]:




