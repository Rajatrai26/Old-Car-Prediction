#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
 




# In[4]:


car_dataset = pd.read_csv('car data.csv')
print(car_dataset.head())
print("Shape of the dataset:", car_dataset.shape)
car_dataset.info()
print("Missing values in dataset:\n", car_dataset.isnull().sum())


# In[5]:


car_dataset.replace({'fuel': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}}, inplace=True)
car_dataset.replace({'seller_type': {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}}, inplace=True)
car_dataset.replace({'transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
car_dataset.replace({'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}}, inplace=True)

car_dataset['selling_price'] = np.log(car_dataset['selling_price'])

X = car_dataset.drop(['name', 'selling_price'], axis=1)
Y = car_dataset['selling_price']



# In[6]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)


# In[7]:


lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

cv_scores = cross_val_score(lin_reg_model, X, Y, cv=5, scoring='neg_mean_squared_error')
print("Linear Regression Cross-Validation MSE:", -cv_scores.mean())

lin_train_predictions = lin_reg_model.predict(X_train)
lin_train_error = metrics.r2_score(Y_train, lin_train_predictions)
print("Linear Regression R-squared Error (Training):", lin_train_error)

lin_test_predictions = lin_reg_model.predict(X_test)
lin_test_error = metrics.r2_score(Y_test, lin_test_predictions)
print("Linear Regression R-squared Error (Test):", lin_test_error)


# In[8]:


lasso_model = Lasso()
lasso_model.fit(X_train, Y_train)

lasso_cv_scores = cross_val_score(lasso_model, X, Y, cv=5, scoring='neg_mean_squared_error')
print("Lasso Regression Cross-Validation MSE:", -lasso_cv_scores.mean())

lasso_train_predictions = lasso_model.predict(X_train)
lasso_train_error = metrics.r2_score(Y_train, lasso_train_predictions)
print("Lasso Regression R-squared Error (Training):", lasso_train_error)

lasso_test_predictions = lasso_model.predict(X_test)
lasso_test_error = metrics.r2_score(Y_test, lasso_test_predictions)
print("Lasso Regression R-squared Error (Test):", lasso_test_error)


# In[9]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=2)
rf_model.fit(X_train, Y_train)

rf_cv_scores = cross_val_score(rf_model, X, Y, cv=5, scoring='neg_mean_squared_error')
print("Random Forest Cross-Validation MSE:", -rf_cv_scores.mean())

rf_train_predictions = rf_model.predict(X_train)
rf_train_error = metrics.r2_score(Y_train, rf_train_predictions)
print("Random Forest R-squared Error (Training):", rf_train_error)

rf_test_predictions = rf_model.predict(X_test)
rf_test_error = metrics.r2_score(Y_test, rf_test_predictions)
print("Random Forest R-squared Error (Test):", rf_test_error)


# In[10]:


car_data = [[2018, 40000, 0, 1, 0, 0]]  # Replace with actual car details

rf_predicted_log_price = rf_model.predict(car_data)
rf_predicted_price = np.exp(rf_predicted_log_price)
print(f"The predicted selling price of the car using Random Forest is: ₹{rf_predicted_price[0]:,.2f}")


# In[11]:


corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[ ]:




