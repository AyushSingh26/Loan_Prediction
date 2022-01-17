#!/usr/bin/env python
# coding: utf-8

# # Importint dependencies
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Data collection and processing
# 

# In[2]:


#loading dataset to pandas
loan_dataset = pd.read_csv('dataset.csv')


# In[3]:


type(loan_dataset)


# In[4]:


#printing the first 5 rows of the data frame
loan_dataset.head()


# In[5]:


#number of rows and coloumn
loan_dataset.shape


# In[6]:


#statistical measures
loan_dataset.describe()


# In[8]:


#number of missing values in each coloumn
loan_dataset.isnull().sum()


# In[9]:


#dropping all the missing values
loan_dataset = loan_dataset.dropna()


# In[10]:


#number of missing values in each coloumn
loan_dataset.isnull().sum()


# In[11]:


#label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[12]:


loan_dataset.head()


# In[14]:


# dependents column values
loan_dataset['Dependents'].value_counts()


# In[15]:


#replacing value of 3+ to 4 in Dependents
loan_dataset.replace({"Dependents":{'3+':4}},inplace=True)
#loan_dataset = loan_dataset.replace(to_replace='3+', value=4)


# In[16]:


# dependents column values
loan_dataset['Dependents'].value_counts()


# # DATA VISUALIZATION
# 

# In[17]:


# educationn & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


# In[18]:


# marital stats & loan status
sns.countplot(x='Married',hue='Loan_Status',data=loa 


# In[20]:


#convert categorical columns to numerical values
loan_dataset.replace({"Married":{'Yes':1, 'No': 0},'Gender': {"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1},
                      "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},"Education":{"Graduate":1,"Not Graduate":0}},
                      inplace=True) 


# In[21]:


loan_dataset.head()


# In[22]:


# separating the data and label
X= loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y=loan_dataset['Loan_Status']


# In[23]:


print(X)
print(Y)


# In[24]:


#Train Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[25]:


print(X.shape,X_train.shape,X_test.shape)


# In[26]:


#Training the model using support vector model
classifier=svm.SVC(kernel='linear')


# In[28]:


#training the support vector machine model
classifier.fit(X_train,Y_train)


# In[29]:


#Model Evaluation
X_tp= classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_tp,Y_train)


# In[30]:


print(training_data_accuracy)


# In[31]:


#Model Evaluation
X_testp= classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_testp,Y_test)


# In[32]:


print(test_data_accuracy)


# In[33]:


#making a predictive system


# In[ ]:




