#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import warnings
warnings.simplefilter("ignore")


# In[10]:


df=pd.read_csv("C:\\Users\\srmve\\Documents\\iris.csv")
df.head()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.isnull().sum()


# In[14]:


x=df.iloc[:,:-1]
x






# In[15]:


y=df.iloc[:,-1]
y


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)



# In[17]:


x_train.shape



# In[18]:


x_test.shape





# In[19]:


y_train.shape


# In[20]:


y_test.shape


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) 


# In[22]:


model.fit(x_train,y_train)








# In[23]:


y_pred=model.predict(x_test)
y_pred


# In[24]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[25]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[26]:


from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)
print(f"\nClassification Report:\n{class_report}")


# In[27]:


new_flower = [[5.1, 3.5, 1.4, 0.2]]  
predicted_class = model.predict(new_flower)
predicted_class


# In[ ]:




