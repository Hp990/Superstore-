#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.computation.check import NUMEXPR_INSTALLED
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('C:/Users/DELL/Downloads/SampleSuperstore.csv')  #loading dataset
df.head()    #display top 5 rows


# In[5]:


df.tail()   


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[11]:


df.columns


# In[12]:


df.duplicated().sum()


# In[13]:


df.nunique()


# In[14]:


df['Postal Code'] = df['Postal Code'].astype('object')


# In[15]:


df.drop_duplicates(subset=None,keep='first',inplace=True)
df.duplicated().sum()


# In[18]:


df = df.drop(['Postal Code'],axis = 1)


# In[19]:


sns.pairplot(df, hue = 'Ship Mode')


# In[20]:


df['Ship Mode'].value_counts()


# In[21]:


sns.countplot(x=df['Ship Mode'])


# In[22]:


df['Segment'].value_counts()   


# In[23]:


sns.pairplot(df,hue = 'Segment')  


# In[24]:


sns.countplot(x = 'Segment',data = df, palette = 'rainbow')


# In[25]:


df['Category'].value_counts()


# In[26]:


sns.countplot(x='Category',data=df,palette='tab10')


# In[27]:


sns.pairplot(df,hue='Category')


# In[28]:


df['Sub-Category'].value_counts()


# In[29]:


plt.figure(figsize=(15,12))
df['Sub-Category'].value_counts().plot.pie(autopct='dark')
plt.show()


# In[30]:


df['State'].value_counts()


# In[31]:


plt.figure(figsize=(15,12))
sns.countplot(x='State',data=df,palette='rocket_r',order=df['State'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# In[33]:


plt.figure(figsize=(10,8))
df['Region'].value_counts().plot.pie(autopct = '%1.1f%%')
plt.show()


# In[34]:


fig,ax=plt.subplots(figsize=(20,8))
ax.scatter(df['Sales'],df['Profit'])
ax.set_xlabel('Sales')
ax.set_ylabel('Profit')
plt.show()


# In[35]:


sns.lineplot(x='Discount',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# In[36]:


sns.lineplot(x='Quantity',y='Profit',label='Profit',data=df)
plt.legend()
plt.show()


# In[37]:


df.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['pink','blue'],figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# In[38]:


plt.figure(figsize=(12,8))
plt.title('Segment wise Sales in each Region')
sns.barplot(x='Region',y='Sales',data=df,hue='Segment',order=df['Region'].value_counts().index,palette='rocket')
plt.xlabel('Region',fontsize=15)
plt.show()


# In[39]:


df.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['blue','red'],figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# In[40]:


ps = df.groupby('State')[['Sales','Profit']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['blue','orange'],figsize=(15,8))
plt.title('Profit/loss & Sales across states')
plt.xlabel('States')
plt.ylabel('Profit/loss & Sales')
plt.show()


# In[41]:


t_states = df['State'].value_counts().nlargest(10)
t_states


# In[42]:


df.groupby('Category')[['Profit','Sales']].sum().plot.bar(color=['yellow','purple'],alpha=0.9,figsize=(8,5))
plt.ylabel('Profit/Loss and sales')
plt.show()


# In[43]:


ps = df.groupby('Sub-Category')[['Sales','Profit']].sum().sort_values(by='Sales',ascending=False)
ps[:].plot.bar(color=['red','lightblue'],figsize=(15,8))
plt.title('Profit/loss & Sales across states')
plt.xlabel('Sub-Category')
plt.ylabel('Profit/loss & Sales')
plt.show()


# In[ ]:




