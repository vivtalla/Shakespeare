
# coding: utf-8

# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# In[42]:


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[43]:


df = pd.read_csv('Shakespeare_data.csv')


# Before I try and classify the data, I want to get a general feel for the data.

# In[44]:


df.head


# In[45]:


df["Player"].value_counts()


# Before we perform classification, I wanted to see all the different player possibilites we have from the data set

# In[46]:


df.info()


# The data provides these attributes. I will be using player as my target variable, and the others as the prediction variables.

# In[47]:


df.columns


# I need to replace NAN variales within my target set in order to get classification methods to work using scikit learn

# In[48]:


df['Player'].fillna(value = 'Other')


# In[49]:


df['Dataline'].fillna(value = 0)


# In[58]:


df['Player'] = df['Player'].astype(str)


# I had to make sure each row of Player is of type string. Previously some of them were not, so the function above converts the type of all player attributes to string

# To get any sort of classification method working, I will need numeric inputs to utilize scikit learn. One way to do this is using LabelEncoding to turn the string predictive attribute to numeric outputs.

# In[59]:


from sklearn import preprocessing


# In[60]:


le = preprocessing.LabelEncoder()


# In[61]:


le.fit(df['Player'])


# In[62]:


df['Player'] = le.transform(df['Player'])


# In[64]:


le.fit(df['Play'])
df['Play'] = le.transform(df['Play'])


# In[68]:


df['ActSceneLine'] = df['ActSceneLine'].astype(str)
le.fit(df['ActSceneLine'])
df['ActSceneLine'] = le.transform(df['ActSceneLine'])


# The code above was taken from scikitlearn's LabelEconder documentation. I encoded the required attributes as needed to provide numeric inputs and outputs

# In[78]:


df.head()


# The play, player, and actSceneLine have now been encoded into integers using label encoding

# In[79]:


df['PlayerLinenumber'].fillna(value = 0)


# Im changing the NAN values to 0 so no errors occur when using scikitlearn

# In[98]:


X = df[['Play','ActSceneLine']]
Y = df[['Player']]


# In[99]:


X = np.array(X)
Y = np.array(Y)


# X is the set of predictor variables, and Y is the outcome variable. This will be used while testing the different classification models. Additionally I will need to make a training and a test set given that I am solving a supervised learning problem.

# In[100]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 42)


# In[101]:


df.fillna(value = 0)


# In[102]:


df = df.reset_index()


# The first classification model I'm going to try to use is a decision tree. 

# In[103]:


clf = tree.DecisionTreeClassifier()


# In[104]:


clf = clf.fit(X_train, y_train)


# In[105]:


clf.predict(X_test)


# In[106]:


print(accuracy_score(y_test, clf.predict(X_test)))


# Using a decision tree, we got 51 percent accuracy. Not bad for my first time doing classification!
