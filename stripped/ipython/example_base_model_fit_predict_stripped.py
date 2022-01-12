#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '..')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np

from config import VISIBLE_GPUS

import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch

from sklearn.model_selection import train_test_split
from lib.main import BERTClassificationModel


# ## Load data - sample of IMDB reviews in english

# In[3]:


SAMPLE_DATA_PATH = 'test/sample_data/sample_data_eng.csv'


# In[4]:


# Loading data for tests
df = pd.read_csv(SAMPLE_DATA_PATH)

texts = df['sentence'].tolist() # list of texts
labels = df['target'].tolist() # list of 0/1 labels


# In[5]:


df


# ## Divide to train and test sets

# In[6]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


# # Fit and predict methods

# ## Fit the model

# In[7]:


# Loading model
model = BERTClassificationModel()
# Fitting a model to training data for 5 epochs
model.fit(X_train,y_train,epochs = 5)


# ## Get predictions

# In[8]:


# Predicted probability for test set
preds = model.predict(X_test)


# In[9]:


preds


# ## Calculate model accuracy on the test data

# In[10]:


predicted_classes = (np.array(preds) >= 0.5)
accurate = sum(predicted_classes == np.array(y_test).astype(bool))
accuracy = accurate/len(y_test)

print(f'Test accuracy: {accuracy}')

