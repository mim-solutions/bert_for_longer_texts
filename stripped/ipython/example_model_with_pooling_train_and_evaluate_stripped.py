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
from lib.main import BERTClassificationModelWithPooling


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


# # Method train_and_evaluate

# In[7]:


# Loading model
model = BERTClassificationModelWithPooling()


# In[8]:


results = model.train_and_evaluate(X_train, X_test, y_train, y_test, epochs = 5)


# ## Get learning curve

# In[12]:


import matplotlib.pyplot as plt


# In[13]:


def plot_learning_curve(result):
    cmap = plt.get_cmap("tab10")
    fig,ax = plt.subplots(figsize = (10,10))

    for i, (key,value) in enumerate(result.items()):
        ax.plot(value, '-',label=key,color=cmap(i))
        ax.legend()


# In[14]:


plot_learning_curve(results)


# In[ ]:




