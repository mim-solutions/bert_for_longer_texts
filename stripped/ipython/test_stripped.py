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
from lib.roberta_main import RobertaClassificationModel

SAMPLE_DATA_PATH = 'test/sample_data/sample_data.csv'

# Loading data for tests
df = pd.read_csv(SAMPLE_DATA_PATH)

texts = df['sentence'].tolist() # list of texts
labels = df['target'].tolist() # list of 0/1 labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Loading model
model = RobertaClassificationModel()
# Fitting a model to training data for 5 epochs
result = model.train_and_evaluate(X_train, X_test, y_train, y_test,epochs = 5)


# In[5]:


df_result = pd.DataFrame(result)


# In[12]:


df_result.to_markdown()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




