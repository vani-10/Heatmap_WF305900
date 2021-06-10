#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel("WF305900_normalized_Intensities.xlsx")
df.head()


# In[2]:


import hdbscan


# In[3]:


clusterer = hdbscan.HDBSCAN()


# In[4]:


clusterer.fit(df)


# In[6]:


clusterer.labels_


# In[7]:


clusterer.labels_.max()


# In[8]:


clusterer.probabilities_


# In[9]:


clusterer = hdbscan.HDBSCAN(metric='euclidean')
clusterer.fit(df)
clusterer.labels_


# In[10]:


import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df.shape


# In[12]:


df


# In[17]:


data=df.values


# In[18]:


data


# # plt.scatter(*data.T,s=13, linewidth=0, c='b', alpha=0.25)
# What is s here?How to set s in my case.

# plt.scatter(*data.T, s=50, linewidth=0, c='b', alpha=0.25)

# In[22]:


clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(data)
color_palette = sns.color_palette('deep', 8)
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
plt.scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)


# In[23]:


clusterer.condensed_tree_


# In[24]:


clusterer.condensed_tree_.plot()


# In[25]:


clusterer.condensed_tree_.plot(select_clusters=True,
                               selection_palette=sns.color_palette('deep', 8))


# In[ ]:




