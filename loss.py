#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch
def loss_fn(outputs, targets,weights):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
   # l1 = nn.CrossEntropyLoss()(o1, t1)
    l1 = nn.CrossEntropyLoss(weight=weights,reduction='mean')(o1, t1)
    l2 = nn.MSELoss()(o2, t2)           # for val
    l3 = nn.MSELoss()(o3, t3)         # for aro
    return (l1 + l2 + l3) / 3


# In[ ]:




