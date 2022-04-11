#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torchvision
#from torchsummary import summary
import torch

class MultiHeadInceptionV3(nn.Module):
    def __init__(self,pretrained, requires_grad):
        super(MultiHeadInceptionV3, self).__init__()
    
        ############# inception v3
        self.model=torchvision.models.inception_v3(pretrained=True)
        self.model.aux_logits = False
        num_ftrs = self.model.fc.in_features
        self.model.fc=nn.Linear(num_ftrs, 16)
        self.l0 = nn.Linear(16, 8)
        self.l1 = nn.Linear(16, 1)
        self.l2 = nn.Linear(16, 1)
        ct=0
        for name, child in self.model.named_children():
          ct=ct+1
          if ct<19:
            for name2, params in child.named_parameters():
              params.requires_grad=False
          elif ct>=19:
            for name2, params in child.named_parameters():
              params.requires_grad=True

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model(x)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


# In[12]:


class MultiHeadResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet50, self).__init__()
        #For Resnet 50 
        if pretrained == True:
            self.model = pretrainedmodels.models.__dict__['resnet50'](pretrained='imagenet')  
        else:
            self.model = pretrainedmodels.models.__dict__['resnet50'](pretrained=None)
        ct=0    
        for child in self.model.children():
         # print(child)
          ct += 1
          if ct < 7:
              for param in child.parameters():
                  param.requires_grad = False
                  #print(param.requires_grad)
          elif ct>=7:
              for param in child.parameters():
                  param.requires_grad = True
                  #print(param.requires_grad)
        self.l0 = nn.Linear(2048, 8) # for gender
        self.l1 = nn.Linear(2048, 1) # for masterCategory
        self.l2 = nn.Linear(2048, 1) # for subCategory
     

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        #print(batch)
        x = self.model.features(x)
       
        

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        #print(x.shape)
   
      
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2

