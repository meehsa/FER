#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import numpy
import torch
from tqdm import tqdm
def train(model, dataloader, optimizer, loss_fn, dataset, device,weights):
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_running_corrects=0.0
    pred=[]
    target=[]
    for i, data in tqdm(enumerate(dataloader), total=np.ceil(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        exp = data['exp'].squeeze().to(device)
        val = data['val'].unsqueeze(1).to(device)
        aro = data['aro'].unsqueeze(1).to(device)
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        outputs = model(image)
        targets = (exp,val,aro)
        loss = loss_fn(outputs, targets,weights)
        train_running_loss += loss.item()

        o1_pred, o2, o3 = outputs
        o1_pred1=o1_pred.squeeze(1)
        _,o1=torch.max(o1_pred1.data,1)
        train_running_corrects += torch.sum( o1== exp)
        pred.append(o1)
        target.append(exp)
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_acc = train_running_corrects/len(dataset)
    return train_loss,train_acc,pred,target


# In[3]:


# validation function
def validate(model, dataloader, loss_fn, dataset, device,weights):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    # valid_loss_min=np.Inf
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        exp = data['exp'].squeeze().to(device)
        val = data['val'].unsqueeze(1).to(device)
        aro = data['aro'].unsqueeze(1).to(device)
       
        outputs = model(image)
        o1_pred, o2, o3 = outputs
        
        targets = (exp,val,aro)
        
        loss = loss_fn(outputs, targets,weights)
       # acc(outputs, targets)
        val_running_loss += loss.item()
        
        o1_pred1=o1_pred.squeeze(1)
        _,o1=torch.max(o1_pred1.data,1)
        val_running_corrects += torch.sum( o1== exp)
        
    val_loss = val_running_loss / counter
    val_acc = val_running_corrects/len(dataset)

    return val_loss,val_acc


# In[5]:


def test(model, dataloader, loss_fn, dataset, device,weights):
    model.eval()
    counter = 0
    target1, pred1 = [],[]
    target2,pred2 =[],[]
    target3,pred3=[],[]
    images=[]
    val_running_loss = 0.0
    val_running_corrects = 0.0
    valid_loss_min=np.Inf
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        exp = data['exp'].squeeze().to(device)
        val = data['val'].unsqueeze(1).to(device)
        aro = data['aro'].unsqueeze(1).to(device)
       
        outputs = model(image.detach())
        o1_pred, o2, o3 = outputs
        
        targets = (exp,val,aro)
        t1,t2,t3=targets
        
       # loss = loss_fn(outputs, targets,weights)

       # val_running_loss += loss.item()
        o1_pred1=o1_pred.squeeze(1)
        _,o1=torch.max(o1_pred1.data,1)
        val_running_corrects += torch.sum( o1== exp)

        for i in image:
          images.append(i.cpu().detach().numpy())

        for i in range(len(o1)):
          pred1.append(o1[i].cpu().numpy().tolist())
          target1.append(t1[i].cpu().numpy().tolist())
        for i in o2:
          for j in range(len(i)):
            pred2.append(i.cpu().detach().numpy()[j])
        for i in t2:
          for j in range(len(i)):
            target2.append(i.cpu().numpy()[j])
       

        for i in o3:
          for j in range(len(i)):
            pred3.append(i.cpu().detach().numpy()[j])
        for i in t3:
          for j in range(len(i)):
            target3.append(i.cpu().numpy()[j])
        
        
    #val_loss = val_running_loss / counter
   # ccc_loss =cc_oss/counter
    val_acc = val_running_corrects/len(dataset)
    
    return val_acc,images,pred1,pred2,pred3,target1,target2,target3 


# In[ ]:




