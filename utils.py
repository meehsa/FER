#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.style.use('ggplot')

def save_loss_plot(train_loss, val_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')
    plt.show()
def save_model(epochs, model, optimizer, criterion):
  torch.save({
              'epoch': epochs,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, 'model.pth')

def ShowImage(train,tr_lbl,tr_pred,w,c):
    
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 7))
    fig.subplots_adjust(wspace=4,hspace=4)
    fig.tight_layout()

    # flatten the axis into a 1-d array to make it easier to access each axes
    axs = axs.flatten()
    r=[]
    # iterate through and enumerate the files, use i to index the axes
    for i in range(10):
        cnt=randint(0,250)
        r.append(cnt)
    for i in range(10):
        val=r[i]
        if w==0:
          axs[i].imshow(train[val].detach().permute(1, 2, 0))
        if w==1:
          img = train[val].swapaxes(0,1)
          img = img.swapaxes(1,2)
          axs[i].imshow(img)

        # add an axes title; 
        if w==0:
            axs[i].set(title='Class:'+str(tr_lbl[val]))
        elif w==1:
            axs[i].set(title=f"Pred: {tr_pred[val]}. Truth: {tr_lbl[val]}")
       
        

    #add a figure title
    if c==0:
          fig.suptitle('Training Dataset', fontsize=16)
          plt.savefig('Train_data.png')
    elif c==1:
        fig.suptitle('Classified_Images', fontsize=16)
        plt.savefig('Classified_Images.png')
    elif c==2:
      fig.suptitle('Miss Classified_Images', fontsize=16)
      plt.savefig('Miss Classified_Images.png')


from random import randint
def show_batch(dl):
    for data in dl:
     # print(cls)
      ShowImage(data['image'],data['exp'].numpy(),data['exp'].numpy(),0,0)
      break


# In[2]:


def misclassfied_images(test,tst_lbl,pred):
    mis_img=[]
    pred_lbl=[]
    act_lbl=[]
    for i in range(len(test)):
        if(pred[i] != tst_lbl[i]):
                # If these labels are not equal, the image has been misclassified
                mis_img.append((test[i]))
                pred_lbl.append(pred[i])
                act_lbl.append(tst_lbl[i])
    return mis_img,pred_lbl,act_lbl
  


# In[3]:


def classfied_images(test,tst_lbl,pred):
    img=[]
    pred_lbl=[]
    act_lbl=[]
    for i in range(len(test)):
        if(pred[i] == tst_lbl[i]):
                # If these labels are not equal, the image has been misclassified
                img.append((test[i]))
                pred_lbl.append(pred[i])
                act_lbl.append(tst_lbl[i])
    return img,pred_lbl,act_lbl


# In[ ]:




