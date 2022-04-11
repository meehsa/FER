#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from torch.utils.data import Dataset
import torch
import joblib
import math
import cv2
import numpy as np
import glob,pathlib
import torchvision.transforms as transforms
class CustomDataset(Dataset):
    def __init__(self, imgs_path,ant_path, img_size,is_train=True):
        self.imgs_path =imgs_path#"D:/ramesha/assignment2/val_set/images/"#"D:/ramesha/assignment2/train_and_val_set/train_set/images/"
        self.ant_path =ant_path#"D:/ramesha/assignment2/val_set/annotations/"#"D:/ramesha/assignment2/train_and_val_set/train_set/annotations/"
        file_list=glob.glob(self.imgs_path + "*")
        self.data=[]
        self.label=[]
        aro=[]
        val=[]
        landmark=[]
        exp=[]
        for class_path in file_list:
            ant = pathlib.Path(class_path).stem
            #print(ant)
            landmark.append(os.path.join(self.ant_path,ant+'lnd.npy'))
            exp.append(os.path.join(self.ant_path,ant+'exp.npy'))
            val.append(os.path.join(self.ant_path,ant+'val.npy'))
            aro.append(os.path.join(self.ant_path,ant+'aro.npy'))
            self.data.append([class_path,os.path.join(self.ant_path,ant+'_lnd.npy'),
                              os.path.join(self.ant_path,ant+'_exp.npy'),os.path.join(self.ant_path,ant+'_val.npy'),
                              os.path.join(self.ant_path,ant+'_aro.npy')])
            self.label.append(int(np.load(os.path.join(self.ant_path,ant+'_exp.npy')).tolist()))
        self.is_train = is_train
        # the training transforms and augmentations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
               # transforms.ColorJitter(brightness=(0,0.7),contrast=(0,0.8)),
               # transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # the validation transforms
        if not self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, landmark,exp,val,aro = self.data[idx]
        img = cv2.imread(img_path,1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        # image to float32 tensor
        #image = torch.tensor(image, dtype=torch.float32)
        # labels to long tensors
        exp=np.load(exp).tolist()
        val=np.load(val).tolist()
        aro=np.load(aro).tolist()
        exp=int(exp)
        val=float(val)
        aro=float(aro)
        img_exp = torch.tensor(exp, dtype=torch.long)
        img_val=torch.tensor(val, dtype=torch.float32)
        img_aro=torch.tensor(aro, dtype=torch.float32)
        return {
            'image': image,
            'exp': img_exp,
            'val': img_val,
            'aro':  img_aro
        }


# In[ ]:




