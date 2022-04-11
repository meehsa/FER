#!/usr/bin/env python
# coding: utf-8

# In[13]:


from __future__ import print_function
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,cohen_kappa_score,classification_report,confusion_matrix
def classification_metrics(target,pred):
  acc=accuracy_score(target,pred)
  print("Accuracy",acc)
  f1=f1_score(target,pred,average='weighted')
  print("f1_score",f1)
  report=classification_report(target,pred)
  print(report)
  print(confusion_matrix(target,pred))
  lis=[target,pred]
  alpha=krippendorff_alpha(lis,nominal_metric,missing_items=None)
  print("Alpha:",alpha)
  kappa=cohen_kappa_score(target,pred)
  print("Kappa:",kappa)


# In[6]:


def regression_metrcis(target,pred):
  rmse=RMSE(target,pred)
  print("RMSE",rmse)
  ccc=CCC(target,pred)
  print("CCC",ccc)
  sagr=SAGR(target,pred)
  print("SAGR",sagr)
  pcc=PCC(target,pred)
  print("Corelation",pcc)


# In[10]:


from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
import numpy as np
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = []
  auc_pr_dict = []
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict.append(roc_auc)


    pre,recall,_ = precision_recall_curve(new_actual_class, new_pred_class)
    pr_auc=auc(recall,pre)
   
    auc_pr_dict.append(pr_auc)
  print("ROC:",np.mean(roc_auc_dict))
  print("AUC-PR:",np.mean(auc_pr_dict))



  return 

# assuming your already have a list of actual_class and predicted_class from the logistic regression classifier


# In[7]:


import numpy as np


def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.corrcoef(ground_truth, predictions)[0,1]


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)


# In[12]:



try:
    import numpy as np
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
    
    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.


# In[ ]:




