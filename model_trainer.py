#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import relevant stuff
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from numpy import random
import time as tm
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

#1. Generate all possible test-train splits for leaving 2 out
# and randomly select a subset of these to run experiment on

n_splits = 50 #test-train splits
vols = [0.3,0.315,0.33,0.345,0.36,0.375,0.39,0.405,0.42,0.435,0.45,0.465,0.48,0.495,0.51,0.525,0.54,0.555,0.57,0.585,0.6]
wids = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
vw_test = np.zeros((176400,4)) #cols 0-1: v, cols 2-3: w to hold out for testing. 21P2*21P2 = 176400

i = 0
for a in range (0,len(vols)): #0th v
    for b in range (0,len(vols)): #1st v (must not be a repeat of v0, as we need 2 unique values of v to hold out for testing)
        for c in range (0,len(wids)): #0th w
            for d in range (0,len(wids)): #1st w (must not be a repeat of w0)
                if (vols[a]!=vols[b]) and (wids[c]!=wids[d]):
                    vw_test[i,0] = vols[a]
                    vw_test[i,1] = vols[b]
                    vw_test[i,2] = wids[c]
                    vw_test[i,3] = wids[d]
                    i=i+1
                    
#Also generate x and y points in the unit square for prediction purposes later 
n = 70
k = n #discretization resolution: Increasing the int coeff dramatically increases model training and testing time!!!
step = 1.0/float(k)
x_values = np.zeros((k+1)) #horizontal dir (x(0))
y_values = np.zeros((k+1)) #vertical dir (x(1))
digits = 6
for i in range (1,k+1): #rounding required to avert issues later 
    x_values[i] = round(x_values[i-1] + step,digits)
    y_values[i] = round(y_values[i-1] + step,digits)
if x_values[k]>1.0: #fixes additional rounding issue
    x_values[k] = 1.0
if y_values[k]>1.0:
    y_values[k] = 1.0
    
Now randomly select subset to run test-train splits on      #maintain vw_test_sel from prior experiment
vw_test_sel = np.zeros((n_splits,4))
track = []
count = 0
while count<n_splits:
    rand = random.randint(0,len(vw_test)-1)
    flag = 0
    for j in track:
        if j==rand:
            flag=1
    if flag==0:
        track.append(rand)
        count = count+1
j = 0
for i in track:
    vw_test_sel[j,:] = vw_test[i,:]
    j = j+1
np.save("vw_test_sel.npy",vw_test_sel) 
    
vw_test_sel = np.load('vw_test_sel.npy') #maintain vw_test_sel from prior experiment
    
    
#2.a-c:Generate and save predictions
data = np.load("hr_data.npy")
lim = n_splits
#lim = n_splits
for split in range(0,lim): #do the following for each train-test split (change to n_splits for experiment)
    #-------------------------------------------------------------------------------------------------------
    #Generate test and train datasets
    
    v_e = vw_test_sel[split,0:2] #Volume fractions to exclude
    w_e = vw_test_sel[split,2:4] #Widths to exclude
    #Find the indices of data points that are to be excluded from training set
    D = len(data)
    inds_test = [] #indices to be excluded from training dataset (ie the test indices)
    inds_train=[] #inidices to be included in training set
    for i in range(0,D):
        v = data[i,2]
        w = data[i,3]
        temp=0
        for v1 in v_e:
            if v==v1:
                temp=1
        if temp==0: #proceed only if v requirement satisfied (improves algorithm efficiency)
            for w1 in w_e:
                if w==w1:
                    temp=1
        if temp==0:
            inds_train.append(i)
        else:
            inds_test.append(i)
    #Use these indices to generate training and testing sets
    P = len(inds_train)
    Q = len(inds_test)
    train_set = np.zeros((P,5))
    test_set = np.zeros((Q,5))
    for i in range(0,P):
        j=inds_train[i]
        train_set[i,:]=data[j,:]
    for i in range(0,Q):
        j=inds_test[i]
        test_set[i,:]=data[j,:]
    inputs_train = train_set[:,range(0,4)]
    outputs_train = train_set[:,4]
    inputs_test = test_set[:,range(0,4)]
    outputs_test = test_set[:,4]
     #---------------------------------------------------------------------------------------------------------
    #Fit k-nearest neighbors (KNN) and random forest (RF) models for this training dataset
    
    #KNN
    KNN = KNeighborsRegressor(n_neighbors = 50, weights = 'uniform') #corrected PMSEM
    #KNN = KNeighborsRegressor(n_neighbors = 1, weights = 'uniform') #from 2nd exp
    KNN.fit(inputs_train, outputs_train)
    #RF
    RF = RandomForestRegressor(n_estimators = 20, min_samples_leaf = 20) #corrected PMSEM the same as before
    #RF = RandomForestRegressor(n_estimators = 1, min_samples_leaf = 1)#from 2nd exp
    RF.fit(inputs_train, outputs_train)
    
    #----------------------------------------------------------------------------------------------------------
    #Now work with EACH of the images in the test set that require interpolation over BOTH v and w
    
    for volf in v_e:
        for wid in w_e:
    
            #--------------------------------------------------------------------------------------------------------------------
            #Generate predictions
            
            #KNN Prediction:
            k = 70
            k_top = np.zeros((71,71))
            for i in range (0,k+1): #x
                for ii in range (0,k+1):#y
                    point = [x_values[i],y_values[ii],volf,wid] #start with top left of unit square
                    temp = KNN.predict([point])
                    if temp > 1.0: #this caps results
                        temp = 1.0
                    if temp < 0.0:
                        temp = 0.0
                    k_top[i,ii] = temp
            
            #RF Prediction:
            r_top = np.zeros((71,71))
            
            for i in range (0,k+1): #x
                for ii in range (0,k+1):#y
                    point = [x_values[i],y_values[ii],volf,wid] #start with top left of unit square
                    temp = RF.predict([point])
                    if temp > 1.0: #this caps results
                        temp = 1.0
                    if temp < 0.0:
                        temp = 0.0
                    r_top[i,ii] = temp
           #Structure: [KNN][RF][v][w]
            #Add columns with VF and w for later use
            v_vals = np.zeros((71,1))
            w_vals = np.zeros((71,1))
            for oo in range (0,71):
                v_vals[oo] = volf
                w_vals[oo] = wid
            p_top = np.hstack([k_top,r_top])
            p_top = np.hstack([p_top,v_vals])
            p_top = np.hstack([p_top,w_vals])
            #------------------------------------------------------------------------------------------------------------------        
            
            #Save Predictions
            if split==0 and volf==v_e[0] and wid==w_e[0]:
                np.save('predictions_1n.npy',p_top)
            else:
                p_array = np.load('predictions_1n.npy')
                p_array = np.vstack([p_array,p_top])
                np.save('predictions_1n.npy',p_array)
    print("Model training and prediction generation progress: "+ str(100.0*float(split+1)/float(lim))+"%")
            