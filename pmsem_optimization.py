#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from numpy import random
import time as time
#Successive 1-D optimizations

data = np.load("hr_data.npy")
#code adapted from enme440 final project

num_trees = [1,5,10,15,20,25,30] 
leaf = [1,5,10,15,20]
num_trials = 10 #number of trials to run on each model
MSE_Train = np.zeros((len(num_trees),len(leaf),num_trials)) #stores training MSE values for  train/test splits 
MSE_Test = np.zeros((len(num_trees),len(leaf),num_trials)) #stores test MSE values for train/test splits 
#---------------------------------------------------------------------------------------------------------------------------
progress = 0
for tree_it in range (0,len(num_trees)):
    for leaf_it in range (0,len(leaf)):
        for trial_number in range (0,num_trials):
            t = time.time()
            L = 2 #length of array (number of rows of w and v each to hold out for testing)
            v_e = [] #vol fracs to be excluded from training. Note: not using np.zeros(()) as this would impact process below
            w_e = [] #half-widths to be excluded from training
            for i in range(0,L):
                temp1=0
                temp2=0 
                while(temp1==0):
                    count = 0
                    rand = round(0.3 + 3.0*random.randint(0,10)/100.0,2) #inclusive
                    #check if "random" number is already in set
                    for v in v_e:
                        if rand== v:
                            count = 1
                    if count == 0 :
                        v_e.append(rand)
                        temp1 = 1 #exit the loop
                while(temp2==0):
                    count = 0
                    rand = round(random.randint(0,10)/10.0,2) #inclusive
                    #check if "random" number is already in set
                    for w in w_e:
                        if rand== w:
                            count = 1
                    if count == 0 :
                        w_e.append(rand)
                        temp2 = 1 #exit the loop


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
                if temp==0: #proceed only if v requirement sat (improves algorithm efficiency)
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
  
 #-----------------------------------------------------------------------------------------------------------------------


            #fit model for each trial

            clf = RandomForestRegressor(n_estimators = num_trees[tree_it], min_samples_leaf = leaf[leaf_it] )
            clf.fit(inputs_train, outputs_train)
            bob = clf

            #to get Test MSEs:   
            outputs_pred_test = bob.predict(inputs_test)
            #this is an adaptation to the model to curb unreal results
            for i in range (0,len(outputs_pred_test)):
                if outputs_pred_test[i] > 1.0:
                    outputs_pred_test[i] = 1.0
                if outputs_pred_test[i] < 0.0:
                    outputs_pred_test[i] = 0.0
            MSE_Test[tree_it,leaf_it,trial_number] = mean_squared_error(outputs_test,outputs_pred_test,sample_weight=None, multioutput='uniform_average', squared=True)


            #to get train MSEs:
            outputs_pred_train = bob.predict(inputs_train)
            for i in range (0,len(outputs_pred_train)):
                if outputs_pred_train[i] > 1.0:
                    outputs_pred_train[i] = 1.0
                if outputs_pred_train[i] < 0.0:
                    outputs_pred_train[i] = 0.0
            MSE_Train[tree_it,leaf_it,trial_number] = mean_squared_error(outputs_train,outputs_pred_train, sample_weight=None, multioutput='uniform_average', squared=True)
            progress = progress + 1
            p = 100.0*float(progress)/float(len(num_trees)*num_trials*len(leaf))
            t = time.time() - t
            print("Progress: " + str(p) + "% ( iteration = "+str(trial_number)+"). Trial time: "+ str(t)+ " seconds.")
            print("n_estimators: " + str(num_trees[tree_it]) + ", min_samples_leaf: " + str(leaf[leaf_it]))
        filename = "RF_Test_MSE.npy"
        np.save(filename,MSE_Test)
        filename = "RF_Train_MSE.npy"
        np.save(filename,MSE_Train)

#Mean MSEs
MSE_Test = np.load("RF_Test_MSE_n.npy")#original was RF_Test_MSE
MSE_Train = np.load("RF_Train_MSE_n.npy")
MSE_test_means = np.zeros((len(num_trees),len(leaf)))
MSE_test_std = np.zeros((len(num_trees),len(leaf)))
MSE_train_means= np.zeros((len(num_trees),len(leaf)))
MSE_train_std = np.zeros((len(num_trees),len(leaf)))
for i in range (0,len(num_trees)):
    for j in range (0,len(leaf)):
        MSE_test_means[i,j] = np.mean(MSE_Test[i,j,:])
        MSE_train_means[i,j] = np.mean(MSE_Train[i,j,:])
        MSE_test_std[i,j] = np.std(MSE_Test[i,j,:])
        MSE_train_std[i,j] = np.std(MSE_Train[i,j,:])
MSE_test_means = MSE_test_means[0:7,:]
MSE_test_std = MSE_test_std[0:7,:]
np.save('RF_mse_test.npy',MSE_test_means)
np.save('RF_mse_test_std.npy',MSE_test_std)
    
#KNN       
neighbors = [1,2,5,10,20,30,40,50,60,70,80,90,100] 
weighting = ['distance','uniform']
num_trials = 10 #number of trials to run on each model
MSE_Train = np.zeros((len(neighbors),len(weighting),num_trials)) #stores training MSE values for  train/test splits 
MSE_Test = np.zeros((len(neighbors),len(weighting),num_trials)) #stores test MSE values for train/test splits 
#---------------------------------------------------------------------------------------------------------------------------
progress = 0
for nei_it in range (0,len(neighbors)):
    for wei_it in range (0,len(weighting)):
        for trial_number in range (0,num_trials):
            t = time.time()
            L = 2 #length of array (number of rows of w and v each to hold out for testing)
            v_e = [] #vol fracs to be excluded from training. Note: not using np.zeros(()) as this would impact process below
            w_e = [] #half-widths to be excluded from training
            for i in range(0,L):
                temp1=0
                temp2=0 
                while(temp1==0):
                    count = 0
                    rand = round(0.3 + 3.0*random.randint(0,10)/100.0,2) #inclusive
                    #check if "random" number is already in set
                    for v in v_e:
                        if rand== v:
                            count = 1
                    if count == 0 :
                        v_e.append(rand)
                        temp1 = 1 #exit the loop
                while(temp2==0):
                    count = 0
                    rand = round(random.randint(0,10)/10.0,2) #inclusive
                    #check if "random" number is already in set
                    for w in w_e:
                        if rand== w:
                            count = 1
                    if count == 0 :
                        w_e.append(rand)
                        temp2 = 1 #exit the loop


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
                if temp==0: #proceed only if v requirement sat (improves algorithm efficiency)
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
  
 #-----------------------------------------------------------------------------------------------------------------------


            #fit model for each trial
            KNN = KNeighborsRegressor(n_neighbors = neighbors[nei_it], weights = weighting[wei_it])
            KNN.fit(inputs_train, outputs_train)
            bob = KNN
            #to get Test MSEs:   
            outputs_pred_test = bob.predict(inputs_test)
            #this is an adaptation to the model to curb unreal results
            for i in range (0,len(outputs_pred_test)):
                if outputs_pred_test[i] > 1.0:
                    outputs_pred_test[i] = 1.0
                if outputs_pred_test[i] < 0.0:
                    outputs_pred_test[i] = 0.0
            MSE_Test[nei_it,wei_it,trial_number] = mean_squared_error(outputs_test,outputs_pred_test,sample_weight=None, multioutput='uniform_average', squared=True)


            #to get train MSEs:
            outputs_pred_train = bob.predict(inputs_train)
            for i in range (0,len(outputs_pred_train)):
                if outputs_pred_train[i] > 1.0:
                    outputs_pred_train[i] = 1.0
                if outputs_pred_train[i] < 0.0:
                    outputs_pred_train[i] = 0.0
            MSE_Train[tree_it,leaf_it,trial_number] = mean_squared_error(outputs_train,outputs_pred_train, sample_weight=None, multioutput='uniform_average', squared=True)
            progress = progress + 1
            p = 100.0*float(progress)/float(len(num_trees)*num_trials*len(leaf))
            t = time.time() - t
            print("Progress: " + str(p) + "% ( iteration = "+str(trial_number)+"). Trial time: "+ str(t)+ " seconds.")
            print("n_estimators: " + str(num_trees[tree_it]) + ", min_samples_leaf: " + str(leaf[leaf_it]))
        filename = "KNN_Test_MSE_n.npy"
        np.save(filename,MSE_Test)
        filename = "KNN_Train_MSE_n.npy"
        np.save(filename,MSE_Train)

#Mean MSEs
MSE_Test2 = np.load("KNN_Test_MSE_n.npy")#original was RF_Test_MSE
MSE_Train2 = np.load("KNN_Train_MSE_n.npy")
MSE_test_means2 = np.zeros((len(neighbors),len(weightings)))
MSE_test_std2 = np.zeros((len(neighbors),len(weightings)))
MSE_train_means2= np.zeros((len(neighbors),len(weightings)))
MSE_train_std2 = np.zeros((len(neighbors),len(weightings)))
for i in range (0,len(neighbors)):
    for j in range (0,len(weightings)):
        MSE_test_means2[i,j] = np.mean(MSE_Test2[i,j,:])
        MSE_train_means2[i,j] = np.mean(MSE_Train2[i,j,:])
        MSE_test_std2[i,j] = np.std(MSE_Test2[i,j,:])
        MSE_train_std2[i,j] = np.std(MSE_Train2[i,j,:])
MSE_test_means2 = MSE_test_means2[0:7,:]
MSE_test_std2 = MSE_test_std2[0:7,:]
np.save('KNN_mse_test.npy',MSE_test_means2)
np.save('KNN_mse_test_std.npy',MSE_test_std2)


print("Done.")
