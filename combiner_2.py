#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os


n_splits = 10

#KNN hyperparameters
neighbors = [1,2,5,10,20,30,40,50,60,70,80,90,100] #number of neighbors to consider 
weight = ['distance','uniform'] #weighting scheme


#RF Hyperparameters
num_trees = [1,5,10,15,20,25,30] #n_estimators
leaf = [1,5,10,15,20] #min_samples_leafs


#retrieve array used in testing
x = np.load('vw_test_sel.npy')
z = 0 #image it
for i in range (0, n_splits): #10 splits
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:
            #KNN:
            for n_it in range (0, len(neighbors)):
                for w_it in range (0, len(weight)):
                    filename ="k_neighbors="+str(neighbors[n_it])+"_weighting="+str(weight[w_it])+"_split_"+str(i)+"_v="+str(v)+"_w="+str(w)+".npy"
                    top = np.load(filename)
                    if z==0:
                        np.save('knn_objectives.npy',top)
                    else:
                        tops = np.load('knn_objectives.npy')
                        new_tops = np.vstack([tops,top])
                        np.save('knn_objectives.npy',new_tops)
                    #Now delete unneeded file
                    os.remove(filename)
                    print(filename)
                    z = z+1
            
z = 0 #image it
for i in range (0, n_splits):
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:     
            #RF:
            for n_it in range (0, len(num_trees)):
                for l_it in range (0, len(leaf)):
                    filename ="r_estimators="+str(num_trees[n_it])+"_min_samples_leaf="+str(leaf[l_it])+"_split_"+str(i)+"_v="+str(v)+"_w="+str(w)+".npy"
                    top = np.load(filename)
                    if z==0:
                        np.save('rf_objectives.npy',top)
                    else:
                        tops = np.load('rf_objectives.npy')
                        new_tops = np.vstack([tops,top])
                        np.save('rf_objectives.npy',new_tops)
                    #Now delete unneeded file
                    os.remove(filename)
                    print(filename)
                    z = z+1
z = 0 #image it
for i in range (0, n_splits):
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:
            #Control:
            filename = "c_"+"_split_"+str(i)+"_v="+str(v)+"_w="+str(w)+".npy" 
            top = np.load(filename)
            if z==0:
                np.save('c_objectives.npy',top)
            else:
                tops = np.load('c_objectives.npy')
                new_tops = np.vstack([tops,top])
                np.save('c_objectives.npy',new_tops)
            #Now delete unneeded file
            os.remove(filename)
            print(filename)
            z = z+1
print("Finished combining files.")

#Now delete additional unneeded files
it_array = np.load('it_array.npy')
print("Iterations Array: "+ str(it_array))
os.remove('it_array.npy')
os.remove('out.xdmf')
os.remove('output/control_iterations.pvd')
print('Finished cleaning unneeded files.')
print('The experiment is now finished. :) ')


