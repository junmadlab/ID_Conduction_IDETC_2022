#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Naming convention: k_v=0.5_w=0.5_3.npy, for example     
# Also: names ending in 1 or 1n: PMSEM. In 3: POFMM. This is an arbitrary choice that can be changed
import numpy as np
import os
#retrieve array used in testing
x = np.load('vw_test_sel.npy')
z = 0 #image it
for i in range (0, 50):
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:
            #KNN:
            filename = "k_v="+str(v)+"_w="+str(w)+"_"+str(z)+".npy"
            top = np.load(filename)
            if z==0:
                np.save('knn_trajectories_1n.npy',top)
            else:
                tops = np.load('knn_trajectories_1n.npy')
                new_tops = np.vstack([tops,top])
                np.save('knn_trajectories_1n.npy',new_tops)
            #Now delete unneeded file
            os.remove(filename)
            print(filename)
            z = z+1
            
z = 0 #image it
for i in range (0, 50):
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:     
            #RF:
            filename = "r_v="+str(v)+"_w="+str(w)+"_"+str(z)+".npy"
            top = np.load(filename)
            if z==0:
                np.save('rf_trajectories_3.npy',top)
            else:
                tops = np.load('rf_trajectories_3.npy')
                new_tops = np.vstack([tops,top])
                np.save('rf_trajectories_3.npy',new_tops)
            #Now delete unneeded file
            os.remove(filename)
            print(filename)
            z = z+1
z = 0 #image it
for i in range (0, 50):
    vols = x[i,0:2]
    widths = x[i,2:4]
    for v in vols:
        for w in widths:
           #Control:
            filename = "c_v="+str(v)+"_w="+str(w)+"_"+str(z)+".npy"
            top = np.load(filename)
            if z==0:
                np.save('c_trajectories.npy',top)
            else:
                tops = np.load('c_trajectories.npy')
                new_tops = np.vstack([tops,top])
                np.save('c_trajectories.npy',new_tops)
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


