#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
it_array = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0]) 
#Iterations tracker: 
#0: knn split it,  1: knn image it, 2: knn run it, 3: knn overall it
#4: rf split it, 5: rf image it, 6: rf run it, 7: rf overall it,
#8: control split it, 9: control image it, 10: control run it, 11: control overall it
np.save('it_array.npy',it_array)

khp = np.asarray([0,0]) #neighbors, weighting 
rhp = np.asarray([0,0]) #n_estimators, min_samples_leaf 
np.save('khp.npy',khp)
np.save('rhp.npy',rhp)
