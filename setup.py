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
#create additional arrays to hold cn values
k_cn = np.asarray([0])
r_cn = np.asarray([0])
c_cn = np.asarray([0])
np.save('k_cn.npy',k_cn)
np.save('r_cn.npy',r_cn)
np.save('c_cn.npy',c_cn)