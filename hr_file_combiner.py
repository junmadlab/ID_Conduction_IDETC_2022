#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Naming convention: hr_data_v=0.5_w=0.5_.npy, for example     

import numpy as np
import os
vols = [0.3,0.315,0.33,0.345,0.36,0.375,0.39,0.405,0.42,0.435,0.45,0.465,0.48,0.495,0.51,0.525,0.54,0.555,0.57,0.585,0.6]
widths = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
for v in vols:
    for w in widths:
        filename = "hr_data_v="+str(v)+"_w="+str(w)+"_.npy"
        top = np.load(filename)
        if v==vols[0] and w==widths[0]:
            np.save('hr_data.npy',top)
        else:
            tops = np.load('hr_data.npy')
            new_tops = np.vstack([tops,top])
            np.save('hr_data.npy',new_tops)
        #Now delete unneeded file
        os.remove(filename)
        
print("Finished combining files.")