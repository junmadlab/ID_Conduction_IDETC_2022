#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
k = np.load('knn_trajectories_3.npy') #POFMM
c = np.load('c_trajectories.npy')
r = np.load('rf_trajectories_3.npy')

k1 = np.load('knn_trajectories_1n.npy') #PMSEM
r1 = np.load('rf_trajectories_1.npy')

#Find indices where new split begins
k_inds = []
r_inds = []
c_inds = []
for i in range (0,len(k)):
    if k[i,0]==0.0:
        k_inds.append(i)
for i in range (0,len(r)):
    if r[i,0]==0.0:
        r_inds.append(i)
for i in range (0,len(c)):
    if c[i,0]==0.0:
        c_inds.append(i)
        
#Find maximum number of iterations per split for each v,w combo
c_max = c_inds[1]-c_inds[0]
for i in range (1,len(c_inds)-1):
    c_max = max(c_max,c_inds[i+1]-c_inds[i])
c_max = max(c_max,len(c)-1-c_inds[len(c_inds)-1])       
k_max = k_inds[1]-k_inds[0]
for i in range (1,len(k_inds)-1):
    k_max = max(k_max,k_inds[i+1]-k_inds[i])
k_max = max(k_max,len(k)-1-k_inds[len(k_inds)-1])#check last applicable entry
r_max = r_inds[1]-r_inds[0]
for i in range (1,len(r_inds)-1):
    r_max = max(r_max,r_inds[i+1]-r_inds[i])
r_max = max(r_max,len(r)-1-r_inds[len(r_inds)-1])
  


#Find indices where new split begins
k1_inds = []
r1_inds = []
for i in range (0,len(k1)):
    if k1[i,0]==0.0:
        k1_inds.append(i)
for i in range (0,len(r1)):
    if r1[i,0]==0.0:
        r1_inds.append(i)
k1_max = k1_inds[1]-k1_inds[0]
for i in range (1,len(k1_inds)-1):
    k1_max = max(k1_max,k1_inds[i+1]-k1_inds[i])
k1_max = max(k1_max,len(k1)-1-k1_inds[len(k1_inds)-1])#check last applicable entry
r1_max = r1_inds[1]-r1_inds[0]
for i in range (1,len(r1_inds)-1):
    r1_max = max(r1_max,r1_inds[i+1]-r1_inds[i])
r1_max = max(r1_max,len(r1)-1-r1_inds[len(r1_inds)-1])


beta = max(k_max,c_max)
beta = max(beta,r_max)
beta = max(beta,k1_max)
beta = max(beta,r1_max)
        
        


cmins = []

#Normalize the values for each run at each iteration wrt the minimum control objective function value for each IMAGE

#Need to extend trajectories by final value reached   

for i in range (0,len(c_inds)):#do the following for each image
    if i<len(c_inds)-1:
        dif = c_inds[i+1]-c_inds[i] #number of iterations for a given image
    else: #final image
        dif = len(c) -1 - c_inds[i]
    c_sel = c[c_inds[i]:c_inds[i]+dif,:]#objective function values for a given image
    
    #extend trajectory by final value attained to max iter
    if len(c_sel)-1<beta: #if final iteration number is less than the maximum
        temp = beta - (len(c_sel)-1) #difference between final it and max
        c_ext = np.zeros((temp,2))
        count = 0
        for j in range(len(c_sel),beta+1):#generate extended trajectory
            c_ext[count,0] = j #iterations
            c_ext[count,1] = c_sel[len(c_sel)-1,1] #final value to extend
            count = count+1
    
    c_sel = np.vstack([c_sel,c_ext])
    
    cm = np.min(c_sel[:,1]) #minimum of the values
    c_sel[:,1] = c_sel[:,1]/cm #normalize
    if i==0:
        c_norm = c_sel
    else:
        c_norm = np.vstack([c_norm,c_sel])
    cmins = np.append(cmins,cm)
    


for i in range(0,len(k_inds)): #do the following for each image
    if i<len(k_inds)-1:
        dif = k_inds[i+1]-k_inds[i] #number of iterations for a given image
    else: #final image
        dif = len(k) -1 - k_inds[i]
    k_sel = k[k_inds[i]:k_inds[i]+dif,:] #objective function values for a given image
    
    #extend trajectory by final value attained to max iter
    if len(k_sel)-1<beta: #if final iteration number is less than the maximum
        temp = beta - (len(k_sel)-1) #difference between final it and max
        k_ext = np.zeros((temp,2))
        count = 0
        for j in range(len(k_sel),beta+1):#generate extended trajectory
            k_ext[count,0] = j #iterations
            k_ext[count,1] = k_sel[len(k_sel)-1,1] #final value to extend
            count = count+1
    k_sel = np.vstack([k_sel,k_ext])
    
    cm = cmins[i]
    k_sel[:,1] = k_sel[:,1]/cm #normalize
    if i==0:
        k_norm = k_sel
    else:
        k_norm = np.vstack([k_norm,k_sel])

for i in range(0,len(r_inds)): #do the following for each image
    if i<len(r_inds)-1:
        dif = r_inds[i+1]-r_inds[i] #number of iterations for a given image
    else: #final image
        dif = len(r) -1 - r_inds[i]
    r_sel = r[r_inds[i]:r_inds[i]+dif,:] #objective function values for a given image
    
        #extend trajectory by final value attained to max iter
    if len(r_sel)-1<beta: #if final iteration number is less than the maximum
        temp = beta - (len(r_sel)-1) #difference between final it and max
        r_ext = np.zeros((temp,2))
        count = 0
        for j in range(len(r_sel),beta+1):#generate extended trajectory
            r_ext[count,0] = j #iterations
            r_ext[count,1] = r_sel[len(r_sel)-1,1] #final value to extend
            count = count+1
    r_sel = np.vstack([r_sel,r_ext])
    
    
    cm = cmins[i]
    r_sel[:,1] = r_sel[:,1]/cm #normalize
    if i==0:
        r_norm = r_sel
    else:
        r_norm = np.vstack([r_norm,r_sel])

        
#Now average values at each iteration and find standard deviations

k_means = [] 
k_std = []
for i in range (0,beta): #for each iteration (i is the iteration number)
    sm = 0.0
    count = 0.0
    temp= [] #store indices
    for j in range(0,len(k_norm[:,1])): 
        if k_norm[j,0]==i: #check if iteration == selected iteration
            sm = sm + k_norm[j,1]
            temp = np.append(temp,j)
            count = count+1.0
    avg = [sm/count]
    k_means = np.append(k_means,avg)
    #standard dev
    dif = 0.0
    for q in temp:
        dif = dif + (k_norm[int(q),1] - avg)**2
    std = (dif/len(temp))**0.5
    k_std = np.append(k_std,std)
    
r_means = []
r_std = []
for i in range (0,beta): #for each iteration (i is the iteration number)
    sm = 0.0
    count = 0.0
    temp = []
    for j in range(0,len(r_norm[:,1])): 
        if r_norm[j,0]==i: #check if iteration == selected iteration
            sm = sm+r_norm[j,1]
            temp = np.append(temp,j)
            count = count+1.0
    avg = [sm/count]
    r_means = np.append(r_means,avg) 
    #standard dev
    dif = 0.0
    for q in temp:
        dif = dif + (r_norm[int(q),1] - avg)**2
    std = (dif/len(temp))**0.5
    r_std = np.append(r_std,std)
    
c_means = []
c_std = []
for i in range (0,beta): #for each iteration (i is the iteration number)
    sm = 0.0
    count = 0.0
    temp = []
    for j in range(0,len(c_norm[:,1])): 
        if c_norm[j,0]==i: #check if iteration == selected iteration
            sm = sm+c_norm[j,1]
            temp = np.append(temp,j)
            count = count+1.0 #will be 200 when finished
    avg = [sm/count]
    c_means = np.append(c_means,avg) 
    #standard dev
    dif = 0.0
    for q in temp:
        dif = dif + (c_norm[int(q),1] - avg)**2
    std = (dif/len(temp))**0.5
    c_std = np.append(c_std,std)
        
#-----------------------------------------------------------------------------------
#pixelwise

        
        


#Normalize the values for each run at each iteration wrt the minimum control objective function value for each IMAGE

#Need to extend trajectories by final value reached   


for i in range(0,len(k1_inds)): #do the following for each image
    if i<len(k1_inds)-1:
        dif = k1_inds[i+1]-k1_inds[i] #number of iterations for a given image
    else: #final image
        dif = len(k1) - 1 - k1_inds[i]
    k1_sel = k1[k1_inds[i]:k1_inds[i]+dif,:] #objective function values for a given imag
    
    
        #extend trajectory by final value attained to max iter
    if len(k1_sel)-1<beta: #if final iteration number is less than the maximum
        temp = beta - (len(k1_sel)-1) #difference between final it and max
        k1_ext = np.zeros((temp,2))
        count = 0
        for j in range(len(k1_sel),beta+1):#generate extended trajectory
            k1_ext[count,0] = j #iterations
            k1_ext[count,1] = k1_sel[len(k1_sel)-1,1] #final value to extend
            count = count+1
    k1_sel = np.vstack([k1_sel,k1_ext])
    
    
    
    cm = cmins[i]
    k1_sel[:,1] = k1_sel[:,1]/cm #normalize
    if i==0:
        k1_norm = k1_sel
    else:
        k1_norm = np.vstack([k1_norm,k1_sel])

for i in range(0,len(r1_inds)): #do the following for each image
    if i<len(r1_inds)-1:
        dif = r1_inds[i+1]-r1_inds[i] #number of iterations for a given image
    else: #final image
        dif = len(r1) -1 -  r1_inds[i]
    r1_sel = r1[r1_inds[i]:r1_inds[i]+dif,:] #objective function values for a given image
    
        #extend trajectory by final value attained to max iter
    if len(r1_sel)-1<beta: #if final iteration number is less than the maximum
        temp = beta - (len(r1_sel)-1) #difference between final it and max
        r1_ext = np.zeros((temp,2))
        count = 0
        for j in range(len(r1_sel),beta+1):#generate extended trajectory
            r1_ext[count,0] = j #iterations
            r1_ext[count,1] = r1_sel[len(r1_sel)-1,1] #final value to extend
            count = count+1
    r1_sel = np.vstack([r1_sel,r1_ext])
    
    
    cm = cmins[i]
    r1_sel[:,1] = r1_sel[:,1]/cm #normalize
    if i==0:
        r1_norm = r1_sel
    else:
        r1_norm = np.vstack([r1_norm,r1_sel])

        
#Now average values at each iteration

k1_means = []
k1_std = []
for i in range (0,beta): #for each iteration (i is the iteration number)
    sm = 0.0
    count = 0.0
    temp = []
    for j in range(0,len(k1_norm[:,1])): 
        if k1_norm[j,0]==i: #check if iteration == selected iteration
            sm = sm+k1_norm[j,1]
            temp = np.append(temp,j)
            count = count+1.0 #will be 200 when finished
    avg = [sm/count]
    k1_means = np.append(k1_means,avg) 
    #standard dev
    dif = 0.0
    for q in temp:
        dif = dif + (k1_norm[int(q),1] - avg)**2
    std = (dif/len(temp))**0.5
    k1_std = np.append(k1_std,std)
    
r1_means = []
r1_std = []
for i in range (0,beta): #for each iteration (i is the iteration number)
    sm = 0.0
    count = 0.0
    temp = []
    for j in range(0,len(r1_norm[:,1])): 
        if r1_norm[j,0]==i: #check if iteration == selected iteration
            sm = sm+r1_norm[j,1]
            temp = np.append(temp,j)
            count = count+1.0 #will be 200 when finished
    avg = [sm/count]
    r1_means = np.append(r1_means,avg) 
    #standard dev
    dif = 0.0
    for q in temp:
        dif = dif + (r1_norm[int(q),1] - avg)**2
    std = (dif/len(temp))**0.5
    r1_std = np.append(r1_std,std)

    

#--------------------------------------------------------------------------------------    





#Plot the data

x_k = np.arange(0,beta)
x_r = np.arange(0,beta)
x_c = np.arange(0,beta)
x_k1 = np.arange(0,beta)
x_r1 = np.arange(0,beta)
c_means = c_means[0:beta]
k_means = k_means[0:beta]
r_means = r_means[0:beta]
r1_means = r1_means[0:beta]
k1_means = k1_means[0:beta]

#stat stuff
#Assuming t-distribution        
k_err= np.zeros((beta))
r_err = np.zeros((beta))
c_err = np.zeros((beta))
k1_err = np.zeros((beta))
r1_err = np.zeros((beta))

#from https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals_print.html#:~:text=The%20t%20value%20for%2095,%3D%209%20is%20t%20%3D%202.262.
t = 2.262 #for 95% two-tale confidence interval using t distribution at 9 dof (10-1=9)
N = 10.0 #number of data points
#C_I = mean +/- t*s/sqrt(N)

#Now find the standard deviation of the objective function values for each iteration
for i in range (0,beta):
    k_err[i] = k_std[i]*t/np.sqrt(N)
    r_err[i] = r_std[i]*t/np.sqrt(N)
    c_err[i] = c_std[i]*t/np.sqrt(N)
    k1_err[i] = k1_std[i]*t/np.sqrt(N)
    r1_err[i] = r1_std[i]*t/np.sqrt(N)





fig, ax = plt.subplots()

#Learning Curve: method from https://stackoverflow.com/questions/20130227/matplotlib-connect-scatterplot-points-with-line-python
pc,=plt.plot(x_c, c_means,label = "Control")
pk,=plt.plot(x_k,k_means, label="KNN (Objective Minimization Method)")
pr,=plt.plot(x_r,r_means, label="RF (Objective Minimization Method)")
pk1,=plt.plot(x_k1,k1_means, label="KNN (MSE Method)")
pr1,=plt.plot(x_r1,r1_means, label="RF (MSE Method)")

ax.fill_between(x_c, c_means-c_err, c_means+ c_err, alpha=0.2)
ax.fill_between(x_k, k_means-k_err, k_means+ k_err, alpha=0.2)
ax.fill_between(x_r, r_means-r_err, r_means+ r_err, alpha=0.2)
ax.fill_between(x_k1, k1_means-k1_err, k1_means+ k1_err, alpha=0.2)
ax.fill_between(x_r1, r1_means-r1_err, r1_means+ r1_err, alpha=0.2)

plt.xlabel("Iterations",fontsize = 20)
plt.ylabel("Mean Normalized Objective Function Value",fontsize = 20)
#plt.title("Mean Normalized Objective Function Value vs Iteration",fontsize = 30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(handles=[pc,pk,pr,pk1,pr1])
plt.legend(prop={"size":10})
#Now save the figure to an external file. MUST occur before plt.show (otherwise saved pic blank)
#https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
#https://blakeaw.github.io/2020-05-25-improve-matplotlib-notebook-inline-res/ for improving resolution
dpi = 300
plt.rcParams['figure.dpi'] = dpi
plt.rcParams['savefig.dpi'] = dpi
plt.savefig('Optimization_Trajectories.png')
plt.show()