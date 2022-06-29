#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import time as tm
import os
vw_test_sel = np.load('vw_test_sel.npy')
it_array = np.load('it_array.npy')
#2.d-f: convert predictions into function form and use these to generate optimization data
preds = np.load('predictions_1n.npy') #predictions for first experiment, predictions_2 for second part 1, predictions_3 for second part 2
#predictions array format: [KNN][Rf][v][w] = [71][71][1][1] columnwise
k_preds = preds[:,0:71]
xx = it_array[0] #split iteration
zz = it_array[1] #image iteration (4 images per split)
uu = it_array[2] #run iteration (4 runs per image to generate trajectory)
v_e = vw_test_sel[xx,0:2]
w_e = vw_test_sel[xx,2:4]
if zz==0:
    volf = v_e[0]
    wid = w_e[0]
elif zz==1:
    volf = v_e[0]
    wid = w_e[1]
elif zz==2:
    volf = v_e[1]
    wid = w_e[0]
else:#zz==3
    volf = v_e[1]
    wid = w_e[1]
#Load predictions
gg = it_array[3] #overall knn iter
k_top = preds[71*gg:71*(gg+1),0:71] #KNN prediction for given volf and wid
#----------------------------------------------------------------------------------------------------
#Convert predictions to function form so Fenics can use them
           
#KNN conversion
from dolfin import *
image = k_top
N = 70 
mesh = UnitSquareMesh(N, N)
x = mesh.coordinates().reshape((-1, 2))
# Vertex with cordinates x[i] has a value in the image at ii[i], jj[i]
h = 1./N
ii, jj = x[:, 0]/h, x[:, 1]/h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)
# Turn image into CG1 function
# Values are vertex ordered here
image_values = image[ii, jj] 
V = FunctionSpace(mesh, 'CG', 1)
image_f = Function(V)
# Values will be dof ordered
d2v = dof_to_vertex_map(V)
image_values = image_values[d2v]
image_f.vector()[:] = image_values
# Image manip
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = inner(image_f, v)*dx
# Create vector that spans the null space and normalize
u1 = Function(V)
null_vec = Vector(u1.vector())
V.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm("l2")
            
            
#--------------------------------------------------------------------------------------------
            
#Now use these predicted functions to warm-start the optimization process for KNN, RF, and a control
            
vol_f = volf
width = wid

                    
#--------------------------------------------------------------------------------------------
#Now set up and run optimization sets
from fenics import *
from fenics_adjoint import *
try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
    When compiling IPOPT, make sure to link against HSL, as it \
    is a necessity for practical problems.""")
    raise
# turn off redundant output in parallel
parameters["std_out_all_processes"] = False
V = Constant(vol_f)  # volume bound on the control.   Default = 0.4
p = Constant(5)  # power used in the solid isotropic material.  Default = 5
# with penalisation (SIMP) rule, to encourage the control
# solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8)  # regularisation coefficient in functional


def k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
    rule, equation (11)."""
    return eps + (1 - eps) * a ** p
n = 70
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

lb_2 = 0.5 - width; #lower bound on section of bottom face which is adiabatic
ub_2 = 0.5 + width; #Upper bound on section of bottom face which is adiabatic
class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""

    def inside(self, x, on_boundary):
        # return (x[0] == 0.0 or x[1] == 1.0) and on_boundary
        return (x[0] == 0.0 or x[1] == 1.0 or x[0] == 1.0 or ( x[1] == 0.0 and  (x[0] < lb_2 or x[0] > ub_2)  )  ) # modified from Fuge


# the Dirichlet BC; the Neumann BC will be implemented implicitly by
# dropping the surface integral after integration by parts

T_bc = 0.0;
bc = [DirichletBC(P, T_bc, WestNorth())]
f_val = 1.0e-2 #Default = 1.0e-2
f = interpolate(Constant(f_val), P)  # the volume source term for the PDE


def forward(a):
    """Solve the forward problem for a given material distribution a(x)."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)
    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,"maximum_iterations": 20}})                                                         
    return T
if __name__ == "__main__":
    if uu==0:
        MM = image_f
    else:
        #Adapted from https://fenicsproject.discourse.group/t/read-mesh-from-xdmf-file-write-checkpoint/3458/3
        mesh1 = UnitSquareMesh(70, 70)
        V1 =  FunctionSpace(mesh, "CG", 1)
        sol = Function(V1)
        with XDMFFile("out.xdmf") as infile:
            infile.read(mesh1)
            infile.read_checkpoint(sol, "u")
        MM = sol
    a = interpolate(MM, A)  # initial guess.
    T = forward(a)  # solve the forward problem once.
    controls = File("output/control_iterations.pvd")
    a_viz = Function(A, name="ControlVisualisation")


def eval_cb(j, a):
    a_viz.assign(a)
    controls << a_viz
J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
m = Control(a)
Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
lb = 0.0
ub = 1.0
class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

    def __init__(self, V):
        self.V = float(V)
        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)
        integral = self.smass.inner(self.tmpvec.vector())
        if MPI.rank(MPI.comm_world) == 0:
            #print("Current control integral: ", integral)
            return [self.V - integral]

    def jacobian(self, m):
        return [-self.smass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1
Problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))
tolerance = 1.0e-3 #default: 1.0e-3
wis = width
yy = 0 #Only complete 1 run per image
vfs = vol_f
    
filename = "k_v="+str(vol_f)+"_w="+str(wis)+"_s="+str(uu)#filetype added later
filename1 = str(filename)+".txt"
parameters = {"acceptable_tol": tolerance, "maximum_iterations": 100,"file_print_level": 5,"output_file":filename1}
m.solver_options = ["print_level 2"]
solver = IPOPTSolver(Problem, parameters=parameters)
t = tm.time()
print("Starting optimization for KNN model, v = "+str(vfs)+", w = "+str(wis)+", run "+ str(uu)+".")
a_opt = solver.solve()
t = tm.time() - t
print("Finished optimization run in "+str(t)+" seconds.")
#Adapted from https://fenicsproject.discourse.group/t/read-mesh-from-xdmf-file-write-checkpoint/3458/3
mesh1 = UnitSquareMesh(70, 70)
V1 =  FunctionSpace(mesh, "CG", 1)
sol = a_opt
with XDMFFile("out.xdmf") as outfile:
    outfile.write(mesh1)
    outfile.write_checkpoint(sol, "u", 0, append=True)
                    
#----------------------------------------------------------------------------------------------
#Now Read the text file and delete it once important info is stored in a .npy array
#open and read text file
#from https://www.askpython.com/python/built-in-methods/python-read-file    and      https://www.pythontutorial.net/python-basics/python-read-text-file/
filename = "k_v="+str(vol_f)+"_w="+str(wis)+"_s="+str(uu)+".txt" #filetype added later
text = open(filename)
text = text.read()



#Pattern: After [ls\n], there are 3 spaces before the iteration number, followed by 2 spaces before the objective 
#function value is listed. This value's information continues until the next space. The next line can be accessed
#after [\n] . There are a maximum of 10 iterations listed per section (0-9). 

#need to account for end of file
objectives  = ''
n = len(text) #includes spaces and \n as whole
for i in range (0, n-2):
    if text[i]=='l' and text[i+1]=='s' and text[i+2]=='\n':
        count = 0 #number of current line in 'block'
        j = i+9 #starting index of obj value
        cond = 0 #condition will be set to 1 if approach end of blocks
        while count < 10 and cond == 0:#only do the following while in a block AND if not near end of blocks  
            while text[j]!=' ':
                objectives = objectives + text[j]
                j = j+1
            objectives = objectives + ", "
            while text[j]!='\n':#when done, j will have index of \n
                j=j+1 
            temp = j+1 #first index of next line
            #Now set j to be the index of the next line's obj function
            j = temp+6
            count = count + 1
            #check to see if we are at the end of the block:
            if text[temp]=='\n':
                    cond = 1
#objectives will always have a comma and a space at the end, which is unacceptable
m = len(objectives)
p = ''
for i in range (0, m-2):
    p = p  + objectives[i]
objectives = p
objectives

objectives = [float(s) for s in objectives.split(',')] #from https://stackoverflow.com/questions/19334374/how-to-convert-a-string-of-space-and-comma-separated-numbers-into-a-list-of-in
objectives = np.asarray(objectives)
if uu==0:
    results = np.zeros((0,2))
else: #uu>0
    results = np.load("k_v="+str(vol_f)+"_w="+str(wis)+"_"+str(gg)+".npy")
k_cn = np.load('k_cn.npy')
cn = k_cn[0]
for i in range (0,len(objectives)):
    info = np.hstack([cn,objectives[i]])
    results = np.vstack([results,info])
    cn = cn+1

#Now delete unecessary files (cleanup)
filename3 = "k_v="+str(vol_f)+"_w="+str(wis)+ "_s="+str(uu)+".txt"
os.remove(filename3)
                
filename2 = "k_v="+str(vol_f)+"_w="+str(wis)+"_"+str(gg)+".npy"
np.save(filename2,results)   

#Now update iteration array
#xx - split it (0,1,2,3), zz - image it (0,1,2,3), uu - run it (0,1,2,3), gg - overall it (update every image)

if uu < 3:
    uu = uu + 1 #update run iteration
else: #uu==3, ie this was the final run for a given image
    uu = 0 #reset runs
    gg = gg + 1 #reset overall it (tracks the number of images used)
    cn = 0 #reset cn values
    if zz < 3:
        zz = zz + 1 #update image iteration
    else:#zz==3, ie this was the final image in a given split
        zz = 0 #reset image number
        xx = xx + 1 #move on to next split
it_array[0] = xx
it_array[1] = zz
it_array[2] = uu
it_array[3] = gg
np.save('it_array.npy', it_array)
k_cn[0] = cn
np.save('k_cn.npy',k_cn)

if uu==4: #after the end of a set of runs for a given image
    progress = 100.0*float(gg)/(4*50) #4 images per split, 50 splits (may change in experiment)
    print("KNN Trajectories Progress: " + str(progress) + "%.")
    

#Now kill the kernel to save resources and accelerate the process
pid = os.getpid()
command = 'kill -9 '+str(pid)
os.system(command)