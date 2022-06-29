#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import time as tm
import os
it_array = np.load('it.npy') #iterations

NNN = 70 #70 for experiments
k = NNN #discretization resolution: somewhat arbitrary. NOTE: Increasing the int coeff dramatically increases model training and testing #time!!!
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

vols = [0.3,0.315,0.33,0.345,0.36,0.375,0.39,0.405,0.42,0.435,0.45,0.465,0.48,0.495,0.51,0.525,0.54,0.555,0.57,0.585,0.6]
widths = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]


it_array = np.load('it.npy') #iterations
vol_it = it_array[0]
width_it = it_array[1]
run_it = it_array[2]
overall_it = it_array[3]

    
vol_f = vols[vol_it]
width = widths[width_it]    
                    
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
    if run_it==0:
        MM = V
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

parameters = {"acceptable_tol": tolerance, "maximum_iterations": 100,"file_print_level": 5, "output_file":'hrdtxt.txt'}
m.solver_options = ["print_level 2"]
solver = IPOPTSolver(Problem, parameters=parameters)
t = tm.time()
print("Starting optimization for v = "+str(vfs)+", w = "+str(wis)+", run "+ str(run_it)+".")
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
#-------------------------------------------------------------------------------------------
#Discretize results

if run_it==2: #if final run reached
    
    #Now store the results of this run (x,y,v,w,a)
    results = np.zeros(((NNN+1)**2,5))
    ind = 0
    for xs in x_values:
        for ys in y_values:
            results[ind,0] = xs
            results[ind,1] = ys
            results[ind,2] = vol_f
            results[ind,3] = width
            results[ind,4] = a_opt(xs,ys)
            ind = ind+1
    #Naming convention: hr_data_v=0.5_w=0.5_.npy, for example     
    filename = "hr_data_v="+str(vol_f)+"_w="+str(width)+"_.npy"
    np.save(filename,results)           
#----------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#Update Iterations
run_it = run_it + 1 #iterate run

if run_it==3: #check if runs completed for an image
    progress = 100.0*float(overall_it+1)/(3.0*441.0) 
    print("Progress: " + str(progress) + "%.")
    run_it=0
    if width_it<len(widths)-1:
        width_it=width_it+1 #iterate width
    else: #width_it==len(widths)-1, need new volume fraction
        width_it = 0
        vol_it = vol_it + 1 #iterate volume frac bound iteration
        
it_array[0] = vol_it
it_array[1] = width_it
it_array[2] = run_it
it_array[3] = overall_it+1 
np.save('it.npy', it_array)
    

#Now kill the kernel to save resources and accelerate the process
pid = os.getpid()
command = 'kill -9 '+str(pid)
os.system(command)

