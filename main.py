import numpy as np
np.set_printoptions(precision=3,suppress=True)

# =============================================================================

from SystemClass import System
from Functions import findStability, constraintChecker
from MPCDesignClass import eMPC
from QuadProgramClass import HQP, PQP
import matplotlib.pyplot as plt

#Begin
S = System()

#state space dimensions (states, inputs, outputs, setpoints)
n = 5
m = 2
p = 5
r = 2

S.updateDims(n,m,p,r)

#LTI Continuous time system model
A_c = [[-0.026, 0.074, -0.804, -9.809, 0.000],\
       [-0.242, -2.017, 73.297, -0.105, -0.001],\
       [0.003, -0.153, -2.941, 0.000, 0.000],\
       [0.000, 0.000, 1.000, 0.000, 0.000],\
       [-0.011, 1.000, 0.000, -75.000, 0.000]]   

B_c = [[4.594, 0.000],\
       [-0.0004, -13.735],\
       [0.0002, -24.410],\
       [0.000, 0.000],\
       [0.000, 0.000]]

C_c = [[1,0,0,0,0],\
       [0,1,0,0,0],\
       [0,0,1,0,0],\
       [0,0,0,1,0],\
       [0,0,0,0,1]]
H = [[1,0,0,0,0],\
     [0,0,0,0,1]]

#Operating Point
X_o = [75.0 , 0.00 , 0.00 , 0.00 , 500.0]

#Update Model
S.updateContinuousStateModel(A_c,B_c,C_c,H,X_o)

#Initial inputs and desired setpoint
U_o = [0.25,-0.3]
R_set = [25,30]
S.updateInitialConditions(U_o,R_set)

dT = 0.05
S.discretize(dT)

#Q (on states) and R(on inputs) weight matrices
Q = [[1,0],[0,49]]
R = [[10,0],[0,15]]

#Update MPC Parameters
S.updateMPCParameters(Q,R)

#Inputs [1 2] maximum and minimum values (no incremental inputs for now)
U_max = [0.4,0.7]
U_min = [-0.4, 0]
#Inputs [1 2] incremental inputs (how much movement per unit sample time)
dU = [0.4,0.0104]

#Update constraints
S.updateConstraints(U_max,U_min,dU,m)

#---------------------------------------------------------------------------#
#MPC Design

#Prediction horizons for output 1 and 2
Np = [3.5, 2.0]

D = eMPC(Np,S.n,S.m,S.r)

#The MPC Internal Model
D.internalModel(S.A,S.B,S.C,S.H,S.Q,S.R,S.n,S.m,S.r)

#The constraint Model (MU<=g), gives M and g matrices
D.constraintModel(S.numABS,S.numInc,S.input_lim,S.m)

#Start a test optimization
flag = 0

#run 3 iterations of this-----------------------------------------------#
if flag == 0:
    for i in range(1,3):
        #Optimization Test run
        S.stepsim()
        S.Zpr = D.F @ np.asarray(S.X)
        S.E = np.reshape(S.R_set,(S.r,1)) - S.Zpr
        S.f = D.G.T @ S.Q @ S.E
        S.U = D.K_eMPC @ S.E
        e = constraintChecker(S.U,D.M_con,D.g_con)
        print(i)
        print('Unconstrained Input:',S.U)
        if e > 0:
            #Call the optimizer
            qp = HQP(D.M_con, D.Hinv, D.g_con,S.f)
            qp.Optimize()
            #Find constrained input
            S.U = -D.Hinv @ (S.f + np.reshape(0.5*D.M_con.T @ qp.lam,(S.m,1)))
            print('\nConstrained Input:',S.U, 'after num iterations:',qp.i_t)
            #Plot the result of the optimization
            x = list(range(0,qp.i_t))
            y = qp.err[:-1]
            title = 'Iteration ' + str(i)
            plt.figure(title)
            plt.plot(x,y)
        #Reset states and inputs
        S.X = S.XO
        #random input for next iteration (you can choose this)
        S.U = np.array([0.8,-0.1324])
    
#-----------------------------------------------------------------------#




