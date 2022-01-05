#System class
import numpy as np
import control as ct
from scipy import linalg

class System:
    #ID is a class variale
    ID = 0
    def __init__(self):
        
        #Update ID
        System.ID +=1
        
        #Default dimensions
        self.n = 0
        self.m = 0
        self.p = 0
        self.r = 0
        
        #Default state models (continuous and discrete)
        self.Ac = 1
        self.A = 1
        self.Bc = 1
        self.B = 1
        self.C = 1
        self.D = 1
        self.H = 1
        self.XO = 1
        
        #Default Initial Conditions
        self.U = 1
        self.R_set = 1
        self.model = 1
        self.E = 1
        self.Zpr = 1
        
        #Default Sampling time and states
        self.dT = 0
        self.X = 1
        self.Y = 1
        self.Z = 1
        
        #MPC Parameters
        self.MPCType = 0
        self.Q = 0
        self.R = 0
        self.input_lim = {}
        self.input_lim['Max'] = []
        self.input_lim['Min'] = []
        self.input_lim['Inc'] = []
        
    #Methods to update the system    
    def updateDims(self,n,m,p,r):
        self.n = n
        self.m = m
        self.p = p
        self.r = r
    
    def updateContinuousStateModel(self,A,B,C,H,XO):
        self.Ac = A
        self.Bc = B
        self.C = C
        self.H = H
        self.D = np.zeros((np.shape(self.C)[0],np.shape(self.Bc)[1]))
        self.XO = XO
        self.X = XO
        
    def updateInitialConditions(self,U,RSET):
        self.U = U
        self.R_set = RSET
        self.model = ct.ss(self.Ac,self.Bc,self.C,self.D)
        self.E = np.zeros(self.r)
        self.Zpr = np.zeros(self.r)
    
    def discretize(self,dT):
        self.dT = dT
        __cont = ct.ss(self.Ac,self.Bc,self.C,self.D)
        __disc = __cont.sample(dT,method = 'zoh')
        (self.A,self.B,self.C,self.D) = ct.ssdata(__disc)
    
    def stepsim(self):
        self.X = self.A @ np.reshape(self.X,(self.n,1)) + self.B @ np.reshape(self.U,(self.m,1)) 
        self.Y = self.C @ np.reshape(self.X,(self.n,1))
        self.Z = self.H @ self.Y
        
    #Methods to update constraints, and MPC type
    def updateMPCParameters(self,Q,R):
        self.Q = Q
        self.R = R
    
    def updateConstraints(self,U_max,U_min,dU,m):
        self.numABS = 0
        self.numInc = 0
        self.input_lim['Max'].append(U_max)
        self.input_lim['Min'].append(U_min)
        self.numABS = 2*m
        if dU is not None:
            self.numInc = m
            self.input_lim['Inc'].append(dU) 
    
        