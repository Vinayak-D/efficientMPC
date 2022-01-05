#MPC Design class
import numpy as np
from scipy import linalg

class MPCDesign:
    def __init__(self,Np,n,m,r):
            self.N_p = Np
            self.F = np.zeros([r, n])
            self.G = np.zeros([r, m])
            
class eMPC(MPCDesign):
    def __init__(self,Np,n,m,r):
        super().__init__(Np,n,m,r)
        
    def internalModel(self,A,B,C,H,Q,R,n,m,r):
        [S,D] = linalg.eig(A)
        #Eigenvalues for diagonal matrix (np.diag(MID)), n = # of states
        MID = np.zeros(n, dtype=complex) 
        for i in range(r):
            for y in range(n):
                MID[y] = np.exp(S[y]*self.N_p[i])
            phi = D @ np.diag(MID) @ linalg.inv(D)
            self.F[i] = np.real(H[i] @ C @ phi)
            self.G[i] = np.real(H[i] @ C @ linalg.inv(A) @ (phi-np.identity(n)) @ B)
        H_fb = self.G.T @ Q @ self.G + R 
        self.Hinv = linalg.inv(H_fb)     
        self.K_eMPC = linalg.inv(self.G.T @ Q @ self.G + R) @ self.G.T @ Q
    
    def constraintModel(self,numABS,numInc,input_lim,m):
        self.g_con = np.zeros(numABS+2*numInc)
        self.M_con = np.zeros([len(self.g_con),m])
        
        #Assign right hand side g vector
        self.g_con[0:numABS] = np.hstack([np.array(input_lim['Max']),np.multiply(input_lim['Min'],-1)])
        if numInc>0:
            self.g_con[-2*numInc:] = np.tile(input_lim['Inc'],2)   
            self.g_con[-numInc:] = np.multiply(self.g_con[-numInc:],-1)
        
        #Assign left hand side M matrix
        __svec1 = np.sign(input_lim['Max'])
        for i in range(0,len(np.squeeze(__svec1))):
            if __svec1[0,i] == 0:
                __svec1[0,i] = 1    
        __svec2 = np.sign(input_lim['Min'])
        for i in range(0,len(np.squeeze(__svec2))):
            if __svec2[0,i] == 0:
                __svec2[0,i] = -1
        M_min = np.diag(np.squeeze(__svec1))
        M_max = np.diag(np.squeeze(__svec2))
        if numInc>0:
            M_inc = np.identity(numInc)
            M_inc2 = np.multiply(M_inc,-1)
            self.M_con = np.vstack((M_min,M_max,M_inc,M_inc2))
        else:
            self.M_con = np.vstack((M_min,M_max))
            
        
     
        
            