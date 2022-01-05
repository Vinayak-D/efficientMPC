#Quadratic Programming class
import numpy as np
import copy

class QuadProg:
    def __init__(self,M_con,Hinv,g_con,f):
        self.W = (M_con @ Hinv @ M_con.T)*0.5
        self.Z = np.reshape(M_con @ Hinv @ f,(np.shape(g_con)[0],)) + g_con
        self.r = np.shape(self.Z)[0]
        self.i_t = 0
        self.err = np.empty(1)
        self.err[self.i_t] = 1
        self.lam = np.zeros(self.r)
    
class HQP(QuadProg):
    def __init__(self,M_con,Hinv,g_con,f):
        super().__init__(M_con,Hinv,g_con,f)
    
    def Optimize(self):
        while (self.err[self.i_t]>=1e-6):        
            old = copy.deepcopy(self.lam)
            for i in range(self.r):
                w = (self.W[i] @ self.lam) - self.W[i,i]*self.lam[i] + self.Z[i]
                la = -w/self.W[i,i]
                self.lam[i] = max(0,la)
            err = np.dot((self.lam-old).T,(self.lam-old))
            self.i_t+=1
            self.err = np.append(self.err,err)

class PQP(QuadProg):
    def __init__(self,M_con,Hinv,g_con,f):
        super().__init__(M_con,Hinv,g_con,f)
        self.lam = np.ones(self.r)  
        self.Zm = np.maximum(-self.Z,0)
        self.Zp = np.maximum(self.Z,0)
        r_vec = np.maximum(-self.W,0) @ np.zeros(self.r)
        self.Wm = np.maximum(-self.W,0) + np.diag(r_vec)
        self.Wp = np.maximum(self.W,0) + np.diag(r_vec)

    def Optimize(self):
        while (self.err[self.i_t]>=1e-6):
            old = copy.deepcopy(self.lam)
            for i in range(self.r):
                Km = self.Wm @ self.lams
                Kp = self.Wp @ self.lam     
                self.lam[i] = self.lam[i]*((self.Zm[i]+Km[i])/(self.Zp[i]+Kp[i]))
            err = np.dot((self.lam-old).T,(self.lam-old))
            self.i_t+=1
            err = np.append(self.err,err)      
        
        
        