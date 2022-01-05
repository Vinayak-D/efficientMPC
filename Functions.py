#Stability function
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size']=15

def findStability(A,B,K,F,dT):
    #Get the closed loop system
    if (np.size(F) == 0 and np.size(K) == 0)==True:
        Acl = A;
    elif np.size(F) == None:
        Acl = A-B@K
    else:
        Acl = A-B@K@F
    #Get open and closed loop eigenvalues
    r_o = [x.real for x in linalg.eig(A)[0]]
    i_o = [x.imag for x in linalg.eig(A)[0]]
    r_c = 0
    i_c = 0
    flag = 0
    if np.array_equal(A,Acl) == False:
        r_c = [x.real for x in linalg.eig(Acl)[0]]
        i_c = [x.imag for x in linalg.eig(Acl)[0]]
    if min(r_c)<0:
        flag = 1
    #Plot on unit circle (z-plane)
    plt.figure("Pole-Zero plot on z-plane")
    t = np.linspace(0,2*np.pi)
    plt.plot(np.cos(t),np.sin(t),'k',linestyle='dashed')
    #Open loop
    plt.scatter(r_o,i_o,s=155,color='b',marker='+',label='Open')
    #Closed Loop
    if type(r_c)!=int and r_c != 0:
        plt.scatter(r_c,i_c,s=155,color='r',marker='x',label='Closed')
    #Other
    plt.legend()
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.xlim(-1.15,1.15)
    plt.ylim(-1.15,1.15)
    plt.show()
    return flag
            
def constraintChecker(U,M,g):
    e = 0;
    for i in range (len(g)):
        LHS = M @ U
        RHS = g[i]
        if (LHS[i]>=RHS):
            e += 1
    return e