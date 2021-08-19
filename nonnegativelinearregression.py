# -*- coding: utf-8 -*-
"""
nonnegative linear regression 
"""
   
import numpy as np 
from numpy import array  
from numpy.linalg import norm 
import matplotlib.pyplot as plt 
import scipy
from scipy.optimize import nnls
from numpy.linalg import inv
import math

#%%
def init_all(eps, n):
    xbar = np.zeros(n) 
    xkm = np.zeros(n)  
    x = np.zeros(n) 
    ktotal = np.int(np.ceil(1/np.sqrt(eps))) 
    ak = 1 
    Ak = 1
    A_sum_ai_xibar = 0 
    v = np.zeros(n)
    return xbar, xkm, x, ktotal, ak, Ak, A_sum_ai_xibar, v

# Add a scaling n in this code
def update_v(jk, pjk, A, y, v, ak): 
    v[jk]+= (ak/n*pjk)*((A.T).dot(y) - 1)[jk]
    return v

def compute_scaling(A):
    scaling_vector = -1/np.power(np.linalg.norm(A, axis = 0), 2)
    return scaling_vector

def update_x(v, jk, xkm, scaling): 
    x = xkm 
    x[jk]= np.max((scaling*v[jk], 0))
    return x      

def update_xbar(x, xkm, ak, akp):
    xbar = x + (ak/akp)*(x - xkm)
    return xbar 
    
def update_ak_Ak(Ak, ak):
    akp = .5* (1 + np.sqrt(1 + 4*Ak))
    Ak = Ak + akp 
    return Ak, akp 
    
def update_y(A_sum_ai_xibar, A, ak, Ak, xbar):
    A_sum_ai_xibar+= A.dot(ak*xbar) #A \sum_{i = 1}^{k} ai xbar_{i-1}
    y_temp = (1/Ak)*A_sum_ai_xibar 
    y = np.maximum(0, y_temp) #np.where(y_temp>0, y_temp, 0)
    return y, A_sum_ai_xibar

#%%
if __name__ == '__main__': 
    eps = 0.000001 
    n = 100 # input dimension 
    m = 500   
    A = np.random.rand(m, n)
    xsum = 0 

#%%
    (xbar, xkm, x, ktotal, ak, Ak, A_sum_ai_xibar, v) = init_all(eps, n)
    scaling_vector = compute_scaling(A)
    our_result = np.zeros(ktotal)
    x_norm = np.zeros(ktotal)
    y_norm = np.zeros(ktotal)
    v_norm = np.zeros(ktotal)
    xsum_norm = np.zeros(ktotal)
    # Xmatrix = np.zeros((n, ktotal))
    Akarray = np.zeros(ktotal)
    # ktotal = 1
    Akarray[0]=Ak
#%%         
    for k in range(ktotal): 
        # sample jk from multinomial distribution
        randomseed=np.random.multinomial(1, [1/n]*n)
        jk = np.min(np.where(randomseed==1))
        pjk = 1/n
#%%         

        # update y 
        (y, A_sum_ai_xibar) = update_y(A_sum_ai_xibar, A, ak, Ak, xbar)
        y_norm[k]  = norm(y, 2)
        # update v 
        v = update_v(jk, pjk, A, y, v, ak)
        v_norm[k]  = norm(v, 2)
        # update x
        x = update_x(v, jk, xkm, scaling_vector[jk])
        x_norm[k]  = norm(x, 2)
#%%         
        
        #update a 
        (Ak, akp) = update_ak_Ak(Ak, ak) 
        Akarray[k] = Ak
#%%         
        
        #update xbar 
        xbar = update_xbar(x, xkm, ak, akp) 
#%%      
        # update xkm 
        xkm = x 
#%%                        
        #compute running sum 
        xsum+= ak*x
        xsum_norm[k] = norm(xsum, 2)
        # update ak  
        ak = akp 
        
        xsol_temp = (1/(Ak-ak))*xsum 
        our_result[k] = norm(A.dot(xsol_temp),2)**2/2-sum(xsol_temp)
    
    xsol = (1/(Ak-ak))*xsum 

    print("our result is "+ str(our_result[ktotal-1]))
    plt.plot(range(ktotal), our_result, 'b', range(ktotal), x_norm, 'r') 
    S=scipy.optimize.nnls(A/math.sqrt(2), A@inv(A.T@A)@np.ones(n)/math.sqrt(2), maxiter=ktotal)
    alg2_result = S[1]**2-norm(A@inv(A.T@A)@np.ones(n)/math.sqrt(2),2)**2
    print("their result is "+ str(alg2_result))

    # print(xsol)
    # print(ktotal)
    # norm(xsol@A.T,2)**2/2-sum(xsol)

   #  """
   # #  Competing algorithm
   # #  """
     
