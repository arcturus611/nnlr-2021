# -*- coding: utf-8 -*-
"""
nonnegative linear regression 
"""
   
import numpy as np 
    
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

def update_v(jk, pjk, A, y, v):
    v[jk]+= (ak/pjk)*((A.T).dot(y) - 1)[jk]
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
    y = np.where(y_temp>0, y_temp, 0)
    return y, A_sum_ai_xibar

if __name__ == '__main__': 
    eps = 0.001 
    n = 5 # input dimension 
    m = 7   
    A = np.random.rand(m, n)
    xsum = 0 

    (xbar, xkm, x, ktotal, ak, Ak, A_sum_ai_xibar, v) = init_all(eps, n)
    scaling_vector = compute_scaling(A)
    
    for k in range(ktotal): 
        # sample jk 
        jk = np.random.randint(0, n)
        pjk = 1/n
        
        # update y 
        (y, A_sum_ai_xibar) = update_y(A_sum_ai_xibar, A, ak, Ak, xbar)
        
        # update v 
        v = update_v(jk, pjk, A, y, v)
        
        # update x
        x = update_x(v, jk, xkm, scaling_vector[jk])
        
        #update a 
        (Ak, akp) = update_ak_Ak(Ak, ak) 
        
        #update xbar 
        xbar = update_xbar(x, xkm, ak, akp) 
        
        #compute running sum 
        xsum+= ak*x
    
    xsol = (1/Ak)*xsum 