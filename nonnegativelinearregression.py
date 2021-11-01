# -*- coding: utf-8 -*-
"""
nonnegative linear regression 
"""

# Notes
# 1. Note we set min A_ij = 1. This can be done either by dividing A by min Aij or by just adding 1. 
# The first approach is kind of cheating because the scaling automatically makes the error small
# If min Aij is not 1, then init error can be LARGE 

#2. If all xj coordinates are updated in parallel, then comparable to scipy (update_v_parllel and update_X_paralle)

#3. If coordinatewise update, then scaling down by n is better than not scaling. 

#4. Our theoreitcal alg in experiment has occasional large jumps in error

#5. The init large error issue is fixed by running a full update step first

#6. x is very very very sparse

#7. TODO m << n case 
   
import numpy as np 
from numpy import array  
from numpy.linalg import norm 
import matplotlib.pyplot as plt 
import scipy
from scipy.optimize import nnls
from numpy.linalg import inv
import math
from keras.datasets import mnist # Import mnist data set.

#%%
def init_all(eps, n):
    xbar = np.zeros(n) 
    xkm = np.zeros(n)  
    x = np.zeros(n) 
    ktotal = np.int(np.ceil(1/np.sqrt(eps))) #note that this is just an approx ktotal
    ak = 1 
    Ak = 1
    A_sum_ai_xibar = 0 #A\sum_i ai bar(x_{i-1})
    v = np.zeros(n) #vector with same length as x, used to obtain x 
    return xbar, xkm, x, ktotal, ak, Ak, A_sum_ai_xibar, v

# (Aug 18) Add a scaling n in this code? 
def update_v(jk, pjk, A, y, v, ak): 
    v[jk]+= (ak/pjk)*((A.T).dot(y) - 1)[jk] #scaling the step (making it larger)  to "compensate" for seq vs par
    return v

def update_v_smallstep(jk, pjk, A, y, v, ak): 
    v[jk]+= (ak)*((A.T).dot(y) - 1)[jk] #scaling the step (making it larger)  to "compensate" for seq vs par
    return v

def update_v_parallel(A, y, v, ak): 
    v+= ak*((A.T).dot(y) - 1)
    return v

def compute_scaling(A):
    scaling_vector = -1/np.power(np.linalg.norm(A, axis = 0), 2)
    return scaling_vector

def update_x(v, jk, xkm, scaling): 
    x = xkm 
    x[jk]= min(np.max((scaling*v[jk], 0)),-scaling)
    return x      

def update_x_parallel(v, svec): 
    # x= np.max((svec*v, 0))
    x = np.where(svec*v>0, svec*v, 0)
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

# Oct 31: rescaling

def remove_col1(A,b):
    s=A.T@b
    B=A[:,s>0]
    s=s[s>0]
    A=B/s
    return A, B, s


#%%
if __name__ == '__main__': 
    eps = 0.00000001 
    #n = 717 # input dimension 
    #m = 60000 # Number of data
    
    #This is A is random positive and objective is max_x ||Ax||^2/2-1^T x case.
    #A = np.random.rand(200, 20) 
    
    #Mnist data example
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    #b = train_y
    #A=np.array(train_X.reshape(60000,28*28))
    
    # b can also be random and negative. m>>n.
    b=np.random.uniform(-0.3,1,10)
    A=np.random.rand(10,500)
    (A,B,s)=remove_col1(A,b)
    (m,n)=A.shape
    # Also need to scale b
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
                # update y 
        (y, A_sum_ai_xibar) = update_y(A_sum_ai_xibar, A, ak, Ak, xbar)
        y_norm[k]  = norm(y, 2)
        
        #if (k==1):
        #    v = update_v_parallel(A, y, v, ak)
        #    x = update_x_parallel(v, scaling_vector)
        #else: 
        # sample jk from multinomial distribution
        randomseed=np.random.multinomial(1, [1/n]*n)
        jk = np.min(np.where(randomseed==1))
        pjk = 1/n
#%%         #%% 

        #update v parallelly 
        #v = update_v_parallel(A, y, v, ak)
        # # update v 
        v = update_v(jk, pjk, A, y, v, ak)
        #v_norm[k]  = norm(v, 2)       
        
        # update x
        x = update_x(v, jk, xkm, scaling_vector[jk])

#%%         
        #update x parallelly 
        #x = update_x_parallel(v, scaling_vector)
        #x_norm[k]  = norm(x, 2)
#%%         
#%% 
        

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
        our_result[k] = norm(B.dot(xsol_temp/s)-b,2)**2
        #our_result[k] = norm(A.dot(xsol_temp),2)**2/2-sum(xsol_temp)
    
    xsol = (1/(Ak-ak))*xsum 

    print("our result is "+ str(our_result[ktotal-1]))
    plt.plot(range(ktotal), our_result, 'b') 
    
    S=scipy.optimize.nnls(B, b, maxiter=ktotal)
    S=scipy.optimize.nnls(B, b, maxiter=30) # This method does not require many iterations.
    alg2_result = S[1]**2
    print("their result is "+ str(alg2_result))
    print(norm(B@S[0]-b,2)**2)
    
    # Also need to plot comparing algorithm's result.
    
    # print(xsol)
    # print(ktotal)
    # norm(xsol@A.T,2)**2/2-sum(xsol)

   #  """
   # #  Competing algorithm
   # #  """
    
    # Some analysis
    print(A.shape)
    #print(s)
    print(norm(A.dot(xsol),2)**2-sum(xsol))
    print(sum(xsol==0))
    print(norm(A.dot(S[0]),2)**2-sum(S[0]))
    
    # print(ktotal)
    # norm(xsol@A.T,2)**2/2-sum(xsol)

   #  """
   # #  Competing algorithm
   # #  """
     
