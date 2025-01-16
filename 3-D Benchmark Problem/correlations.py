"""
Created on Mon Mar 20 10:06:55 2023

@author: Zheyu Jiang
"""
import numpy as np
import math

def theta(psi,theta_r,theta_s,alpha):
        return theta_r+(theta_s-theta_r)*np.exp(alpha*psi)
def calculate_bc(alpha,hr,x,y,a,b):
    part=np.exp(alpha*hr)+(1-np.exp(alpha*hr))*np.sin(math.pi*x/a)*np.sin(math.pi*y/b)
    return 1/alpha*np.log(part)
def calculate_K(K_s,alpha,psi):
    return  K_s*np.exp(alpha*psi)
def calculate_K_prime(K_s,alpha,psi):
    return  K_s*alpha*np.exp(alpha*psi)

