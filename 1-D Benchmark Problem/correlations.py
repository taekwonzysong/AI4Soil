# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:31:28 2024

@author: Zheyu Jiang
"""
import numpy as np
def theta(psi,K_s,theta_r,theta_s,alpha,beta):
    return theta_r+alpha*(theta_s-theta_r)/(alpha+np.power(abs(psi), beta))
  
def calculate_K(psi,K_s,A,gama):
    return K_s*A/(A+np.power(abs(psi), gama))

def calculate_K_prime(psi,K_s,A,alpha):
    return K_s*A*alpha*(abs(psi)**(alpha-1))/(A+np.power(abs(psi), alpha))**2