# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:08:49 2024

@author: Zheyu Jiang
"""
import numpy as np
def spetral_raduis(A):
    en_value,en_vec=np.linalg.eig(A)
    return np.max(np.abs(en_value))
def C(psi,K_s,theta_r,theta_s,alpha,beta):
    return alpha*(theta_s-theta_r)*beta*(-psi)**(beta-1)/(alpha+(-psi)**beta)**2
def condition_number(A):
    U, sigma, Vt = np.linalg.svd(A)
    return np.max(sigma)/np.min(sigma)