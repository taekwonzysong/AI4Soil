# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:17:31 2025

@author: Zheyu Jiang
"""
import numpy as np

def theta(psi, theta_r, theta_s, alpha, n):
    psi = np.asarray(psi)
    m = 1.0 - 1.0/n  # van Genuchten relation

    # For psi < 0
    psi_neg = (psi < 0)
    
    Se_unsat = (1.0 + (alpha * np.abs(psi[psi_neg]))**n)**(-m)
    theta_unsat = (theta_r 
                   + (theta_s - theta_r) * Se_unsat)

    # For psi >= 0, fully saturated
    theta_sat = theta_s * np.ones_like(psi[~psi_neg])
    theta_out = np.empty_like(psi)
    theta_out[psi_neg] = theta_unsat
    theta_out[~psi_neg] = theta_sat
    return theta_out


def calculate_K(psi, K_s, theta_r, theta_s, alpha, n, l=0.5):
    
    psi = np.asarray(psi)
    
    m = 1.0 - 1.0/n
    psi_neg = (psi < 0)
    
    
    Se_unsat = (1.0 + (alpha * np.abs(psi[psi_neg]))**n)**(-m)
    
    # Mualem's K(Se):   Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
    term = (1.0 - (1.0 - Se_unsat**(1.0/m))**m)**2
    K_unsat = K_s * Se_unsat**l * term
    
    # For psi >= 0
    K_sat = K_s * np.ones_like(psi[~psi_neg])

    K_out = np.empty_like(psi)
    K_out[psi_neg] = K_unsat
    K_out[~psi_neg] = K_sat
    return K_out

def calculate_K_prime(psi, K_s, theta_r, theta_s, alpha, n, l=0.5):
    psi = np.asarray(psi)
    m = 1.0 - 1.0 / n
    psi_neg = (psi < 0)
    
    # Unsaturated case
    abs_psi_neg = np.abs(psi[psi_neg])
    Se = (1.0 + (alpha * abs_psi_neg)**n)**(-m)
    
    dSe_dpsi = -m * n * alpha**n * (abs_psi_neg**(n - 1)) * Se**(1 + 1/m)
    dTerm_dSe = 2 * (1 - (1 - Se**(1/m))**m) * m * (1 - Se**(1/m))**(m - 1) * Se**(-1 + 1/m)
    
    dK_unsat_dpsi = K_s * (
        l * Se**(l - 1) * dSe_dpsi * (1 - (1 - Se**(1/m))**m)**2 +
        Se**l * dTerm_dSe * dSe_dpsi
    )
    
    # Saturated case
    dK_sat_dpsi = np.zeros_like(psi[~psi_neg])
    
    # Combine results
    dK_dpsi = np.empty_like(psi)
    dK_dpsi[psi_neg] = dK_unsat_dpsi
    dK_dpsi[~psi_neg] = dK_sat_dpsi
    return dK_dpsi
