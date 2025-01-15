# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:06:55 2023

@author: Zheyu Jiang
"""

import numpy as np
import torch
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from models import MLP1, MLP2, NN
from data_processing import load_data1, load_data2
from training import train_model
from correlations import theta, calculate_K, calculate_bc, calculate_K_prime
def main():
    start_time = time.time()
    
    # Load data
    x, y = load_data1("3-D Benchmark Problem/Data/reference_solutions_3-D.csv")
    v, w = load_data2("3-D Benchmark Problem/Data/reference_solutions_processed.csv")
    
    # Initialize models
    mlp1 = MLP1()
    mlp2 = MLP2()
    mlp3 = NN()

    # Define optimizers
    mlp_optimizer1 = torch.optim.SGD(mlp1.parameters(), lr=1e-3)
    mlp_optimizer2 = torch.optim.SGD(mlp2.parameters(), lr=1e-3)
    mlp_optimizer3 = torch.optim.SGD(mlp2.parameters(), lr=1e-2)
    
    # Train models
    epoch = 1000
    mlp_loss1 = train_model(mlp1, mlp_optimizer1, x, y, epoch)
    mlp_loss2 = train_model(mlp2, mlp_optimizer2, y, x, epoch)
    mlp_loss3 = train_model(mlp3, mlp_optimizer3, v, w, epoch)

    # Define parameters
    num_nodes_x=21
    num_nodes_y=21
    num_nodes_z=21
    initial_length=0
    end_length=2
    initial_width=0 
    end_width=2
    initial_depth=0 
    end_depth=2

    dx=(end_length-initial_length)/(num_nodes_x-1)
   
    dy=(end_width-initial_depth)/(num_nodes_y-1)
    
    dz=(end_depth-initial_depth)/(num_nodes_z-1)
    

    K_s = 1.1*24*3600
    theta_r=0
    theta_s=0.5
    alpha=0.1
    Total_time = 1
    iterations=500
    maxr=0.8
    scale=1e10
    tol=1e-5
    h_r=-15.24
    psi_0=h_r*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))
    g=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
    h=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
    for num1 in range(0,num_nodes_x-1):
        for num2 in range(0,num_nodes_z-1):
            psi_0[num1,num_nodes_y-1,num2]=calculate_bc(alpha,h_r,num2*dz,num1*dx,end_depth,end_length)
    psi=psi_0/10
    psi=np.reshape(psi,(num_nodes_x*num_nodes_y*num_nodes_z))
    psi=torch.FloatTensor(psi)
    psi=psi.unsqueeze(1)
    n0=mlp2(psi)
    n0=n0.squeeze(1)
    n0=n0.detach().numpy()
    n0=np.reshape(n0,(num_nodes_x,num_nodes_y,num_nodes_z))
    n0=scale*(10*n0)
    psi=psi*10
    psi=psi.squeeze(1)
    psi=psi.detach().numpy()
    psi=np.reshape(psi,(num_nodes_x,num_nodes_y,num_nodes_z))
    new_number_of_particles=n0
    n=n0
    t_num=1
    tol_iterations=np.zeros(iterations)
    soil_moisture_content=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
    r_x=np.zeros((num_nodes_x-1,num_nodes_y-2,num_nodes_z-2))
    r_y=np.zeros((num_nodes_x-2,num_nodes_y-1,num_nodes_z-2))
    r_z=np.zeros((num_nodes_x-2,num_nodes_y-2,num_nodes_z-1))
    for num1 in range(0,num_nodes_x):
        for num2 in range(0,num_nodes_y):
            for num3 in range(0,num_nodes_z):
                soil_moisture_content[num1,num2,num3]=theta(psi[num1,num2,num3],theta_r,theta_s,alpha)
    psi_current=psi
    soil_moisture_content0=soil_moisture_content
    L=0.5*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))
    L0=L
    current_time=0
    K=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
    while current_time<=Total_time:
        for iteration in range (0,iterations):
            OP=L
            for num1 in range(0,num_nodes_x):
                for num2 in range(0,num_nodes_y):
                    for num3 in range(0,num_nodes_z):
                        K[num1,num2,num3]=calculate_K(K_s,alpha,psi[num1,num2,num3])
            K_x=np.zeros((num_nodes_x-1,num_nodes_y-2,num_nodes_z-2))
            K_y=np.zeros((num_nodes_x-2,num_nodes_y-1,num_nodes_z-2))
            K_z=np.zeros((num_nodes_x-2,num_nodes_y-2,num_nodes_z-1))
            for num1 in range(0,num_nodes_x-1):
                 for num2 in range(0,num_nodes_y-2):
                     for num3 in range(0,num_nodes_z-2):
                         K_x[num1,num2,num3]=(K[num1,num2+1,num3+1]+K[num1+1,num2+1,num3+1])/2
            for num1 in range(0,num_nodes_x-2):
                 for num2 in range(0,num_nodes_y-1):
                     for num3 in range(0,num_nodes_z-2):            
                         K_y[num1,num2,num3]=(K[num1+1,num2,num3+1]+K[num1+1,num2+1,num3+1])/2
            for num1 in range(0,num_nodes_x-2):
                 for num2 in range(0,num_nodes_y-2):
                     for num3 in range(0,num_nodes_z-1):    
                         K_z[num1,num2,num3]=(K[num1+1,num2+1,num3]+K[num1+1,num2+1,num3+1])/2
            dt=(np.power(dx,2)*maxr)/(2*np.max(K_x))
            for num1 in range(0,num_nodes_x-1):
                for num2 in range(0,num_nodes_y-2):
                    for num3 in range(0,num_nodes_z-2):
                        r_x[num1,num2,num3]=dt*K_x[num1,num2,num3]/np.power(dx,2)/L[num1,num2,num3]
            for num1 in range(0,num_nodes_x-2):
                for num2 in range(0,num_nodes_y-1):
                    for num3 in range(0,num_nodes_z-2):
                        r_y[num1,num2,num3]=dt*K_y[num1,num2,num3]/np.power(dy,2)/L[num1,num2,num3]
            for num1 in range(0,num_nodes_x-2):
                for num2 in range(0,num_nodes_y-2):
                    for num3 in range(0,num_nodes_z-1):
                        r_z[num1,num2,num3]=dt*K_z[num1,num2,num3]/np.power(dz,2)/L[num1,num2,num3]

            residual_coefficients=np.ones((num_nodes_x-2,num_nodes_y-2,num_nodes_z-2))
            residual_coefficients=1-(r_x[0:num_nodes_x-2,:,:]+r_x[1:num_nodes_x-1,:,:]+r_y[:,0:num_nodes_y-2,:]+r_y[:,1:num_nodes_y-1,:]+r_z[:,:,0:num_nodes_z-2]+r_z[:,:,1:num_nodes_z-1])
        
        ########################################################################################    
        # Calculate RHS: n^(s+1) 
        ########################################################################################
        
        
            new_number_of_particles[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]=np.multiply(residual_coefficients,psi[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1])+\
               np.multiply(r_x[0:num_nodes_x-2,:,:],psi[0:num_nodes_x-2,1:num_nodes_y-1,1:num_nodes_z-1])+np.multiply(r_x[1:num_nodes_x-1,:,:],psi[2:num_nodes_x,1:num_nodes_y-1,1:num_nodes_z-1])+\
                np.multiply(r_y[:,0:num_nodes_y-2,:],psi[1:num_nodes_x-1,0:num_nodes_y-2,1:num_nodes_z-1]) +np.multiply(r_y[:,1:num_nodes_y-1,:],psi[1:num_nodes_x-1,2:num_nodes_y,1:num_nodes_z-1])+\
                np.multiply(r_z[:,:,0:num_nodes_z-2],psi[1:num_nodes_x-1,1:num_nodes_y-1,0:num_nodes_z-2])+np.multiply(r_z[:,:,1:num_nodes_z-1],psi[1:num_nodes_x-1,1:num_nodes_y-1,2:num_nodes_z])
            second_term_adjustment=new_number_of_particles
            second_term_adjustment[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]=np.multiply(r_x[0:num_nodes_x-2,:,:],psi[0:num_nodes_x-2,1:num_nodes_y-1,1:num_nodes_z-1])+np.multiply(r_x[1:num_nodes_x-1,:,:],psi[2:num_nodes_x,1:num_nodes_y-1,1:num_nodes_z-1])+\
                np.multiply(r_y[:,0:num_nodes_y-2,:],psi[1:num_nodes_x-1,0:num_nodes_y-2,1:num_nodes_z-1]) +np.multiply(r_y[:,1:num_nodes_y-1,:],psi[1:num_nodes_x-1,2:num_nodes_y,1:num_nodes_z-1])+\
                np.multiply(r_z[:,:,0:num_nodes_z-2],psi[1:num_nodes_x-1,1:num_nodes_y-1,0:num_nodes_z-2])+np.multiply(r_z[:,:,1:num_nodes_z-1],psi[1:num_nodes_x-1,1:num_nodes_y-1,2:num_nodes_z])
            second_term_adjustment=np.floor(second_term_adjustment)
            new_number_of_particles[0,:,:]=h_r*scale
            new_number_of_particles[num_nodes_x-1,:,:]=h_r*scale
            new_number_of_particles[:,0,:]=h_r*scale
            new_number_of_particles[:,:,0]=h_r*scale
            new_number_of_particles[:,:,num_nodes_z-1]=h_r*scale
            new_number_of_particles[:,num_nodes_y-1,:]=psi_0[:,num_nodes_y-1,:]*scale
            
            soil_moisture_content_diff=(soil_moisture_content0-soil_moisture_content)/L;
            third_term_init=(r_z[:,:,1:num_nodes_z-1]-r_z[:,:,0:num_nodes_z-2])*dz + soil_moisture_content_diff[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]
            
            third_term_init=np.reshape(third_term_init,(num_nodes_x-1*num_nodes_y-1*num_nodes_z-1))
            third_term_init=torch.FloatTensor(third_term_init)
            third_term_initi=third_term_init.unsqueeze(1)
            flux_residual=mlp3(third_term_init)
            flux_residual=flux_residual.squeeze(1)
            flux_residual=flux_residual.detach().numpy()
            flux_residual=np.reshape(flux_residual,(num_nodes_x-1,num_nodes_y-1,num_nodes_z-1))
            third_term_initi=third_term_initi.squeeze(1)
            third_term_initi=third_term_initi.detach().numpy()
            third_term_initi=np.reshape(third_term_initi,(num_nodes_x-1,num_nodes_y-1,num_nodes_z-1))
            
        ######################################################################################
        # Map n^(s+1) to psi^(s+1) by neural network
        ######################################################################################
        
            new_number_of_particles[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]=new_number_of_particles[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]+flux_residual
            n1=new_number_of_particles
            n=n1*1e-11
            n=np.reshape(n,(num_nodes_x*num_nodes_y*num_nodes_z))
            n=torch.FloatTensor(n)
            n=n.unsqueeze(1)
            psi=mlp1(n)
            psi=psi*10
            psi=psi.squeeze(1)
            psi=psi.detach().numpy()
            psi=np.reshape(psi,(num_nodes_x,num_nodes_y,num_nodes_z))
            n=n*1e11
            n=n.squeeze(1)
            n=n.detach().numpy()
            n=np.reshape(n,(num_nodes_x,num_nodes_y,num_nodes_z))
            #psi1=np.expand_dims(np.expand_dims(n, axis=0), axis=-1)
            #psi=model.predict(psi1)[0]
            tol_iteration=np.linalg.norm(psi-psi_current)/np.linalg.norm(psi)
        
        #######################################################################################    
        # Condition of adaptive L-scheme
        #######################################################################################
        
            if abs(calculate_K(K_s,alpha,psi_0[0,0,0])*2/L[0,0,0]-1/(abs((1/(1e-2*psi[0,0,num_nodes_z-2]))*g[0,0,num_nodes_z-2]))*(calculate_K(K_s,alpha,psi[0,0,num_nodes_z-2])+calculate_K(K_s,alpha,psi[0,0,num_nodes_z-3])+h[0,0,num_nodes_z-2]))>1e-2*dt/dz**2 and iteration>1:
                current_iteration=iteration
                
                #break
                # Calculate the L
                if iteration>=current_iteration:
                    for num in range(1,num_nodes_z-1):
                        L[0,0,num]=max(L0[0,0,num],abs((1/(1e-4*psi[0,0,num]))*g[0,0,num]))
                
            for num in range(1,num_nodes_z-1):
                g[0,0,num]=(-(r_z[0,0,num-1]+r_z[0,0,num])*n[0,0,num]+second_term_adjustment[0,0,num-1]+third_term_init[num-1])*OP[num]/scale/dt
                h[0,0,num]+=calculate_K_prime(K_s,alpha,psi[0,0,num])*g[0,0,num]+calculate_K_prime(K_s,alpha,psi[0,0,num-1])*g[0,0,num-1]
            if t_num*Total_time/3>=current_time and t_num*Total_time/3<current_time+dt and current_time<=Total_time: 
                tol_iterations[iteration]=tol_iteration
            if tol_iteration <= tol:
                break
            for num1 in range(0,num_nodes_x):
                for num2 in range(0,num_nodes_y):
                    for num3 in range(0,num_nodes_z):
                        soil_moisture_content[num1,num2,num3]=theta(psi[num1,num2,num3],theta_r,theta_s,alpha)
            current_time+=dt
            end_time = time.time()
            print("Total execution time:", end_time - start_time)
    
    if __name__ == "__main__":
        main()
