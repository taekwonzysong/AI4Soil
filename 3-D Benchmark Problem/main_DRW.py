# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:06:55 2023

@author: Zheyu Jiang
"""
import numpy as np
import torch
import time
import os
import pandas as pd
from models import MLP1, MLP2, NN
from data_processing import load_data1, load_data2
from training import train_model
from correlations import theta, calculate_K, calculate_bc, calculate_K_prime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def GRW_solver(L_value,num_nodes_x,num_nodes_y,num_nodes_z,iterations):
    # Define parameters
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
    maxr=0.8
    number_of_particles=1e10
    tol=1e-10
    h_r=-15.24
    psi_0=h_r*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))

    for num1 in range(0,num_nodes_x-1):
        for num2 in range(0,num_nodes_z-1):
            psi_0[num1,num_nodes_y-1,num2]=calculate_bc(alpha,h_r,num2*dz,num1*dx,end_depth,end_length)
    psi=psi_0/10
    n0=psi_0*number_of_particles
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
    L=L_value*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))
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
            new_number_of_particles[0,:,:]=h_r*number_of_particles
            new_number_of_particles[num_nodes_x-1,:,:]=h_r*number_of_particles
            new_number_of_particles[:,0,:]=h_r*number_of_particles
            new_number_of_particles[:,:,0]=h_r*number_of_particles
            new_number_of_particles[:,:,num_nodes_z-1]=h_r*number_of_particles
            new_number_of_particles[:,num_nodes_y-1,:]=psi_0[:,num_nodes_y-1,:]*number_of_particles
            
            soil_moisture_content_diff=(soil_moisture_content0-soil_moisture_content)/L
            third_term_init=(r_z[:,:,1:num_nodes_z-1]-r_z[:,:,0:num_nodes_z-2])*dz + soil_moisture_content_diff[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]
            flux_residual=third_term_init*number_of_particles
            
            ######################################################################################
            # Map n^(s+1) to psi^(s+1)
            ######################################################################################
        
            new_number_of_particles[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]=new_number_of_particles[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]+flux_residual
            psi=new_number_of_particles/number_of_particles
            #psi1=np.expand_dims(np.expand_dims(n, axis=0), axis=-1)
            #psi=model.predict(psi1)[0]
            tol_iteration=np.linalg.norm(psi-psi_current)/np.linalg.norm(psi)
        
            #######################################################################################    
            # Condition of adaptive L-scheme
            #######################################################################################
        
            if t_num*Total_time/3>=current_time and t_num*Total_time/3<current_time+dt and current_time<=Total_time: 
                tol_iterations[iteration]=tol_iteration
            if tol_iteration <= tol:
                break
            for num1 in range(0,num_nodes_x):
                for num2 in range(0,num_nodes_y):
                    for num3 in range(0,num_nodes_z):
                        soil_moisture_content[num1,num2,num3]=theta(psi[num1,num2,num3],theta_r,theta_s,alpha)
        current_time+=dt
    return psi, new_number_of_particles
def main():
    start_time = time.time()
    index = 1
    # Set index to 0 if the user wants to generate their own original reference solution for different parameter L and iteration counts:
    # Set index to 1 if the user wants to rerun the instance presented in the manuscript
    if index==0:   
        # Parameters for L and S
        L_values = [0.5,1]  # Example range of L
        S_values = [100,200,300,400,500] # Example range of S
        num_nodes_x=6
        num_nodes_y=6
        num_nodes_z=6
        # Gaussian noise parameters
        noise_mean = 0.0
        noise_std = 0.05

        # Collecting numerical solutions
        psi_solutions = []
        n_solutions = []
        psi_solutions_original = []
        n_solutions_original  = []

        for L in L_values:
            for S in S_values:
                print("Currently starting L = ", L, " and S = ", S)
                # Generate solution
                psi, n = GRW_solver(L,num_nodes_x,num_nodes_y,num_nodes_z,S)
                psi=psi.reshape(num_nodes_x*num_nodes_y*num_nodes_z,1)
                n=n.reshape(num_nodes_x*num_nodes_y*num_nodes_z,1)
                psi_solutions_original.append(psi)
                n_solutions_original.append(n)
                
                # Add Gaussian noise to both psi and n
                noisy_psi = psi + np.random.normal(noise_mean, noise_std, psi.shape)
                noisy_n = n + np.random.normal(noise_mean, noise_std, n.shape)
                
                # Append to the list
                psi_solutions.append(noisy_psi)
                n_solutions.append(noisy_n)
                
        # Convert to PyTorch tensors
        psi_tensor_original = torch.tensor(psi_solutions_original, dtype=torch.float32)
        n_tensor_original = torch.tensor(n_solutions_original, dtype=torch.float32)
        psi_tensor = torch.tensor(psi_solutions, dtype=torch.float32)
        n_tensor = torch.tensor(n_solutions, dtype=torch.float32)
        
        x, y=n_tensor*1e-11, psi_tensor*1e-1
        x = x.reshape(-1, 1)  # or x.view(-1, 1)
        y = y.reshape(-1, 1)  # or y.view(-1, 1)
        # Convert tensors to NumPy arrays
        psi_array = psi_tensor_original.numpy().flatten()  
        n_array = n_tensor_original.numpy().flatten()    
        # Save reference solutions to csv, only if you want to save them for later use, not needed for carrying on the calculations
        '''
        df = pd.DataFrame({
            "n": n_array,
            "psi": psi_array
        })
        
        # Save to a CSV file
        csv_filename = "reference_solutions_original.csv"
        df.to_csv(csv_filename, index=False)'''

        print("Begin rescaling...")
        # Rescaling data x, y to [-10,10] and [-10*10^10, 10*10^10], respectively
        # Rescaling is needed to compute f^{-1}(J) using NN trained on psi because 1) psi can only take negative values whereas J (in Equation 21) can be either positive or negative; 2) the range of psi is different from that of J
        new_min, new_max = -10, 10
        psi_min, psi_max = psi_tensor.min(), psi_tensor.max()
        v = (psi_tensor - psi_min) / (psi_max - psi_min) * (new_max - new_min) + new_min
        new_min2, new_max2 = -10e10, 10e10
        n_min, n_max = n_tensor.min(), n_tensor.max()
        w = (n_tensor - n_min) / (n_max - n_min) * (new_max2 - new_min2) + new_min2
        v = v.reshape(-1, 1)  # or v.view(-1, 1)
        w = w.reshape(-1, 1)  # or w.view(-1, 1)
        print("Rescaling complete.")
    else:
        # Load data
        print("Loading reference solution data...")
        x, y = load_data1("3-D Benchmark Problem/Data/reference_solutions_3-D.csv")
        df = pd.read_csv("3-D Benchmark Problem/Data/reference_solutions_3-D.csv")
        print("Loading reference solution data complete, begin rescaling...")
        # Rescaling data x, y to [-10,10] and [-10*10^10, 10*10^10], respectively
        # Rescaling is needed to compute f^{-1}(J) using NN trained on psi because 1) psi can only take negative values whereas J (in Equation 21) can be either positive or negative; 2) the range of psi is different from that of J
        new_min, new_max = -10, 10
        df['x'] = ((df.iloc[:, 0] - df.iloc[:, 0].min()) / (df.iloc[:, 0].max() - df.iloc[:, 0].min())) * (new_max - new_min) + new_min
        new_min2, new_max2 = -10e10, 10e10
        df['y'] = ((df.iloc[:, 1] - df.iloc[:, 1].min()) / (df.iloc[:, 1].max() - df.iloc[:, 1].min())) * (new_max2 - new_min2) + new_min2
        # Store rescaled x, y as tensor v, w
        v = torch.tensor(df['x'].values, dtype=torch.float32).view(-1, 1)
        w = torch.tensor(df['y'].values, dtype=torch.float32).view(-1, 1)
        print("Rescaling complete.")
    
    # Initialize models
    mlp1 = MLP1()
    mlp2 = MLP2()
    mlp3 = NN()

    # Define optimizers
    mlp_optimizer1 = torch.optim.SGD(mlp1.parameters(), lr=1e-3)
    mlp_optimizer2 = torch.optim.SGD(mlp2.parameters(), lr=1e-3)
    mlp_optimizer3 = torch.optim.SGD(mlp2.parameters(), lr=1e-2)
    
    print("NN training...")
    # Train models
    epoch = 1000
    mlp_loss1 = train_model(mlp1, mlp_optimizer1, x, y, epoch)
    mlp_loss2 = train_model(mlp2, mlp_optimizer2, y, x, epoch)
    mlp_loss3 = train_model(mlp3, mlp_optimizer3, v, w, epoch)
    print("NN training complete, begin solution process...")

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
        print("Solving for t =", current_time)
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
            
            soil_moisture_content_diff=(soil_moisture_content0-soil_moisture_content)/L
            third_term_init=(r_z[:,:,1:num_nodes_z-1]-r_z[:,:,0:num_nodes_z-2])*dz + soil_moisture_content_diff[1:num_nodes_x-1,1:num_nodes_y-1,1:num_nodes_z-1]
            
            third_term_init=np.reshape(third_term_init,(num_nodes_x-1*num_nodes_y-1*num_nodes_z-1))
            third_term_init=torch.FloatTensor(third_term_init)
            third_term_init=third_term_init.unsqueeze(1)
            flux_residual=mlp3(third_term_init)
            flux_residual=flux_residual.squeeze(1)
            flux_residual=flux_residual.detach().numpy()
            flux_residual=np.reshape(flux_residual,(num_nodes_x-2,num_nodes_y-2,num_nodes_z-2))
            third_term_init=third_term_init.squeeze(1)
            third_term_init=third_term_init.detach().numpy()
            third_term_init=np.reshape(third_term_init,(num_nodes_x-2,num_nodes_y-2,num_nodes_z-2))
            
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
                g[0,0,num]=(-(r_z[0,0,num-1]+r_z[0,0,num])*n[0,0,num]+second_term_adjustment[0,0,num-1]+third_term_init[0,0,num-1])*OP[0,0,num]/scale/dt
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
    print("Solution process complete. Total CPU time:", end_time - start_time)
    
if __name__ == "__main__":
    main()
