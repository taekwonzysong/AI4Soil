# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import time
import os
from correlations import theta, calculate_K, calculate_bc, calculate_K_prime
start_time = time.time()

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
number_of_particles=1e10
tol=1e-5
h_r=-15.24
psi_0=h_r*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))
g=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
h=np.zeros((num_nodes_x,num_nodes_y,num_nodes_z))
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
L=0.5*np.ones((num_nodes_x,num_nodes_y,num_nodes_z))
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
        
        soil_moisture_content_diff=(soil_moisture_content0-soil_moisture_content)/L;
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
        end_time = time.time()
        print("Total execution time:", end_time - start_time)
