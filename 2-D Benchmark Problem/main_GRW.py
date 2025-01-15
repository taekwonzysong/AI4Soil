# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import time
import os
from correlations import theta, calculate_K
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def main():
    start_time = time.time()
    num_nodes_x=51
    num_nodes_z=51
    initial_length=0
    end_length=1 
    initial_depth=0 
    end_depth=1 
    dx=(end_length-initial_length)/(num_nodes_x-1)
    #grid_x= np.arange(initial_length,end_length+dx,dx)
    dz=(end_depth-initial_depth)/(num_nodes_z-1)
    #grid_z= np.arange(initial_depth,end_depth+dz,dz)
    K_s = 2.89*1e-6
    theta_r=0.078
    theta_s=0.43
    alpha=3.6
    n_v=1.56
    Total_time = 1.26*1e4
    iterations= 500
    number_of_praticles=1e10
    tolerance=1e-5
    l=0.5
    psi_0=-10*np.ones((num_nodes_x,num_nodes_z))
    for num in range(23,28):
            psi_0[num,0]=0
    psi=psi_0
    n0=psi_0*number_of_praticles
    n=n0
    new_number_of_particles=n
    t_num=1
    soil_moisture_content=np.zeros((num_nodes_x,num_nodes_z))
    for num1 in range(0,num_nodes_x):
        for num2 in range(0,num_nodes_z):
                soil_moisture_content[num1,num2]=theta(psi[num1,num2],theta_r,theta_s,alpha,n_v)
    soil_moisture_content_0=soil_moisture_content
    L=0.5*np.ones((num_nodes_x,num_nodes_z))
    residual_coefficients=np.ones((num_nodes_x-2,num_nodes_z-2))
    r_x=np.zeros((num_nodes_x-1,num_nodes_z-2))
    r_z=np.zeros((num_nodes_x-2,num_nodes_z-1))
    K_x=np.zeros((num_nodes_x-1,num_nodes_z-2))
    K_z=np.zeros((num_nodes_x-2,num_nodes_z-1))
    K=np.zeros((num_nodes_x,num_nodes_z))
    tol_iterations=np.zeros(iterations)
    psi_current=psi
    current_time=0
    while current_time<=Total_time:
        dt=10 #dt=(np.power(dx,2)*maxr)/(2*np.max(Dx))
        current_time+=dt
        for iteration in range (0,iterations):
            for num1 in range(0,num_nodes_x):
                for num2 in range(0,num_nodes_z):
                    K[num1,num2]=calculate_K(psi[num1,num2],K_s,theta_r, theta_s,alpha,n_v,l)
            for num1 in range(0,num_nodes_x-1):
                 for num2 in range(0,num_nodes_z-2):
                    K_x[num1,num2]=(K[num1,num2+1]+K[num1+1,num2+1])/2
            for num1 in range(0,num_nodes_x-2):
                 for num2 in range(0,num_nodes_z-1):
                     K_z[num1,num2]=(K[num1+1,num2]+K[num1+1,num2+1])/2
            
            
            for num1 in range(0,num_nodes_x-1):
                for num2 in range(0,num_nodes_z-2):
                    r_x[num1,num2]=dt*K_x[num1,num2]/np.power(dx,2)/L[num1,num2]
            for num1 in range(0,num_nodes_x-2):
                for num2 in range(0,num_nodes_z-1):
                    r_z[num1,num2]=dt*K_z[num1,num2]/np.power(dz,2)/L[num1,num2]
        
            residual_coefficients=1-(r_x[0:num_nodes_x-2,:]+r_x[1:num_nodes_x-1,:]+r_z[:,0:num_nodes_z-2]+r_z[:,1:num_nodes_z-1])
            new_number_of_particles[1:num_nodes_x-1,1:num_nodes_z-1]=np.multiply(residual_coefficients,n[1:num_nodes_x-1,1:num_nodes_z-1])+\
               np.multiply(r_x[0:num_nodes_x-2,:],n[0:num_nodes_x-2,1:num_nodes_z-1])+np.multiply(r_x[1:num_nodes_x-1,:],n[2:num_nodes_x,1:num_nodes_z-1])+\
                np.multiply(r_z[:,0:num_nodes_z-2],n[1:num_nodes_x-1,0:num_nodes_z-2]) +np.multiply(r_z[:,1:num_nodes_z-1],n[1:num_nodes_x-1,2:num_nodes_z])
            new_number_of_particles[23:28,0]=0
            new_number_of_particles[0,:] = new_number_of_particles[1,:]
            new_number_of_particles[num_nodes_x-1,:] = new_number_of_particles[num_nodes_x-2,:]
            new_number_of_particles[0:23, 0]  = new_number_of_particles[0:23, 1]
            new_number_of_particles[28:,   0] = new_number_of_particles[28:,   1]
            new_number_of_particles[:,num_nodes_z-1] = new_number_of_particles[:,num_nodes_z-2]
            soil_moisture_content_diff=(soil_moisture_content_0-soil_moisture_content)/L
            third_term_init=(r_z[:,1:num_nodes_z-1]-r_z[:,0:num_nodes_z-2])*dz +soil_moisture_content_diff[1:num_nodes_x-1,1:num_nodes_z-1]
            flux_residual=number_of_praticles*third_term_init
            new_number_of_particles[1:num_nodes_x-1,1:num_nodes_z-1]=new_number_of_particles[1:num_nodes_x-1,1:num_nodes_z-1]+np.floor(flux_residual)
            psi=new_number_of_particles/number_of_praticles
            tol_iteration=np.linalg.norm(psi-psi_current)/np.linalg.norm(psi)
            if t_num*Total_time/3>=current_time and t_num*Total_time/3<current_time+dt and current_time<=Total_time: 
                tol_iterations[iteration]=tol_iteration
            if tol_iteration <= tolerance:
                break
            for num1 in range(0,num_nodes_x):
                for num2 in range(0,num_nodes_z):
                        soil_moisture_content[num1,num2]=theta(psi[num1,num2],theta_r,theta_s,alpha,n_v)
            psi_current=psi
            if  t_num*Total_time/3>=current_time and t_num*Total_time/3<current_time+dt and current_time<=Total_time:
                t_num+=1
                for num1 in range(0,num_nodes_x):
                    for num2 in range(0,num_nodes_z):
                            soil_moisture_content[num1,num2]=theta(psi[num1,num2],theta_r,theta_s,alpha,n_v)
        end_time = time.time()
        print("Total execution time:", end_time - start_time)
if __name__ == "__main__":
    main()
