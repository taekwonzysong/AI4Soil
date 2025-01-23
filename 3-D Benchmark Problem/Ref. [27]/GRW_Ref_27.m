clear
num_nodes_x=21;
num_nodes_y=21;
num_nodes_z=21;
initial_length=0;
end_length=2;
initial_width=0; 
end_width=2;
initial_depth=0;
end_depth=2;
dx=(end_length-initial_length)/(num_nodes_x-1);
dy=(end_width-initial_depth)/(num_nodes_y-1);
dz=(end_depth-initial_depth)/(num_nodes_z-1);
grid_x=initial_length:dx:end_length;
grid_y=initial_width:dy:end_width;
grid_z=initial_depth:dy:end_depth;
K_s = 0.0496;
theta_r=0.131;
theta_s=0.396;
alpha=0.423;
n_model=2.06;
Total_time = 3/16; 
past=Total_time/3;
iterations= 500; 
dt=1/48;
t1=Total_time/3;      
Tolerance = 1e-5;
number_of_particles=10^10;
theta =  @(psi)  theta_GM(theta_r,theta_s,psi,alpha,n_model,(n_model-1)/n_model);
calculate_K = @(theta) K_s.*((theta - theta_r)./(theta_s-theta_r)).^0.5.*(1-(1-((theta - theta_r)./(theta_s-theta_r)).^(1/(n_model-1)/n_model)).^((n_model-1)/n_model)).^2;  
ini = 1 - ((grid_z'-initial_width)/(end_width-initial_width))*3+0*grid_x+0*grid_y;
for i=1:21
    psi_0(:,:,i) = ini;
end
i0=1/dx; j0=1/dy; k0=1/dz;
psi_0(21,1:i0,1:j0)=-2; 
n0=number_of_particles*(reshape(psi_0,1,9261));
n0=reshape(n0,21,21,21);
psi = psi_0;
n=n0;
kt=1;
new_number_of_particles=zeros(21,21,21);
soil_moisture_content = theta(psi);
soil_moisture_content0=soil_moisture_content;
L=0.5;  
pa=psi;
current_time=0;
tol_iterations=zeros(1,iterations);
while current_time<=Total_time
    current_time=current_time+dt; 
    for iteration=1:iterations
        K1=calculate_K(soil_moisture_content);
        K_x=(K1(2:20,1:20,1:20)+K1(2:20,2:21,2:21))/2;
        K_y=(K1(1:20,1:20,2:20)+K1(2:21,2:21,2:20))/2;
        K_z=(K1(1:20,2:20,1:20)+K1(2:21,2:20,2:21))/2;
        r_x=dt*K_x/(dx^2*L); 
        r_y=dt*K_y/(dy^2*L);
        r_z=dt*K_z/(dz^2*L);
        residual_coefficients=1-(r_x(:,1:19,1:19)+r_x(:,2:20,2:20)+r_y(1:19,1:19,:)+r_y(2:20,2:20,:)+r_z(1:19,:,1:19)+r_z(2:20,:,2:20));
        new_number_of_particles(2:20,2:20,2:20)=residual_coefficients.*n(2:20,2:20,2:20) ...
            +r_x(:,1:19,1:19).*n(2:20,1:19,1:19)+r_x(:,2:20,2:20).*n(2:20,3:21,3:21) ...
            +r_y(1:19,1:19,:).*n(1:19,1:19,2:20) +r_y(2:20,2:20,:).*n(3:21,3:21,2:20)...
            +r_z(1:19,:,1:19).*n(1:19,2:20,1:19) +r_z(2:20,:,2:20).*n(3:21,2:20,3:21);
       
        new_number_of_particles(:,:,1)=new_number_of_particles(:,:,2); 
        new_number_of_particles(1:j0,:,21)=psi_0(1:j0,:,21); 
        new_number_of_particles(1:j0,21,:)=psi_0(1:j0,21,:); 
        new_number_of_particles(j0+1:21,21,21)=new_number_of_particles(j0+1:21,21,20);
        new_number_of_particles(j0+1:21,21,21)=new_number_of_particles(j0+1:21,20,21);
        new_number_of_particles(1,:,2:20)=new_number_of_particles(2,:,2:20)+number_of_particles*(dz); 
        new_number_of_particles(:,1,2:20)=new_number_of_particles(:,2,2:20)+number_of_particles*(dz);
        if current_time<=t1 
            new_number_of_particles(21,:,1:i0)=n0(21,:,1:i0)+number_of_particles*(2.2*current_time/t1);
            new_number_of_particles(21,1:i0,:)=n0(21,1:i0,:)+number_of_particles*(2.2*current_time/t1);
        else
            new_number_of_particles(21,:,1:i0)=number_of_particles*(0.2);
        end
        new_number_of_particles(21,:,i0+1:21-1)=new_number_of_particles(20,:,i0+1:20)-number_of_particles*(dy); 
        new_number_of_particles(21,i0+1:21-1,:)=new_number_of_particles(20,i0+1:20,:)-number_of_particles*(dy);% no flux
        soil_moisture_content_diff=(soil_moisture_content0-soil_moisture_content)/L;
        third_term_init=(r_y(2:20,2:20,:)-r_y(1:19,1:19,:))*dy + soil_moisture_content_diff(2:20,2:20,2:20);
        flux_residual=number_of_particles*(reshape(third_term_init,1,6859));
        flux_residual=reshape(flux_residual,19,19,19);
        new_number_of_particles(2:20,2:20,2:20)=new_number_of_particles(2:20,2:20,2:20)+flux_residual;
        pp1=reshape(new_number_of_particles,1,9261);
        psi=reshape(1/number_of_particles*(pp1),21,21,21);
        tol_iteration=dx*norm(psi-pa,"fro")+norm(psi-pa,"fro")/norm(psi,"fro");
        if kt*past>=current_time && kt*past<current_time+dt && current_time<=Total_time
            tol_iterations(iteration)=tol_iteration;
        end 
        if tol_iteration <= Tolerance
            break
        end        
        soil_moisture_content = theta(psi);
        pa=psi;
    end
end
p1=psi(:,:,17);
soil_moisture_content1=soil_moisture_content(:,:,17);
pp1=reshape(new_number_of_particles,1,9261);
p21=reshape(psi,1,9261);
% Plot the 3D mesh of p1
figure;
mesh(grid_x, grid_y, p1); % Create a 3D mesh plot
xlabel('$x$', 'Interpreter', 'latex'); % Label for x-axis
ylabel('$z$', 'Interpreter', 'latex'); % Label for z-axis
zlabel('$\psi(x,z,t)$', 'Interpreter', 'latex'); % Label for y-axis
view(115, 15); % Set view angle (azimuth: 115, elevation: 15)
grid on; % Enable grid

% Plot the contour of soil moisture content
figure;
contourf(grid_x, grid_y, soil_moisture_content1, 12); % Filled contour plot with 12 levels
colormap(flipud(parula)); % Use reversed parula colormap
colorbar; % Add a colorbar
xlabel('$x$', 'Interpreter', 'latex'); % Label for x-axis
ylabel('$z$', 'Interpreter', 'latex'); % Label for z-axis
title('$\theta(x,z,t)$', 'Interpreter', 'latex'); % Title of the plot



