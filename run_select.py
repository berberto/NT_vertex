# # All the function to run a simulation

#Import libraries

#########################################################################PARALELL################################################

#########################################################################PARALELL################################################
# %matplotlib tk 
#get_ipython().magic(u'matplotlib') #to use model.animate and see video alive
import itertools
import numpy as np
import matplotlib.pyplot as plt
#import vertex_model as model
import mesh as msh
#import vertex_model.initialisation as init
import initialisation as init
#import vertex_model.initialisationEd as init2
import initialisationEd as init2
#from vertex_model.forces import TargetArea, Tension, Perimeter, Pressure\\
import plotting as pltn
from forces import TargetArea, Tension, Perimeter, Pressure
from cells import *
import os
import copy 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #Don't show warnings
from Global_Constant import dt, viscosity, t_G1, t_G2, t_S, A_c, J, pos_d, T1_eps, P, microns, time_hours, expansion_constant #file with necessary constants

diff_rate_hours=0.1 #differentiation rate (1/h) 



def check_estage():
    print("Running a %s hours"%J + " %s"%pos_d_v)
    print("dt=%s"%dt)            #time step
    print("viscosity=%s" %viscosity)  #viscosity*dv/dt = F
    print("A_c=%s"%A_c) #critical area
    print("T1_eps =%s"%T1_eps)
    

# run simulation
def run(simulation,N_step,skip):
    return [cells.copy() for cells in itertools.islice(simulation,0,N_step,skip)]

def run_deep(simulation,N_step,skip):
    """
    deepcopy version of run.  Used for NT simulation. 
    """
    return [copy.deepcopy(cells) for cells in itertools.islice(simulation,0,N_step,skip)]
    



def division_axis(mesh,face_id,rand):
    """Choose a random division axis (given as a pair of boundary edges to bisect) for the given cell.
    
    The first edge is chosen randomly from the bounding edges of the cell with probability proportional 
    to edge length. The second edge is then fixed to be n_edge/2 from the first. 
    """
    edges = mesh.boundary(face_id) #was mesh.boundary
    if edges==[-1]: ##was if edges==[-1]
        print('here')
        os._exit(1)
    p = np.cumsum(mesh.length[edges])
    e0 = p.searchsorted(rand.rand()*p[-1])
    return edges[e0],edges[e0-len(edges)/2] 

def mod_division_axis(mesh,face_id):
    """Choose a random division axis (given as a pair of boundary edges to bisect) for the given cell.
    
    The first edge is chosen randomly from the bounding edges of the cell with probability proportional 
    to edge length. The second edge is then fixed to be n_edge/2 from the first. 
    """
    edges = mesh.boundary_liv(face_id) #list
    l = len(edges)
    rand_ind = np.random.random_integers(0, l-1) #index that will have a node as a midpt
    approx_opp_ind = (rand_ind + int(l/2)) % l # index of opposite edge
    return edges[rand_ind],edges[approx_opp_ind]  

def bin_by_xpos(cells,percentiles):
    vx = cells.mesh.vertices[0]
    #simple 'midpoint' as mean of vertex positions
    mid_x = np.bincount(cells.mesh.face_id_by_edge,weights=vx)
    counts = np.maximum(np.bincount(cells.mesh.face_id_by_edge),1.0)
    mid_x = mid_x / counts 
    width = cells.mesh.geometry.width
    return np.searchsorted(percentiles,(mid_x/width + 0.5) % 1.0)   
#simulation without division
def basic_simulation(cells,force,dt=dt,T1_eps=0.04):
    while True:
        cells.mesh , number_T1 = cells.mesh.transition(T1_eps)
        F = force(cells)/viscosity #not sure this works without defining 'A0' etc
        expansion = 0.05*np.average(F*cells.mesh.vertices,1)*dt
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) 
        cells.mesh = cells.mesh.moved(dv).scaled(1.0 + expansion)
        yield cells

# simulation with division and INM (no differentiation rate domain)
def simulation_with_division(cells,force,dt=dt,T1_eps=T1_eps,lifespan=100.0,rand=None): #(cells,force,dt=0.001,T1_eps=0.04,lifespan=100.0,rand=Non
    properties = cells.properties
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    expansion = np.array([0.0,0.0])
    while True:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
        if len(ready): #these are the cells ready to undergo division at the current timestep
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,2.0*len(ready))))
            properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
            properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
            properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2))  # Daugthers and parent have the same ids
            edge_pairs = [division_axis(cells.mesh,cell_id,rand) for cell_id in ready] #New edges after division 
            cells.mesh = cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)  #check edges verifing T1 transition
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F

        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells 
"""
simulation_with_division did not work because there was no initialisation of the 'age' property.
There was also no initialisation of 'parent_group'
Also, there seems to be a problem with division axis when rand=None as in the default. 
"""

     
def _modified_simulation_with_division(cells,force,dt=dt,T1_eps=T1_eps,lifespan=100.0,rand=None): #(cells,force,dt=0.001,T1_eps=0.04,lifespan=100.0,rand=Non
    properties = cells.properties
    properties['age']=np.random.normal(0.8,0.15,len(cells)) #random ages
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    #'parent_group' is not initialised.
    #properties['parent_group']
    expansion = np.array([0.0,0.0])
    while True:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
        if len(ready): #these are the cells ready to undergo division at the current timestep
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,2*len(ready)))) #error here.  Was 2.0*len(ready)
            properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
            properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
            #properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2))  # Daugthers and parent have the same ids
            edge_pairs = [mod_division_axis(cells.mesh,cell_id) for cell_id in ready] #New edges after division 
            cells.mesh = cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)  #check edges verifing T1 transition
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F

        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells         


# simulation with division, IKNM and no differentiation rate, follow clones
def simulation_with_division_clone(cells,force,dt=dt,T1_eps=T1_eps,lifespan=100.0,rand=None): #(cells,force,dt=0.001,T1_eps=0.04,lifespan=100.0,rand=Non
    properties = cells.properties
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    expansion = np.array([0.0,0.0])
    while True:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
        if len(ready): #these are the cells ready to undergo division at the current timestep
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,2.0*len(ready))))
            properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
            properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
            properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2)) #use to draw clones
            edge_pairs = [division_axis(cells.mesh,cell_id,rand) for cell_id in ready] #New edges after division 
            cells.mesh = cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)

        cells.mesh , number_T1= cells.mesh.transition(T1_eps)  #check edges verifing T1 transition
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F 
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells 


# simulation with division with INM and 2 diferent populations (with and without differentiation rate, pD and pMN) 
def simulation_with_division_clone_differenciation_3stripes(cells,force,dt=dt,T1_eps=T1_eps,lifespan=100.0,rand=None): #(cells,force,dt=0.001,T1_eps=0.04,lifespan=100.0,rand=Non
    # print T1_eps
    properties = cells.properties
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    properties['leaving'] = np.zeros(len(cells)) ### to add diferenciation rate in PMN
    # properties['differentiation_rate']= np.zeros(len(cells),dtype=int)
    expansion = np.array([0.0,0.0])
    while True:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
        if len(ready): #these are the cells ready to undergo division at the current timestep
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,2.0*len(ready))))
            properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
            properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
            properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2)) #use to draw clones
            properties['leaving'] = np.append(properties['leaving'], np.zeros(2*len(ready))) ### to add diferenciation rate in PMN
            edge_pairs = [division_axis(cells.mesh,cell_id,rand) for cell_id in ready] #New edges after division 
            cells.mesh = cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        ###### Defferentiation rate
        properties['differentiation_rate'] = 0.5*time_hours*dt*(np.array([0.0,diff_rate_hours,0.0]))[properties['parent_group']] #Used 0.02, 0.0002 & 1/13
        properties['leaving'] = properties['leaving'] - (properties['leaving']-1) * (~(cells.empty()) & (rand.rand(len(cells)) < properties['differentiation_rate']))
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        
        N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)*(1.0-cells.properties['leaving'])
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells 

# simulation with division with INM and all the cells with differentiation rate
def simulation_with_division_clone_whole_tissue_differenciation(cells,force,dt=dt,T1_eps=T1_eps,lifespan=100.0,rand=None): #(cells,force,dt=0.001,T1_eps=0.04,lifespan=100.0,rand=None
    properties = cells.properties
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    properties['leaving'] = np.zeros(len(cells)) ### to add diferenciation rate in PMN

    expansion = np.array([0.0,0.0])
    while True:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
        if len(ready): #these are the cells ready to undergo division at the current timestep
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,2.0*len(ready))))
            properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
            properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
            properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2)) #use to draw clones
            properties['leaving'] = np.append(properties['leaving'], np.zeros(2*len(ready))) ### to add diferenciation rate in PMN
            edge_pairs = [division_axis(cells.mesh,cell_id,rand) for cell_id in ready] #New edges after division 
            cells.mesh = cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        ###### Defferentiation rate
        properties['differentiation_rate'] = 0.5*time_hours*dt*(np.array([diff_rate_hours,diff_rate_hours,diff_rate_hours]))[properties['parent_group']] #Used 0.02, 0.0002 & 1/13
        properties['leaving'] = properties['leaving'] - (properties['leaving']-1) * (~(cells.empty()) & (rand.rand(len(cells)) < properties['differentiation_rate']))
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        
        N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)*(1.0-cells.properties['leaving'])
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*msh.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F
    
        
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells 


#_clones
def definecolors(cells):
    peach = '#eed5b7'
    light_blue ='#87cefa'
    pink = '#ffc0cb'
    light_green = '#98fb98'
    import matplotlib.colors as colors
    vv=sns.color_palette("hls", 10)
    v=[colors.rgb2hex(colorrgb) for colorrgb in vv]
    palette = np.array([light_green, pink,light_green,'g','r','g','m','c','',peach])
    palette = np.array([v[1],v[0],v[1], v[1],v[4],v[5],v[6],v[7],v[8],v[9],peach])
    colors = cells.properties['parent_group']
    return palette[colors]

"""Run simulation and save data functions"""
def run_simulation(x):
    K=x[0]
    G=x[1]
    L=x[2]
    P=0.5 #added by G.
    rand = np.random #np.random.RandomState(123456) #I have modified the random function because RamdomState takes always the same numbers
    mesh = init.cylindrical_hex_mesh(6,6,noise=0.2,rand=rand)
    cells = Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = np.random.rand(len(cells))
    force = TargetArea() + Tension() + Perimeter() + Pressure()
    history = run(simulation_with_division(cells,force,rand=rand),500.0/dt,1.0/dt)
    # model.animate_video_mpg(history)
    return history

def _modified_run_simulation(x, N_cell_across, N_cell_up):
    '''
    force requires some parameters
    '''
    K=x[0]
    G=x[1]
    L=x[2]
    P=x[3]
    rand = np.random #np.random.RandomState(123456) #I have modified the random function because RamdomState takes always the same numbers
    dummy=init.hexagonal_centres(N_cell_across,N_cell_up,0.2,rand)
    mesh = init2._modified_toroidal_voronoi_mesh(*dummy)
    cells = Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = np.random.rand(len(cells))
    force = TargetArea() + Tension() + Perimeter() + Pressure() 
    history = run(_modified_simulation_with_division(cells,force,rand=rand),int(1.0/dt), int(1.0/dt))
    _modified_animate_video_mpg(history)
    return history


def run_simulation_INM(x, timend,rand, sim_type):
    global dt
    #sim_type 0 simulation_with_division_clone (no differentiation rate)
    #sim_type 1 simulation_with_division_clone_differentiation (all differentiation rate)
    #sim_type 2 simulation_with_division_clone_differenciation_3stripes (2 population with and without diffentiation rate)
    K=x[0]
    G=x[1]
    L=x[2]
    mesh = init.toroidal_hex_mesh(10,10,noise=0.2,rand=rand)
    cells = pltn.Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = np.random.rand(len(cells))
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int)
    force = TargetArea() + Tension() + Perimeter() + Pressure()
    history1 = run(simulation_with_division(cells,force,rand=rand),200/dt,1.0/dt)
    cells = history1[-1].copy()
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int) #use to draw clone
    cells.properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    cells.properties['parent_group'] = bin_by_xpos(cells,np.cumsum([0.35,0.3,0.35]))
    if sim_type == 0:
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)
        history = run(simulation_with_division_clone(cells,force,rand=rand),(timend)/dt,1.0/dt)
        history[-1].properties['parent_group'] = np.zeros(len(history[-1].properties['parent_group']),dtype=int)
    if sim_type == 1:
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)+1
        history = run(simulation_with_division_clone_whole_tissue_differenciation(cells,force,rand=rand),(timend)/dt,1.0/dt)
        history[-1].properties['parent_group'] = np.zeros(len(history[-1].properties['parent_group']),dtype=int)+1
    if sim_type == 2:
        #we take ventral and dorsal time per phase cell cycle if we are in the 2 pop part, because pNM are ventral and pD are dorsal
        history = run(simulation_with_division_clone_differenciation_3stripes(cells,force,rand=rand),(timend)/dt,1.0/dt)
        cells.properties['parent_group'] = cells.properties['parent_group']
    return history1, history

def _modified_run_simulation_INM(x, timend,rand, sim_type, N_cell_across=10,N_cell_up=10):
    global dt
    #sim_type 0 simulation_with_division_clone (no differentiation rate)
    #sim_type 1 simulation_with_division_clone_differentiation (all differentiation rate)
    #sim_type 2 simulation_with_division_clone_differenciation_3stripes (2 population with and without diffentiation rate)
    K=x[0]
    G=x[1]
    L=x[2]
    dummy=init.hexagonal_centres(N_cell_across,N_cell_up,0.2,rand)
    mesh = init2._modified_toroidal_voronoi_mesh(*dummy)
    cells = Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = np.random.rand(len(cells))
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int)
    force = TargetArea() + Tension() + Perimeter() + Pressure()
    history1 = run(simulation_with_division(cells,force,rand=rand),200/dt,1.0/dt)
    cells = history1[-1].copy()
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int) #use to draw clone
    cells.properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    cells.properties['parent_group'] = bin_by_xpos(cells,np.cumsum([0.35,0.3,0.35]))
    if sim_type == 0:
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)
        history = run(simulation_with_division_clone(cells,force,rand=rand),(timend)/dt,1.0/dt)
        history[-1].properties['parent_group'] = np.zeros(len(history[-1].properties['parent_group']),dtype=int)
    if sim_type == 1:
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)+1
        history = run(simulation_with_division_clone_whole_tissue_differenciation(cells,force,rand=rand),(timend)/dt,1.0/dt)
        history[-1].properties['parent_group'] = np.zeros(len(history[-1].properties['parent_group']),dtype=int)+1
    if sim_type == 2:
        #we take ventral and dorsal time per phase cell cycle if we are in the 2 pop part, because pNM are ventral and pD are dorsal
        history = run(simulation_with_division_clone_differenciation_3stripes(cells,force,rand=rand),(timend)/dt,1.0/dt)
        cells.properties['parent_group'] = cells.properties['parent_group']
    return history1, history

def _cut_modified_run_simulation_INM(x, timend,rand, sim_type, N_cell_across=10,N_cell_up=10):
    global dt
    #sim_type 0 simulation_with_division_clone (no differentiation rate)
    #sim_type 1 simulation_with_division_clone_differentiation (all differentiation rate)
    #sim_type 2 simulation_with_division_clone_differenciation_3stripes (2 population with and without diffentiation rate)
    K=x[0]
    G=x[1]
    L=x[2]
    dummy=init.hexagonal_centres(N_cell_across,N_cell_up,0.2,rand)
    mesh = init2._modified_toroidal_voronoi_mesh(*dummy)
    cells = Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = np.random.rand(len(cells))
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int)
    force = TargetArea() + Tension() + Perimeter() + Pressure()
    history= run(simulation_with_division(cells,force,rand=rand),200/dt,1.0/dt)
    return history