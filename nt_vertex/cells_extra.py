#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:38:32 2020

@author: andrewg
"""

from .mesh import sum_vertices
import numpy as np
from .constants import (expansion_constant,
                       t_G1, t_G2, t_S, time_hours,
                       diff_rate_hours,
                       T1_eps, viscosity, A_c)
from .cells import Cells
from .forces import (TargetArea, Pressure, Perimeter, Tension)
import copy
from .initialisationEd import (_modified_toroidal_hex_mesh,
                              _modified_toroidal_random_mesh,
                              _modified_toroidal_voronoi_mesh)
from .division import division_axis, mod_division_axis
from scipy.integrate import odeint
import sys

#rand=np.random
default_vm_parameters = [1.0,   # K for the group_name faces
                         1.0,   # A0 for the group_name faces
                         0.04,  # Gamma for the group_name faces
                         0.075, # Lambda for the edges of the group_name faces
                         0.5,   # lambda_boundary
                         0.0,   # P for the edges of the group_name faces
                         0.0    # boundary_P for the edges of the group_name faces
                        ]


rand=np.random

lifespan=100.0

force=TargetArea()+Pressure()+Perimeter()+Tension()

def property_update(cells, ready):
    """
    Updates the cells object's properties after division. 
    
    Args:
        cells is a cells object
        ready is an array of face_ids which about to undergo division.
        
    Returns:
        an updated version of the cells object's properties dictionary.
        
    """
    properties = cells.properties
    n_face = cells.mesh.n_face
    if 'K' in properties:
        properties['K'] = np.append(properties['K'],np.repeat(properties['K'][ready],2))
    if 'Gamma' in properties:
        properties['Gamma'] = np.append(properties['Gamma'],np.repeat(properties['Gamma'][ready],2))
    if 'Lambda'in properties:
        properties['Lambda'] = np.append(properties['Lambda'],np.repeat(properties['Lambda'][ready],2))
    if 'P'in properties:
        properties['P'] = np.append(properties['P'],np.repeat(properties['P'][ready],2))
    if 'source' in properties:
        properties['source'] = np.append(properties['source'],np.repeat(properties['source'][ready],2))
    if 'cluster' in properties:
        properties['cluster'] = np.append(properties['cluster'],np.repeat(properties['cluster'][ready],2))    
    if 'parent' in properties:
        properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))
    if 'left' in properties:
        properties['left'] = np.append(properties['left'],np.repeat(properties['left'][ready],2))
    if 'turnover' in properties:
        properties['turnover'] = np.append(properties['turnover'],np.repeat(properties['turnover'][ready],2))
    if  'ageingrate' in properties: #
        if 'turnover' in properties:
            new_turnover_rates = properties['turnover'][-2*len(ready):] #already updated.  Correspond to turnover for new cells
            extension = np.zeros_like(new_turnover_rates)
            for k in range(len(extension)): #set ageingrate as function of turnover.
                extension[k] = np.abs(np.random.normal(1.0*new_turnover_rates[k]/lifespan,0.2*new_turnover_rates[k]/lifespan))
            properties['ageingrate'] =np.append(properties['ageingrate'],extension)
        else:   
            properties['ageingrate'] =np.append(properties['ageingrate'], np.maximum(np.random.normal(1.0/lifespan,0.2/lifespan,len(2*len(ready))), 0.5/lifespan))
            # properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,int(2*len(ready)))))
    if  'parent_group' in properties: #heredity of type
        properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2))  # Daugthers and parent have the same ids  
    if 'age' in properties: #if
        properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))#extend
    if 'A0' in properties:    
        properties['A0']=np.append(properties['A0'],np.repeat(properties['A0'][ready],2))
    if 'leaving' in properties: #not sure what this is for.  To make leaving cells shrink and die.
        properties['leaving'] = np.append(properties['leaving'], np.zeros(2*len(ready)))  
    if 'offspring' in properties: #trace family tree.  NOT IN USE ANYMORE
        properties['offspring'] = np.append(properties['offspring'],-1*np.ones(2*len(ready))) #extend (MAKES INTO 1-D ARRAY)
        for k in range(len(ready)):
            properties['offspring'][2*ready[k]] = n_face + k #id of daughter cell of cell ready[k]
            properties['offspring'][2*ready[k]+1] = n_face + k +1 #id of other daughter cell of cell ready[k]
        l = len(properties['offspring'])
        properties['offspring'] = properties['offspring'].reshape(l/2,2)
    if 'family' in properties: #will be used to trace full family tree.
        for k in range(len(ready)):
            fill = ready[k]
            properties['family'] = np.append(properties['family'],fill)
            properties['family'] = np.append(properties['family'],fill)
    return properties #properties for the new cells object


def ready_to_divide(cells):
    """
    Args:
        cells is a cells object
    Returns:
        An array of indices of faces which are ready to divide.
    """
    properties = cells.properties
    if 'age' in properties:
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0] 
    else:#~cells.empty()[i] is True iff cell i is alive, it's False otherwise
        ready = np.where(~cells.empty() & (cells.mesh.area>=0.95*A_c))[0]
    return ready


def build_cells_simple(size=None, grid_type=None): #used in Neural_Tube2_setup, in NT_full_sim_seq
    """
    size is a list of two positive, even integers
    size[0] is the number of cells horizontally in the cell grid
    size[1] is the number of cells vertically in the cell grid
    The default size is [10,10]
    If grid type is none, the initial arrangement of cells will be hexagonal
    If not, it will be random.
    The geometry of the initial mesh will always be toroidal for now.
    source_width is an approximate thickness (as an integer number of cells)
    for the floor plate.  It must be less than size[0].
    
    """
    if size is None:
        size=[10,10] #size[0] is the number of cells in the grid
    else:
        size=size
    if grid_type is None:
        mesh=_modified_toroidal_hex_mesh(size[0], size[1],noise=0.2,rand=np.random)
    else:
        mesh=_modified_toroidal_random_mesh(size[0], size[1],noise=0.2,rand=np.random)
    cells = Cells(mesh)
    return cells
                   

def setup_source(cells, width=None): #used in NT_full_sim_seq
    """
    Defines the faces which are morphogen producers as a vertical strip.
    The default width of the strip is approximately two cells.
    Args:
    width is approximately the number of cells wide the source is.
    Also defines a property 'left' for the faces which are in the source strip,
    or on the left of it.
    """  
    approx_cell_width = 2*np.mean(cells.mesh.length)
    cents=cells.mesh.centres.T # centroids2(cells)
    n_face = cells.mesh.n_face
    cells.properties['source']=np.zeros(n_face)
    cells.properties['left']=np.zeros(n_face)
    if width is None:
        width=2
    for k in range(n_face):
        if (cents[k][0] < 0.5*width*approx_cell_width):
            cells.properties['left'][k]=1.0
            if (cents[k][0] > -0.5*width*approx_cell_width):
                cells.properties['source'][k]=1.0


def setup_cluster_cells(cells, centre=None, size=None):
    """
    FE is a finite element object 
    centre is an array or list.  It's the approximate centre of the cluster
    centre[0] is the approx number of cells right  of is the middle
    centre[1] is the approx number of cells up from the x axis
    size is an array or list.  
    size[0] is the approx number of cells across the cluster
    size[1] is the approx number of cells vertically
    """  
    approx_cell_width = 2*np.mean(cells.mesh.length)
    cents= cells.mesh.centres.T # centroids2(cells) #from FinElt4
    n_face =cells.mesh.n_face
    cells.properties['cluster']=np.zeros(n_face)
    if centre is None:
        centre=[3,2]
    if size is None:
        size=[2,2]
    alive=~cells.empty()
    for k in range(n_face):
        if alive[k]:
            if (cents[k][0] > (centre[0]-0.5*size[0])*approx_cell_width):
                if (cents[k][0] < (centre[0]+0.5*size[0])*approx_cell_width):
                    if (cents[k][1] > (centre[1]-0.5*size[1])*approx_cell_width):
                        if (cents[k][1] < (centre[1]+0.5*size[1])*approx_cell_width):
                            cells.properties['cluster'][k]=1.0    
    

def set_physical_properties(cells, physical_parameters=None): #used in NT_full_sim_seq
    """
    Creates physical property vectors for the cells object.  Sets the values 
    for all cells to those input in group parameters.
    Args:
        cells is a cells object
        physical_parameters is a vector of floats.
        
    physical_parameters is a vector of floats
    physical_parameters[0] is K for the group_name faces
    physical_parameters[1] is A0 for the group_name faces
    physical_parameters[2] is Gamma for the group_name faces
    physical_parameters[3] is Lambda for the edges of the group_name faces
    physical_parameters[4] is Lambda_boundary
    physical_parameters[5] is P for the edges of the group_name faces
    physical_parameters[6] is boundary_P for the edges of the group_name faces
    
     ['K','A0','Gamma','Lambda','Lambda_boundary','P','boundary_P']
    The physical parameters 'K', 'A0',.. are assumed to already exist.
    """
    n_face=cells.mesh.n_face
    dummy = np.zeros(n_face)
    if physical_parameters is None:
        physical_parameters = [1.0,1.0,0.04,0.075,0.0,0.5,0.0] #check values 
    cells.properties['K'] = physical_parameters[0] * np.ones(n_face)
    cells.properties['A0'] = physical_parameters[1] * np.ones(n_face)
    cells.properties['Gamma'] = physical_parameters[2] * np.ones(n_face)
    cells.properties['Lambda'] = physical_parameters[3] * np.ones(n_face)
    cells.properties['Lambda_boundary'] = physical_parameters[4] #bdy properties are scalars?
    cells.properties['P'] = physical_parameters[5] * np.ones(n_face)
    cells.properties['boundary_P']=physical_parameters[6] #bdy properties are scalars?
    

def set_group_properties(cells,group_name, group_parameters): #used in NT_full_sim_seq
    """
    Assumes physical property vectors exist.
    Sets the property vector values by group according to the group
    parameters.
    Args:
        cells is a cells object
        group_name (string) is the name of a group of cells
        cells.properties[group_name] is an array of length n_face
        cells.properties[group_name][i] = 1 iff cell i belongs to group_name
        cells.properties[group_name][i] = 0 otherwise
        group_parameters is a vector of physical parameters to be assigned
        to the cells in group_name.
    group_parameters is a vector of floats
    group_parameters[0] is K for the group_name faces
    group_parameters[1] is A0 for the group_name faces
    Note that if age is a property (as in IKNM), we do not set A0
    group_parameters[2] is Gamma for the group_name faces
    group_parameters[3] is Lambda for the edges of the group_name faces
    group_parameters[4] is lambda_boundary
    group_parameters[5] is P for the edges of the group_name faces
    group_parameters[6] is boundary_P for the edges of the group_name faces
    
    Order is ['K','A0','Gamma','Lambda','lambda_boundary','P','boundary_P']
    
    Note sure about what the boundary properties 'Lambda_boundary' and 'boundary_P'
    do.  For the moment we do not set them by group.  We include them in the input,
    (i.e. group_parameters will have length 7) but they will not be changed from 
    the default values for now.
    """
    group_ids=np.where(cells.properties[group_name]==1)[0]
    cells.properties['K'][group_ids]=group_parameters[0]
    if 'age' not in cells.properties: #
        cells.properties['A0'][group_ids]=group_parameters[1]
    cells.properties['Gamma'][group_ids]=group_parameters[2]
    cells.properties['Lambda'][group_ids]=group_parameters[3]
    #cells.properties['Lambda_boundary'][group_ids]=group_parameters[4]
    cells.properties['P'][group_ids]=group_parameters[5]
    #cells.properties['boundary_P'][group_ids]=group_parameters[6]        


def set_zposn_A0(cells):
    N_G1=1-1.0/t_G1*cells.properties['age'] #nuclei position in G1 phase
    N_S=0
    N_G2=1.0/(t_G2)*(cells.properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
    cells.properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
    """Target area function depending age and z nuclei position"""
    cells.properties['A0'] = (cells.properties['age']+1.0)*0.5*(1.0+cells.properties['zposn']**2)
    
def update_zposn_and_A0(cells):
    """
    Args:
        cells object
        
    updates the properties 'zposn' and 'A0' in the cells object's properties
    dictionary.
    
    """
    N_G1=1-1.0/t_G1*cells.properties['age'] #nuclei position in G1 phase
    N_S=0
    N_G2=1.0/(t_G2)*(cells.properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
    cells.properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
    """Target area function depending age and z nuclei position"""
    cells.properties['A0'] = (cells.properties['age']+1.0)*0.5*(1.0+cells.properties['zposn']**2)
    if 'leaving' in cells.properties:
        cells.properties['A0'] *= 1 - cells.properties['leaving']

def update_age(cells,dt):
    """
    Must be placed before 'update_zposn_and_A0'.
    
    """
    cells.properties['age'] = cells.properties['age']+dt*cells.properties['ageingrate']

def update_leaving(cells,dt,diff_rates=None):
    if diff_rates is not None:
        rates = diff_rates
    else:
        rates = 0.5 * time_hours * diff_rate_hours * np.zeros(len(cells))
    cells.properties['leaving'] = cells.properties['leaving'] + (1 - cells.properties['leaving']) * (~(cells.empty()) & (rand.rand(len(cells)) < dt*rates))

    
def cells_setup(size=None, vm_parameters=None,source_data=None,cluster_data=None,differentiation=True):
    """
    Args:
        size can be a list [a,b] where a and b are positive integers
        vm_parameters are the physical parameters for the vertex model
        If no parameters are input, the default parameters will be given to each cell
        source_data is a list [width, source_vm_parameters]
            width is an integer.  It is the approximate width of the source
            as a number of cells
            source_vm_parameters is a list of parameters for the source cells.
                same order as explained above.
            If source_data is not None, it may be either [width] or 
            [width, source_vm_parameters].
          cluster_data is a list [centre, size,cluster_vm_parameters]:
              centre is a list [a,b]. a and b are the approx coords of the 
              centre of the cluster in numbers of cells
              size is a list [c,d], c is the number of cells across the cluster
              d is the number of cells  the  cluster vertically
              Either no centre or size are specified or they BOTH must be.
              cluster_vm_parameters are a list of vm parameters for the cluster
              Either cluster data is not specified, or it is [] (default cluster), 
              or[centre, size] (center and size specified but std vm parameters) 
              or it's [centre, size, cluster_parameters] when we  also specify 
              cluster_parameters.
    Returns:
        A cells object which is initialised and ready to be the starting cells
        object for a simulation.
    """
    cells=build_cells_simple(size) #make a size[0],size[1] array of cells. Default 10x10
    #cells.properties['all']=np.ones(cells.mesh.n_face)
    if vm_parameters is None:
        vm_parameters = default_vm_parameters
    set_physical_properties(cells, vm_parameters) #sets the vm parameters for each cell as specified
    cells.properties['parent'] = np.array(list(range(cells.mesh.n_face))) #to track descendents
    if source_data is None:
        setup_source(cells) #defines the 'source' and 'left' properties 
    else:
        setup_source(cells,source_data[0]) #sets width of source if specified
        if len(source_data ==2): #if vm parameters are input, this sets them
            set_group_properties(cells,'source', source_data[1]) 
    if cluster_data is not None:
        if len(cluster_data)==0: #input is [], generate default cluster
            setup_cluster_cells(cells)
        else:
            setup_cluster_cells(cells, cluster_data[0], cluster_data[1]) #create cluster as specified
            if len(cluster_data)==3: #set up cluster properties if specified
                set_group_properties(cells,'cluster', cluster_data[2]) 
    if differentiation:
        cells.properties['leaving'] = np.zeros(len(cells))
        cells.properties['diff_rates'] = np.zeros(len(cells))
    return cells 

def divide_ready(cells, ready):  
    """
    Performs divisions AND updates properties dictionary.
    Args:
        ready is an array of cell ids to divide.
    
    """
    if len(ready)==0: #do nothing
        return cells
    edge_pairs=[]
    for cell_id in ready:
        edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #print edge_pairs  
    mesh=cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        #print "add edges
    props = property_update(cells, ready) #update cells.properties
    cells2 = Cells(mesh,props) #form a new cells object with the updated properties. WHen we update properties, change to cells2 =Cells(mesh,prop)
    return cells2

def T1_and_T2_transitions(cells):
    """
    Performs T1 and T2 transitions.
    Args:
        cells object
    Returns:
        cells object after any requires T1 and T2 transitions have taken place.
    """
    cells=T1_cells(cells)
    cells=rem_collapsed_cells(cells)
    return cells

def add_IKNM_properties(cells):   
    cells.properties['age']=np.random.rand(len(cells)) #np.random.normal(0.8,0.15,len(cells)) #create age property
    cells.properties['ageingrate'] = np.maximum(np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)), 0.5/lifespan) #create ageing_rate property
    set_zposn_A0(cells)
    

def cells_evolve(cells,dt,expansion=None,vertex=True,diff_rates=None,diff_adhesion=None):
    """
    Same as evolve, just renamed (for use in another method called 'evolve')
    """
    old_verts = cells.mesh.vertices
    if vertex: # move with forces
        force= TargetArea()+Perimeter()+Pressure() # +Tension()
        F = force(cells)
        
        # if differential adhesion, it's convenient to compute the surface
        # tension force explicitly here.
        # since the Lambda parameter is a cell property (applying to all
        # edges belonging to a given cell), rather than changing
        # the line tension of the specific edges, we change the vector 
        # of edges lengths that is used to compute the tension force
        len_modified = cells.mesh.length.copy()

        if diff_adhesion is not None:
            # find indices of edges of the floor plate (FP) region
            # 1. find the edges associated to FP cells
            edges_fp = (cells.properties['source'][cells.mesh.face_id_by_edge]==1)
            # 2. find the index of the edges whose REVERSE are NOT associated to FP cells
            rev_not_fp = (cells.properties['source'][cells.mesh.face_id_by_edge[cells.mesh.edges.reverse]]==0)
            # 3. the intersection between the two are the indices of the edges of FP cells bordering with other cells
            bdr_fp = np.where(edges_fp & rev_not_fp)[0]
            # 4. together with their reverse, thei give the complete boundary between FP cells and other cells
            bdr_fp = np.unique(np.hstack([bdr_fp, cells.mesh.edges.reverse[bdr_fp]]))

            # 5. we divide the length by a parameter larger than 1,
            #    that is equivalent to multiply Lambda by that parameter.
            len_modified[bdr_fp] /= diff_adhesion
        
        # compute the tension with the modified edges lengths
        tension = (0.5*cells.by_edge('Lambda','Lambda_boundary')/len_modified)*cells.mesh.edge_vect

        F = F + tension
        dv = dt*sum_vertices(cells.mesh.edges,F/viscosity) #viscosity is from 'constants'
        cells.mesh = cells.mesh.moved(dv)
        
    if expansion is None:
        expansion=np.zeros(2)
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])/(cells.mesh.geometry.height**2)

    # move only by stretching
    cells.mesh = cells.mesh.scaled(1.0+dt*expansion) #expansion a global constant
    new_verts = cells.mesh.vertices
    cells.mesh.velocities = (new_verts - old_verts)/dt

    if 'leaving' in cells.properties:
        update_leaving(cells,dt,diff_rates=diff_rates)
    if 'age' in cells.properties:
        update_age(cells,dt)
    if 'zposn' in cells.properties:
        update_zposn_and_A0(cells)
    return cells #, expansion
