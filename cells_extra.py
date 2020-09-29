#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:38:32 2020

@author: andrewg
"""

from mesh import *
import numpy as np
from Global_Constant import *
from cells import Cells
from forces import *
import copy
from initialisationEd import *
from initialisationEd import _modified_toroidal_hex_mesh, _modified_toroidal_random_mesh#, _modified_toroidal_voronoi_mesh
from Finite_Element import *
from Finite_Element import _add_edges
from run_select import division_axis, mod_division_axis
from scipy.integrate import odeint
import sys

#rand=np.random
default_vm_parameters = [1.0,1.0,0.04,0.075,0.5,0.0,0.0]


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
            properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(np.random.normal(1.0/lifespan,0.2/lifespan,int(2.0*len(ready)))))
    if  'parent_group' in properties: #heredity of type
        properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2))  # Daugthers and parent have the same ids  
    if 'age' in properties: #if
        properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))#extend
    if 'A0' in properties:    
        properties['A0']=np.append(properties['A0'],np.repeat(properties['A0'][ready],2))
    if 'poisoned' in properties: #not sure what this is for.  To make poisoned cells shrink and die.
        properties['poisoned'] = np.append(properties['poisoned'], np.zeros(2*len(ready)))  
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

#offspring[k] = [-1,-1] if cell k has not divided.
#If offspring[k]=[a,b], a,b>0 if cell k has divided and a,b are the ids of the
#daughter cells.
#The point of this is to allow us to work backwards and find out a cell's
#family tree if we want.

def physical_property_update(cells):
    """
    If there are groups of cells with different physical parameters, this function
    sets/updates the physical parameters appropriately. 
    
    Args:
        cells is a cells object.
    
    """
    if 'default_parameters' in cells.properties:
        cells.properties['all']=np.ones(cells.mesh.n_face)
        set_group_properties(cells,'all', cells.properties['default_parameters'])
    if 'source_parameters' in cells.properties:
        set_group_properties(cells,'source', cells.properties['source_parameters'])
    if 'cluster_parameters' in cells.properties():  
        set_group_properties(cells,'cluster_parameters', cells.properties['cluster_parameters'])

def property_updateIKNM(cells, dt):
    """
    As in Pilar's work.
    Updates a cell's age as a function of its ageing_rate.
    Updates the z nuclei position ('zposn') as a function of age.
    Sets target area 'A0' for each cells as a function of the cell's age
    and its z nucleus position  and its ageing_rate ('ageingrate')
    
    Args:
        cells is a cells object.
        dt is the time step.
        
    """
    cells.properties['age']=cells.properties['age']+ dt*cells.properties['ageingrate']
    """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
    N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
    N_S=0
    N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
    cells.properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
    """Target area function depending age and z nuclei position"""
    cells.properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)
    
def divide2_cells(cells):  
    """
    divide2 modified for cells objects
    """
    properties = cells.properties
    if 'age' in properties:
        ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0] 
    else: #~cells.empty() is list. ith entry True if cell i alive, False otherwise.
        ready = np.where(~cells.empty() & (cells.mesh.area>=0.95*A_c))[0] #added 0.5
    edge_pairs=[]
    if len(ready)==0: #do nothing
        return cells
    for cell_id in ready:
        edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #print edge_pairs  
    mesh=cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        #print "add edges
    props = property_update(cells, ready) #update cells.properties
    cells2 = Cells(mesh,props) #form a new cells object with the updated properties. WHen we update properties, change to cells2 =Cells(mesh,prop)
    return cells2

def divide2_cells2(cells,ready):  
    """
    divide2 modified for cells objects
    """
    edge_pairs=[]
    if len(ready)==0: #do nothing
        return cells
    for cell_id in ready:
        edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #print edge_pairs  
    mesh=cells.mesh.add_edges(edge_pairs) #Add new edges in the mesh
        #print "add edges
    props = property_update(cells, ready) #update cells.properties
    cells2 = Cells(mesh,props) #form a new cells object with the updated properties. WHen we update properties, change to cells2 =Cells(mesh,prop)
    return cells2

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
    
def build_cells(size=None, grid_type=None , source_width=None,age=None,parent=None,left= None):
    """
    size is a list of two positive, even integers
    size[0] is the number of cells horizontally in the cell grid
    size[1] is the number of cells vertically in the cell grid
    The default size is [10,10]
    If grid type is none, the initial arrangement of cells will be hexagonal.
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
    properties=cells.properties
    n_face=cells.mesh.n_face
    if source_width is not None:
        approx_cell_width = cells.mesh.geometry.width/float(size[0])
        centroids = centroids2(cells)
        source_vect=np.zeros(n_face)
        left_vect=np.zeros(n_face)
        left_xlim = -0.5*source_width*approx_cell_width 
        right_xlim = -left_xlim
        for k in range(n_face):
            if centroids[k][0] < right_xlim:
                left_vect[k]=1.0
                if (centroids[k][0] > left_xlim):
                    source_vect[k]=1.0
        properties['source']=source_vect
        properties['left']=left_vect
    return cells

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
                   
def set_edge_parameters(cells,prop_name,default,prop_vector=None,param=None):
    """
    prop_name is a string
    param is a 2-d list of parameters
    prop_vector[i]=1 means cell i is of type 1
    and prop[i]=0 means cell i is of type 0. 
    type 0 faces get param[0]
    type 1 faces get param[1]
    default is a default value for the parameter in question
    """   
    if param==None:
        cells.properties[prop_name]=default
    if type(param)==float or type(param)==int:
        cells.properties[prop_name]=param
    else:
        n_edge=len(cells.mesh.edges.reverse)
        n_face=cells.mesh.n_face
        cells.properties[prop_name]=param[0]*np.ones(n_edge)
        for i in range(n_face):
            if prop_vector[i]==1:
                dummy= cells.mesh.boundary(i)
                for k in dummy:
                    cells.properties[prop_name][k]=param[1]
                    
def set_face_parameters(cells,prop_name,default,prop_vector=None,param=None):
    """
    prop_name is a string
    param is a 2-d list of parameters
    prop_vector[i]=1 means cell i is of type 1
    and prop[i]=0 means cell i is of type 0. 
    type 0 faces get param[0]
    type 1 faces get param[1]
    default is a default value for the parameter in question
    """   
    if param==None:
        cells.properties[prop_name]=default
    if type(param)==float or type(param)==int:
        cells.properties[prop_name]=param
    else:
        n_face=cells.mesh.n_face
        cells.properties[prop_name]=param[0]*np.ones(n_face)
        for i in range(n_face):
            if prop_vector[i]==1:
                cells.properties[prop_name][i]=param[1]

def set_physical_properties(cells, prop=None,K=None,A0=None,Gamma=None,Lambda=None,Lambda_bdy=None,P=None,boundary_P=None):
    """
    Sets the properties for the cells object according to their type.  The 'type'
    of cells is specifed by the input 'prop.'  The input 'prop' is either None 
    or a vector of length n_face.  
    If prop is None, all faces and edges get the same parameters.  These will 
    either be set by input values (which must be 1-d) of K,A0,... or they will
    be given default values.
    
    Suppose that prop is a vector of length n_face >0.
    prop[i]=1 means cell i is of type 1
    and prop[i]=0 means cell i is of type 0.  Note that the dimension of
    K,A0,... equals 2 when prop is specified.
    
    
    Consider K.  This is a face parameter.
    
    If a cell is of type i, it gets K[i] as its K parameter.
    
    Consider Lambda.
    
    If a cell is of type i, its edges (given by cells.mesh.boundary(i)) get
    parameter Lambda[i]
    
    
    """
    set_face_parameters(cells,'K',1.0,prop,K)
    set_face_parameters(cells,'A0',1.0,prop,A0)
    set_face_parameters(cells,'Gamma',0.04,prop,Gamma)
    set_edge_parameters(cells,'Lambda',0.075,prop,Lambda)
    set_edge_parameters(cells,'P',P,prop,0.0)
    set_edge_parameters(cells,'boundary_P',P,prop,boundary_P)
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
    cents=centroids2(cells)
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

        
                
def setup_cluster(FE, centre, size, parameters):#used in NT_full_sim_seq. Argument 'parameters' is not used
    """
    FE is a finite element object 
    centre is an array.  It's the approximate centre of the cluster
    centre[0] is the approx number of cells right  of is the middle
    centre[1] is the approx number of cells up from the x axis
    size is an array.  
    size[0] is the approx number of cells across the cluster
    size[1] is the approx number of cells vertically
    parameters is a vector of floats
    parameters[0] is K for the cluster faces
    parameters[1] is A0 for the cluster faces
    parameters[2] is Gamma for the cluster faces
    parameters[3] is Lambda for the edges of the cluster faces
    parameters[4] is P for the edges of the cluster faces
    parameters[5] is boundary_P for the edges of the cluster faces
    """  
    approx_cell_width = 2*np.mean(FE.cells.mesh.length)
    cents=FE.centroids
    n_face = cells.mesh.n_face
    FE.cells.properties['cluster']=np.zeros(n_face)
    for k in FE.living_faces:
        if (centoids[k][0] > (centre[0]-0.5*size[0])*approx_cell_width):
            if (centoids[k][0] < (centre[0]+0.5*size[0])*approx_cell_width):
                if (centoids[k][1] > (centre[1]-0.5*size[1])*approx_cell_width):
                    if (centoids[k][1] > (centre[1]+0.5*size[1])*approx_cell_width):
                        FE.cells.properties['cluster'][k]=1.0
                        
def setup_cluster2(FE, centre=None, size=None, parameters=None):
    """
    FE is a finite element object 
    centre is an array or list.  It's the approximate centre of the cluster
    centre[0] is the approx number of cells right  of is the middle
    centre[1] is the approx number of cells up from the x axis
    size is an array or list.  
    size[0] is the approx number of cells across the cluster
    size[1] is the approx number of cells vertically
    parameters is a vector of floats
    parameters[0] is K for the cluster faces
    parameters[1] is A0 for the cluster faces
    parameters[2] is Gamma for the cluster faces
    parameters[3] is Lambda for the edges of the cluster faces
    parameters[4] is P for the edges of the cluster faces
    parameters[5] is boundary_P for the edges of the cluster faces
    """  
    approx_cell_width = 2*np.mean(FE.cells.mesh.length)
    cents=FE.centroids
    n_face = FE.cells.mesh.n_face
    FE.cells.properties['cluster']=np.zeros(n_face)
    if centre is None:
        centre=[3,2]
    if size is None:
        size=[2,2]
    for k in FE.living_faces:
        if (FE.centroids[k][0] > (centre[0]-0.5*size[0])*approx_cell_width):
            if (FE.centroids[k][0] < (centre[0]+0.5*size[0])*approx_cell_width):
                if (FE.centroids[k][1] > (centre[1]-0.5*size[1])*approx_cell_width):
                    if (FE.centroids[k][1] < (centre[1]+0.5*size[1])*approx_cell_width):
                        FE.cells.properties['cluster'][k]=1.0
    if parameters is not None:
        FE.cells.properties['cluster_parameters']=parameters
        
def setup_cluster3(FE, centre=None, size=None): #used in NT_full_sim_seq
    """
    FE is a finite element object 
    centre is an array or list.  It's the approximate centre of the cluster
    centre[0] is the approx number of cells right  of is the middle
    centre[1] is the approx number of cells up from the x axis
    size is an array or list.  
    size[0] is the approx number of cells across the cluster
    size[1] is the approx number of cells vertically
    """  
    approx_cell_width = 2*np.mean(FE.cells.mesh.length)
    cents=FE.centroids
    n_face = FE.cells.mesh.n_face
    FE.cells.properties['cluster']=np.zeros(n_face)
    if centre is None:
        centre=[3,2]
    if size is None:
        size=[2,2]
    for k in FE.living_faces:
        if (FE.centroids[k][0] > (centre[0]-0.5*size[0])*approx_cell_width):
            if (FE.centroids[k][0] < (centre[0]+0.5*size[0])*approx_cell_width):
                if (FE.centroids[k][1] > (centre[1]-0.5*size[1])*approx_cell_width):
                    if (FE.centroids[k][1] < (centre[1]+0.5*size[1])*approx_cell_width):
                        FE.cells.properties['cluster'][k]=1.0

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
    cents= centroids2(cells) #from FinElt4
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
    
def set_group_properties(cells,group_name, group_parameters):
    """
    Creates physical property vectors for the cells object if they do
    not exist already.
    Then it sets the property vectors by group according to the group
    parameters.
    cells is a cells object
    group_name (string) is the name of a group of cells
    cells.properties[group_name] is an array of length n_face
    cells.properties[group_name][i] = 1 iff cell i belongs to group_name
    cells.properties[group_name][i] = 0 otherwise
    The physical parameters 'K', 'A0',.. are assumed to already exist.
    parameters is a vector of floats
    parameters[0] is K for the group_name faces
    parameters[1] is A0 for the group_name faces
    Note that if age is a property (as in IKNM), we do not set A0
    parameters[2] is Gamma for the group_name faces
    parameters[3] is Lambda for the edges of the group_name faces
    parameters[4] is lambda_boundary
    parameters[5] is P for the edges of the group_name faces
    parameters[6] is boundary_P for the edges of the group_name faces
    NEED TO INCLUDE LAMBDA_BDY PARAMETER FOR GEOMETRIES WITH BOUNDARIES
    """
    n_face=cells.mesh.n_face
    n_edge = len(cells.mesh.edges.reverse)
    group_ids=np.where(cells.properties[group_name]==1)[0]
    if "K" not in cells.properties: #
        cells.properties['K']=np.ones(n_face)
    cells.properties['K'][group_ids]=group_parameters[0]
    if "A0" not in cells.properties:
        cells.properties['A0']=np.ones(n_face)
    if 'age' not in cells.properties: #
        cells.properties['A0'][group_ids]=group_parameters[1]
    if "Gamma" not in 'Lambda':
        cells.properties['Gamma']=np.ones(n_face)
    cells.properties['Gamma'][group_ids]=group_parameters[2]
    for k in group_ids:
        if 'Lambda' not in cells.properties:
            cells.properties['Lambda']=np.ones(n_edge)   
        cells.properties['Lambda'][cells.mesh.boundary(k)]=group_parameters[3]
        if 'Lambda_boundary' not in cells.properties:
            cells.properties['Lambda_boundary']=np.ones(n_edge)   
        cells.properties['Lambda_boundary'][cells.mesh.boundary(k)]=group_parameters[4]
        if 'P' not in cells.properties:
            cells.properties['P']=np.ones(n_edge)   
        cells.properties['P'][cells.mesh.boundary(k)]=group_parameters[5]
        if 'boundary_P' not in cells.properties:
            cells.properties['boundary_P']=np.ones(n_edge)   
        cells.properties['boundary_P'][cells.mesh.boundary(k)]=group_parameters[6]

def set_physical_properties2(cells, physical_parameters=None): #used in NT_full_sim_seq
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
    cells.properties['K'] = np.zeros(n_face)  
    cells.properties['K'].fill(physical_parameters[0])
    cells.properties['A0'] = np.zeros(n_face)
    cells.properties['A0'].fill(physical_parameters[1])
    cells.properties['Gamma'] = np.zeros(n_face) 
    cells.properties['Gamma'].fill(physical_parameters[2])
    cells.properties['Lambda'] = np.zeros(n_face) 
    cells.properties['Lambda'].fill(physical_parameters[3])
    cells.properties['Lambda_boundary']=physical_parameters[4] #bdy properties are scalars?
    cells.properties['P'] = np.zeros(n_face) 
    cells.properties['P'].fill(physical_parameters[5])
    cells.properties['boundary_P']=physical_parameters[6] #bdy properties are scalars?
    

def set_group_properties2(cells,group_name, group_parameters): #used in NT_full_sim_seq
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
               
def randomize(cells, N, dt):
    """
    Hacked up version of simulation_with_division (Pilar)
    Carries out N steps of IKNM to randomise the initial cell grid.
    Args:
        cells is a cells object
        N is the number of steps
        dt is the time step
    Returns:
        a randomised cells object, with empty properties.
    """
    default_physical_parameters = [1.0,1.0,0.04,0.075,0.0,0.5]
    cells.properties['all']=np.ones(cells.mesh.n_face) 
    set_group_properties(cells,'all', default_physical_parameters)
    cells.properties['age']=np.random.normal(0.8,0.15,len(cells)) #random ages
    cells.properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    cells.properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    #'parent_group' is not initialised.
    #properties['parent_group']
    expansion = np.array([0.0,0.0])
    counter=0
    x=np.array([1.0,0.04,0.075]) #K,G,L, parameters for force?
    K_val,G,L=x[0],x[1],x[2]
    rand=np.random
    while counter < N:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        cells.properties['age'] = cells.properties['age']+dt*cells.properties['ageingrate'] #add time step depending of the degradation rate 
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        N_G1=1-1.0/t_G1*cells.properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(cells.properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        cells.properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        cells.properties['A0'] = (cells.properties['age']+1.0)*0.5*(1.0+cells.properties['zposn']**2)
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)  #check edges verifing T1 transition
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F

        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        counter+=1            
    cells.properties={} #clear cell properties
    return cells

def randomize2(cells, N, dt):
    """
    Hacked up version of simulation_with_division (Pilar?)
    Carries out N steps of IKNM to randomise the intial cell grid.
    Args:
        cells is a cells object
        N is the number of steps
        dt is the time step
    Returns:
        a randomised cells object, with empty properties.
    """
    default_physical_parameters = [1.0,1.0,0.04,0.075,0.0,0.5]
    cells.properties['all']=np.ones(cells.mesh.n_face) 
    set_group_properties(cells,'all', default_physical_parameters)
    cells.properties['age']=np.random.normal(0.8,0.15,len(cells)) #random ages
    cells.properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    cells.properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #degradation rate per each cell
    #'parent_group' is not initialised.
    #properties['parent_group']
    expansion = np.array([0.0,0.0])
    counter=0
    x=np.array([1.0,0.04,0.075]) #K,G,L, parameters for force?
    K_val,G,L=x[0],x[1],x[2]
    rand=np.random
    history=[]
    force= TargetArea()+Perimeter()+Tension()+Pressure()
    while counter < N:
        #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
        cells.properties['age'] = cells.properties['age']+dt*cells.properties['ageingrate'] #add time step depending of the degradation rate 
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        N_G1=1-1.0/t_G1*cells.properties['age'] #nuclei position in G1 phase
        N_S=0
        N_G2=1.0/(t_G2)*(cells.properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
        cells.properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
        
        
        """Target area function depending age and z nuclei position"""
        cells.properties['A0'] = (cells.properties['age']+1.0)*0.5*(1.0+cells.properties['zposn']**2)
        
        cells.mesh , number_T1= cells.mesh.transition(T1_eps)  #check edges verifing T1 transition
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F

        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        counter+=1  
        history.append(cells)          
    #cells.properties={} #clear cell properties
    return history

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

def update_age(cells,dt):
    """
    Must be placed before 'update_zposn_and_A0'.
    
    """
    cells.properties['age'] = cells.properties['age']+dt*cells.properties['ageingrate']
    
def colour_offsp(cells):
    cells2=cells.copy()
    cells2.properties['color']=np.ones((cells2.mesh.n_face, 3)) #to store RGB number for each face
    for k in range(1,7):
        rgb_value = int_to_rgb(k)
        #print "ou est",np.where(cells2.properties['parent'] == k)[0]
        #print "just a number", cells2.mesh.n_face
        cells2.properties['color'][np.where(cells2.properties['parent'] == k)] = rgb_value
    return cells2

def colour_source(cells , rgb_value):
    """
    Args:
        cells is a cells object
        rgb_value is a np.array([i,j,k]) where i,j,k in {0,1}
    """
    n_face=cells.mesh.n_face
    cells.properties['color']=np.ones((n_face, 3)) #to store RGB number for each face
    for k in range(n_face):
        if cells.properties['source'][k]==1: #if cell k has ancestor s, it gets the colour rgb_value
            cells.properties['color'][k]=rgb_value
    return cells 

def colour_group(cells, group_name , rgb_value):
    """
    Args:
        cells is a cells object
        group_name (string) is the name of a group of cells.
        cells.properties[group_name][k] = 1 iff cell k belongs to the group
        (It's zero otherwise.)
        rgb_value is a np.array([i,j,k]) where i,j,k in {0,1}
    """
    n_face=cells.mesh.n_face
    cells.properties['color']=np.ones((n_face, 3)) #to store RGB number for each face
    for k in range(n_face):
        if cells.properties[group_name][k]==1: #if cell k has ancestor s, it gets the colour rgb_value
            cells.properties['color'][k]=rgb_value
    return cells   

def int_to_rgb(v):
    if v==1:
        return np.array([0,0,1])
    if v==2:
        return np.array([0,1,0])
    if v==3:
        return np.array([0,1,1])
    if v==4:
        return np.array([1,0,0])
    if v==5:
        return np.array([1,0,1])
    if v==6:
        return np.array([1,1,1])
    
def cells_setup(size=None, vm_parameters=None,source_data=None,cluster_data=None):
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
    set_physical_properties2(cells, vm_parameters) #sets the vm parameters for each cell as specified
    cells.properties['parent'] = np.array(list(range(cells.mesh.n_face))) #to track descendents
    if source_data is None:
        setup_source(cells) #defines the 'source' and 'left' properties 
    else:
        setup_source(cells,source_data[0]) #sets width of source if specified
        if len(source_data ==2): #if vm parameters are input, this sets them
            set_group_properties2(cells,'source', source_data[1]) 
    if cluster_data is not None:
        if len(cluster_data)==0: #input is [], generate default cluster
            setup_cluster_cells(cells)
        else:
            setup_cluster_cells(cells, cluster_data[0], cluster_data[1]) #create cluster as specified
            if len(cluster_data)==3: #set up cluster properties if specified
                set_group_properties2(cells,'cluster', cluster_data[2]) 
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
    cells.properties['age']=np.random.normal(0.8,0.15,len(cells)) #create age property
    cells.properties['ageingrate'] = np.random.normal(1.0/lifespan,0.2/lifespan,len(cells)) #create ageing_rate property
    set_zposn_A0(cells)
    
def evolve(cells,dt,expansion=None):
    """
    taken a piece from run select and made it into a code.
    dt is the time step
    force is the force function for the motion
    ADAPT FOR IKNM.
    """
    force= TargetArea()+Perimeter()+Tension()+Pressure()
    F = force(cells)/viscosity #viscosity is from Global_Constant
    dv = dt*sum_vertices(cells.mesh.edges,F)
    if expansion is None:
        expansion=np.array([0.0,0.0]) #initialise
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = 0.00015
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = 0.00015
        expansion[0] = np.abs(expansion[0])
        expansion[1] = np.abs(expansion[1])
        #print "Expansion turns out to be " , expansion
    cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion) #expansion a global constant
    if 'age' in cells.properties:
        update_age(cells,dt)
    if 'zposn' in cells.properties:
        update_zposn_and_A0(cells)
    return cells, expansion

def cells_evolve(cells,dt,expansion=None,vertex=True):
    """
    Same as evolve, just renamed (for use in another method called 'evolve')
    """
    old_verts = cells.mesh.vertices
    if vertex: # move with forces
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(cells)/viscosity #viscosity is from Global_Constant
        dv = dt*sum_vertices(cells.mesh.edges,F)
        cells.mesh = cells.mesh.moved(dv)
        
    if expansion is None:
        expansion=np.zeros(2) #initialise
        # if hasattr(cells.mesh.geometry,'width'):
        #     expansion[0] = 0.00015
        # if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
        #     expansion[1] = 0.00015
        # expansion[0] = np.abs(expansion[0])
        # expansion[1] = np.abs(expansion[1])
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)

    # move only by stretching
    cells.mesh = cells.mesh.scaled(1.0+expansion) #expansion a global constant
    new_verts = cells.mesh.vertices
    cells.mesh.velocities = (new_verts - old_verts)/dt

    if 'age' in cells.properties:
        update_age(cells,dt)
    if 'zposn' in cells.properties:
        update_zposn_and_A0(cells)
    return cells #, expansion

        

def living_not_source(cells):
    """
    cells.properties dictionary must contain
    """
    if 'source' in cells.properties:
        return ~cells.empty()[cells.properties['source'].astype(bool)]
    else:
        return ~cells.empty()
    
