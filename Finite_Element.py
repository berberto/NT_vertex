#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:39:42 2020

@author: andrewg
"""
import multiprocessing
import numpy as np
#from FELT import periodicFEMstep_v2, tri_cover, tri_exposure, cc_bdy, cc_bdy2, periodicFEMstep_v2_parallel
from run_select import run, basic_simulation
from forces import TargetArea, Perimeter, Tension, Pressure
from mesh import Edges, _T1, sum_vertices
from mesh import _remove as _mesh_remove #?????????
from cells import Cells
from Global_Constant import dt, viscosity, t_G1, t_G2, t_S, A_c, J, pos_d, T1_eps, P, microns, time_hours, expansion_constant #file with necessary constants
#import vertex_model as model
import copy



class FE2(object):
    """
    Attributes:
        cells: a cells object
        grid_boundary: the  edges which form the boundary of the grid.
        periodic_boundary: a subset of grid_boundary.  These are the edges 
                            through/over which the morphogen flows.
        living_faces: a list of the faces (cells) which are currently alive
        edges_to_nodes:  edges_to_nodes[i] is the index of the FE node
                                corresponding to the root of edge i.
        centroids: centroids[i] is the coordinates of the centroid of face [i],
                    if face i is alive.  Otherwise it is [-1,-1]
        nodes:  This is the full list of coordinates of nodes.
                For i in 0,..,max(edges_to_nodes), the ith node corresponds to
                the root of an edge.
                For i > max(edges_to_nodes), the ith node is the centroid of 
                a living face.
        concentration: concentration[i] is the 'amount' of morphogen at node i
        triangles: triangles[i] is a list of three node indices which form a 
                    triangle in the FE mesh.
        periodic_boundary_map: np.where(periodic_boundary_map==i) for an 
                                appropriate i is the set of all nodes which
                                are 'the same' in the FE mesh.
        source is an array of length equal to the number of nodes.
        source[k]=1 if node k is a producing morphogen, otherwise it's zero  
    """

    def __init__(self,cells, grid_boundary,periodic_boundary,living_faces,
             edges_to_nodes,centroids,nodes,source,concentration,triangles, periodic_boundary_map):
        self.cells=cells
        self.grid_boundary=grid_boundary
        self.periodic_boundary=periodic_boundary
        self.living_faces=living_faces
        self.edges_to_nodes=edges_to_nodes
        self.centroids=centroids
        self.nodes=nodes
        self.source = source
        self.concentration=concentration
        self.triangles=triangles
        self.periodic_boundary_map=periodic_boundary_map
    

    
    def concentration_by_edge(self):
        """
        Returns:
            An array whose ith entry is the concentration at the root of edge i.
        """
        """
        eTn = self.edges_to_nodes
        for k in range(len(eTn)):
            eTn[k] =  int(eTn[k])
        """
        concentration_by_edge = self.concentration[self.edges_to_nodes.astype(int)]
        return concentration_by_edge

    def concentration_by_centroid(self):
        """
        Returns:
            An array of length n_face whose ith entry is -1.0 if the face is dead 
            and is the concentration at the centroid of face i otherwise.
        """
        n_face=self.cells.mesh.n_face
        n_vtx_node = int(max(self.edges_to_nodes))
        con_by_centroid=-1*np.ones(n_face)
        for i in range(n_face):
            if i in self.living_faces:
                ind=int(np.where(self.living_faces==i)[0])+ n_vtx_node +1 
                con_by_centroid[i]=self.concentration[ind]  
            else:
                con_by_centroid[i]=-1.0
        return con_by_centroid
                
    def flow(self,v,dt):
        """
        Moves the grid one step and 'flows' the morphogen one step.
        Args:
            dt is the time step
            v is the diffusion coefficient
        Remarks: Seems ok when dt < 0.1*v.
        """
        prev_nodes = self.nodes
        old_alpha = self.concentration
        f=self.source 
        tris = self.triangles
        self.move_grid(v,dt)
        new_nodes = self.nodes
        h = self.periodic_boundary_map
        self.concentration = periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,dt)
        
    def flow_ret(self,v,dt):
        """
        Moves the grid one step and 'flows' the morphogen one step.
        Args:
            dt is the time step
            v is the diffusion coefficient
        Remarks: Seems ok when dt < 0.1*v.
        """
        prev_nodes = self.nodes
        old_alpha = self.concentration
        f=self.source 
        tris = self.triangles
        self.move_grid(v,dt)
        new_nodes = self.nodes
        h = self.periodic_boundary_map
        self.concentration = periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,dt)    
        return self
        
    def flow2(self,v,pr,dr,dt,binding=None):
        """
        Moves the grid one step and 'flows' the morphogen one step.
        Args:
            dt is the time step
            v is the diffusion coefficient
            pr is the 'rate' of production of morphogen at the centroid source
            dr (>0) is the decay 'rate' of morphogen at all nodes
        Remarks: Seems ok when dt < 0.1*v.
        """
        prev_nodes = self.nodes
        old_alpha = self.concentration
        production=pr*self.source 
        if binding is not None:
            degradation = -dr*self.concentration - binding
        else:
            degradation = -dr*self.concentration #modified
        f = production + degradation
        tris = self.triangles
        self.move_grow2(dt) #was move_grid 
        new_nodes = self.nodes
        h = self.periodic_boundary_map
        self.concentration = periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,dt)
        
    def flow2_probe(self,v,pr,dr,dt,exp_const,binding=None):
        """
        Moves the grid one step and 'flows' the morphogen one step.
        Args:
            dt is the time step
            v is the diffusion coefficient
            pr is the 'rate' of production of morphogen at the centroid source
            dr (>0) is the decay 'rate' of morphogen at all nodes
        Remarks: Seems ok when dt < 0.1*v.
        """
        prev_nodes = self.nodes
        old_alpha = self.concentration
        production=pr*self.source 
        if binding is not None:
            degradation = -dr*self.concentration - binding
        else:
            degradation = -dr*self.concentration #modified
        f = production + degradation
        tris = self.triangles
        self.move_grow_probe(exp_const,dt) #was move_grid 
        new_nodes = self.nodes
        h = self.periodic_boundary_map
        self.concentration = periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,dt)
        
    def flow2_parallel(self,v,pr,dr,dt,binding=None):
        """
        Moves the grid one step and 'flows' the morphogen one step.
        Args:
            dt is the time step
            v is the diffusion coefficient
            pr is the 'rate' of production of morphogen at the centroid source
            dr (>0) is the decay 'rate' of morphogen at all nodes
        Remarks: Seems ok when dt < 0.1*v.
        """
        prev_nodes = self.nodes
        old_alpha = self.concentration
        production=pr*self.source 
        if binding is not None:
            degradation = -dr*self.concentration - binding
        else:
            degradation = -dr*self.concentration #modified
        f = production + degradation
        tris = self.triangles
        self.move_grow2(dt) #was move_grid 
        new_nodes = self.nodes
        h = self.periodic_boundary_map
        self.concentration = periodicFEMstep_v2_parallel(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,dt)
        
        
    def flow_extra(self,v,pr,dr,k,dt, binding=None):
        """
        Moves the grid one step with time step dt.
        Then 'flow' the morphogen over the same time period, but takes k+1 steps.
        k is the number of extra steps taken by the FE method.
        Args:
            dt is the time step
            v is the diffusion coefficient
            pr is the 'rate' of production of morphogen at the centroid source
            dr (>0) is the decay rate of morphogen at all nodes
            k is the number of extra steps we will take, during the cells time step
            k >=0
        Note:  We must be careful with the choice of pr and dr so that we don't
        end up with no flow. 
        """
        prev_nodes = self.nodes
        new_nodes=self.nodes #just creating a variable 
        old_alpha = self.concentration
        production=pr*self.source 
        if binding is not None:
            degradation = -dr*self.concentration - binding
        else:
            degradation = -dr*self.concentration #modified
        new_con=np.zeros_like(self.concentration)#
        f = production + degradation
        tris = self.triangles
        self.move_grow2(dt) #Moves the cell grid and 'grows' it
        final_nodes = self.nodes
        h = self.periodic_boundary_map
        if k==0: #no extra steps
            self.concentration = periodicFEMstep_v2(final_nodes, prev_nodes, tris,f, new_con, h , v ,dt)
        else: #k extra steps
            mve = (final_nodes - prev_nodes)/float(k+1) # movement of cells at each step
            fine_dt = dt/float(k+1) #finer time step for flow
            count=0
            while count< k-1:
                new_nodes = prev_nodes + mve
                new_con=periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,fine_dt)
                prev_nodes = new_nodes
                f = pr*self.source -dr*new_con #update f, pr*self.source is production, dr*new_con is decay
                count+=1
            prev_nodes=new_nodes
            new_nodes=final_nodes
            old_alpha=new_con
            self.concentration =periodicFEMstep_v2(new_nodes, prev_nodes, tris,f, old_alpha, h , v ,fine_dt)    
    
    def move_grid(self,v,dt):
        """
        Args:
            dt is the time step
            v is the diffusion coefficient
        """
        #rand=np.random
        x=np.array([1.0,0.04,0.075]) #K,G,L, parameters for force?
        K,G,L=x[0],x[1],x[2]
        force=TargetArea()+Perimeter()+Tension()+Pressure()
        dummy_cells = run(basic_simulation(self.cells,force,dt=dt,T1_eps=0.0),1,1)[0]
        temp =nodes_generator(dummy_cells, self.edges_to_nodes, self.living_faces)
        self.cells=dummy_cells #updates cells atribute
        self.nodes = temp[0] #updates nodes attribute
        self.centroids = temp[1] #updates centroids attribute
        
    def move_grow(self,dt):
        """
        taken a piece from run select and made it into a code.
        dt is the time step
        force is the force function for the motion
        
        ADAPT FOR IKNM.
    
        """
        
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(self.cells)/viscosity 
        expansion=np.array([0.0,0.0]) #initialise
        dv = dt*sum_vertices(self.cells.mesh.edges,F)
        if hasattr(self.cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*self.cells.mesh.vertices[0])*dt/(self.cells.mesh.geometry.width**2)
        if hasattr(self.cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*self.cells.mesh.vertices[1])*dt/(self.cells.mesh.geometry.height**2)
        self.cells.mesh = self.cells.mesh.moved(dv).scaled(1.0+expansion)
        
    def move_grow_probe(self,exp_const,dt):
        """
        taken a piece from run select and made it into a code.
        dt is the time step
        force is the force function for the motion
        
        ADAPT FOR IKNM.
    
        """
        
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(self.cells)/viscosity 
        expansion=np.array([0.0,0.0]) #initialise
        dv = dt*sum_vertices(self.cells.mesh.edges,F)
        #expansion[0] = 5
        expansion[0] = exp_const*np.average(F[0]*self.cells.mesh.vertices[0])*dt/(self.cells.mesh.geometry.width**2)
        self.cells.mesh = self.cells.mesh.moved(dv).scaled(1.0+expansion)
    
    def move_grow2(self,dt):
        """
        taken a piece from run select and made it into a code.
        dt is the time step
        force is the force function for the motion
        
        ADAPT FOR IKNM.
    
        """
        props=self.cells.properties
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(self.cells)/viscosity #viscosity is from the Global parameters.
        expansion=np.array([0.0001,0.0001]) #initialise
        dv = dt*sum_vertices(self.cells.mesh.edges,F)
        #if hasattr(self.cells.mesh.geometry,'width'):
         #   expansion[0] = expansion_constant*np.average(F[0]*self.cells.mesh.vertices[0])*dt/(self.cells.mesh.geometry.width**2)
        #if hasattr(self.cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
         #   expansion[1] = np.average(F[1]*self.cells.mesh.vertices[1])*dt/(self.cells.mesh.geometry.height**2)
        #print expansion
        dummy_mesh = self.cells.mesh.moved(dv).scaled(1.0+expansion)
        dummy_cells = Cells(dummy_mesh,props)
        temp =nodes_generator(dummy_cells, self.edges_to_nodes, self.living_faces)
        self.cells=dummy_cells #updates cells atribute
        self.nodes = temp[0] #updates nodes attribute
        self.centroids = temp[1] #updates centroids attribute
        
        
    def exposure(self ,i):
        """
        Args:
            i is a face_id.
        Returns:
            The amount of morphogen outside a cell.
        """
        tri_covering = tri_cover(self.cells,i) #indices of triangles which cover face i
        exposure_value=0.0
        for w in tri_covering:
            exposure_value +=  tri_exposure(self.triangles[w],self.nodes,self.concentration)
        return exposure_value 
    
    def exposure_by_face(self):
        """
        Returns a vector exposure s.t. for a living face_id i, exposure[i] is
        the amount of morphogen outside cell i.
        If face i is dead, exposure[i]=0
        Tested (once) with a surface concentration==1 everywhere.  This should equal 
        the output of the area function in mesh.  It does.
        """
        liv_faces = self.living_faces
        exposure_vector = np.zeros(self.cells.mesh.n_face)
        for k in liv_faces:
            exposure_vector[k] = self.exposure(k)
        return exposure_vector
            
    def copy(self):
        fe = copy.copy(self)
        return fe

 
    
    def T1(self,eps=None):
        """
        Performs t1 transitions on the mesh.  Almost entirely the same as
        the first part of _transition from mesh.  Adapted to return a finite
        element object.
        Args:
            eps >0 is a tolerance.  We perform type 1 transitions on edges which are
            shorter than eps.
        Returns:
            A finite element object.
            
        """
        props = self.cells.properties
        if eps==None:
            eps=0.1 #include a default epsilon value
        mesh = self.cells.mesh
        grid_boundary = self.grid_boundary 
        con_by_edge =self.concentration_by_edge()
        con_by_cent = self.concentration_by_centroid()
        edges = mesh.edges #shorthand
        half_edges = edges.ids[edges.ids < edges.reverse]  # array of indices of edges with id < reverse_id
        dv = mesh.edge_vect.take(half_edges, 1) #dv[i] is the vector from root of edge half_edges[i] to its tip. 
        short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps]) #why did they make this a set?
        ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps] #edges which are shorter than the tolerance eps>0
            
        if not short_edges: # if we have 
            return self
        reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
        rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
        while short_edges:
            edge = short_edges.pop()
            if edge in grid_boundary:
                edge = reverse[edge]
            neighbours = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
            for x in neighbours:
                short_edges.discard(x)
        mesh = mesh.copy() #copy mesh 
        mesh.edges = Edges(reverse) #set attributes
        mesh.vertices = vertices    
        mesh.face_id_by_edge = face_id_by_edge        
        cells = Cells(mesh, props) #make a cells object
        #T1 transitions do not change the concentration_by_edge or concentration_by_centroid
        return build_FE(cells, con_by_edge, con_by_cent)
        
    def rem_collapsed(self):
        """
        Removes collapsed faces:
        
            Copied overall structure from function in mesh.
            Uses modified _remove function.
        Returns:
            An new finite element object with collapsed faces removed.
            
        Remark:
            We do not need to update the concentration_by_centroid.  We would only
            be changing values which are not used when we call build_FE.  
        
        """
        props=self.cells.properties #to pass on to the new object
        edges = self.cells.mesh.edges
        reverse= self.cells.mesh.edges.reverse
        vertices = self.cells.mesh.vertices
        face_id_by_edge = self.cells.mesh.face_id_by_edge
        concentration_by_edge = self.concentration_by_edge() #added()
        rotate=self.cells.mesh.edges.rotate
        reverse=self.cells.mesh.edges.reverse
        while True:
            nxt = rotate[reverse]
            two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
            if not len(two_sided):
                break
            while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
                reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
                prev_face_id_by_edge = face_id_by_edge
                reverse, vertices, face_id_by_edge,concentration_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)#_remove only returns a tuple of length 3
                ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        # if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #     print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)
        mesh = self.cells.mesh.copy()
        mesh.edges = Edges(reverse)
        mesh.vertices = vertices
        mesh.face_id_by_edge = face_id_by_edge
        cells = Cells(mesh , props)
        return build_FE2(cells, concentration_by_edge, self.concentration_by_centroid()) #added ()

#def flow_basic(new_nodes, prev_nodes,seq_array,tris , f , old_alpha, h , v ,dt):
    """
    Will perform one step of morhogen flow with sequestration.    
    """
    


def rem_collapsed_cells(cells):#move to cells_extra
        """
        Removes collapsed faces:
        
            Copied overall structure from function in mesh.
            Uses modified _remove function.
        Returns:
            An new finite element object with collapsed faces removed.
            
        Remark:
            We do not need to update the concentration_by_centroid.  We would only
            be changing values which are not used when we call build_FE.  
        
        """
        props=cells.properties #to pass on to the new object
        edges = cells.mesh.edges
        reverse= cells.mesh.edges.reverse
        vertices = cells.mesh.vertices
        face_id_by_edge = cells.mesh.face_id_by_edge
        #concentration_by_edge = concentration_by_edge() #added()
        rotate=cells.mesh.edges.rotate
        reverse=cells.mesh.edges.reverse
        while True:
            nxt = rotate[reverse]
            two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
            if not len(two_sided):
                break
            while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
                reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
                prev_face_id_by_edge = face_id_by_edge
                reverse, vertices, face_id_by_edge = _mesh_remove(two_sided, reverse, vertices, face_id_by_edge)
                ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge) #used for?
        # if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #     print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)
        mesh = cells.mesh.copy()
        mesh.edges = Edges(reverse)
        mesh.vertices = vertices
        mesh.face_id_by_edge = face_id_by_edge
        cells = Cells(mesh , props) #create new cells object with the same properties as the original
        return cells
    
def T1_cells(cells,eps=None):
        """
        Performs t1 transitions on the mesh.  Almost entirely the same as
        the first part of _transition from mesh.  Adapted to return a finite
        element object.
        Args:
            eps >0 is a tolerance.  We perform type 1 transitions on edges which are
            shorter than eps.
        Returns:
            A finite element object.
            
        """
        props = cells.properties
        mesh = cells.mesh
        if eps==None:
            eps=0.1 #include a default epsilon value
        mesh = mesh.transition(T1_eps)[0] #returns new mesh
        cells = Cells(mesh, props)
        return cells
    
#insert divide

def move_grow_cells(cells,dt):
        """
        taken a piece from run select and made it into a code.
        dt is the time step
        force is the force function for the motion
        
        ADAPT FOR IKNM.
    
        """
        
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(cells)/viscosity #viscosity is from Global_Consant
        expansion=np.array([0.0,0.0]) #initialise
        dv = dt*sum_vertices(cells.mesh.edges,F)
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = expansion_constant*np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        expansion[0] = np.abs(expansion[0])
        expansion[1] = np.abs(expansion[1])
        print("Expansion turns out to be " , expansion)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion) #expansion a global constant
        return cells    
    
def move_basic_grow_cells(cells,dt):
        """
        taken a piece from run select and made it into a code.
        dt is the time step
        force is the force function for the motion
        
        ADAPT FOR IKNM.
    
        """
        
        force= TargetArea()+Perimeter()+Tension()+Pressure()
        F = force(cells)/viscosity #viscosity is from Global_Consant
        expansion=np.array([0.0,0.0]) #initialise
        dv = dt*sum_vertices(cells.mesh.edges,F)
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = 0.00015
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = 0.00015
        expansion[0] = np.abs(expansion[0])
        expansion[1] = np.abs(expansion[1])
        #print "Expansion turns out to be " , expansion
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion) #expansion a global constant
        return cells
        
def lr_bdy_length(fe):
    """
    Args:
        fe is a finite element object
        
    Creates a key 'left' in the properties dictionary of fe.cells, if it does 
    not already exist.  fe.cells.properties['left'][k]=1 iff face k has 
    centroid which has a negative x coordinate.  self.cells.properties['left'][k]=0
    otherwise.
    
    Returns:
        A tuple t of two numbers.
        t[0] is the boundary length
        t[1] is a flag variable, which is one if the boundary is intact, and
        is zero if some cells have detached from the 'left' group.
    """
    if "left" not in fe.cells.properties: #if "left" is not defined, create it
        fe.cells.properties['left']=np.zeros(fe.cells.mesh.n_face)
        for k in range(fe.cells.mesh.n_face):
            if fe.centroids[k][0] < 0:
                fe.cells.properties['left'][k]=1.0
    return frontier_len(fe, 'left')
        
       
"""        
def _transition(mesh, eps):
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    
    if not short_edges:
        return mesh, ids_t1
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    while short_edges:
        edge = short_edges.pop()
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        for x in neighbours:
            short_edges.discard(x)

    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
        prev_face_id_by_edge = face_id_by_edge
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        # if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #     print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)

    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1
"""               
            
def _remove(edges, reverse, vertices, face_id_by_edge , concentration_by_edge):
    """
    Mostly a direct copy of fn in mesh.  Removes edges in edges from the mesh
    and updates reverse, vertices etc.
    Args:
        edges is an array of edge ids to be removed
        all other args are standard
    Returns:
        Updated attributes of fe object.
        
    """    
    es = np.unique(edges/3*3) #for all j in edges, we obtain j , rot[j] , rot[rot[j]] in  
    to_del= np.dstack([es, es+1, es+2]).ravel()                         #to_del
    reverse = np.delete(reverse, to_del)
    reverse = (np.cumsum(np.bincount(reverse))-1)[reverse]  # relabel to get a perm of [0,...,N-1]
    vertices = np.delete(vertices, to_del, 1)
    face_id_by_edge = np.delete(face_id_by_edge, to_del)
    concentration_by_edge = np.delete(concentration_by_edge,to_del)
    return reverse, vertices, face_id_by_edge, concentration_by_edge
            
 ##No need to modify _T1() from mesh as it has no effect on the concentration 

            
def _add_edges(mesh,edge_pairs,concentration_by_edge, concentration_by_centroid):
    """
    edge_pairs is output by mesh.division_axis?
    Again mostly taken from the corresponding function in mesh.  Adapted for 
    FE.
    
    
    """
    cbye = concentration_by_edge #shorthand
    cbycent = concentration_by_centroid #shorthand
    #_add_edges(mesh, edge_pairs):
    n_edge_old = len(mesh.edges)
    n_edge_new = n_edge_old + 6*len(edge_pairs)

    n_face_old = mesh.n_face

    reverse = np.resize(mesh.edges.reverse, n_edge_new) #extend reverse
    cbye = np.resize(cbye, n_edge_new) #extend
    cbycent = np.resize(cbycent, n_face_old+ 2*len(edge_pairs)) #extend
    vertices = np.empty((2, n_edge_new)) #empty array
    vertices[:, :n_edge_old] = mesh.vertices #fill first part with vertices before dvision
    face_id_by_edge = np.resize(mesh.face_id_by_edge, n_edge_new) #extend
    rotate = Edges.ROTATE #shorthand

    v = vertices.T  # easier to work with transpose here
    n = n_edge_old #shorthand
    for i, (e1, e2) in enumerate(edge_pairs):
        # find midpoints: rotate[reverse] = next
        v1, v2 = 0.5*(v.take(rotate[reverse[[e1, e2]]], 0) + v.take([e1, e2], 0)) #calculate mid pts of e1 and e2
        v[[n, n+1, n+2]] = v1 #assign valyes to vertices.  roots of edges n, n+1 and n+2
        v[[n+3, n+4, n+5]] = v2   #are v1.  Similar for v2.
        cbye[[n,n+1,n+2]]=0.5*(cbye[e1]+ cbye[rotate[reverse[e1]]]) #concentration at mid pt = ave of root and tip concentration
        cbye[[n+3,n+4,n+5]]=0.5*(cbye[e2]+ cbye[rotate[reverse[e2]]])   #concentration at mid pt = ave of root and tip concentration
        a = [n, n+1, n+2, n+3, n+5]
        b = [e1, n+4, reverse[e1], e2, reverse[e2]]
        reverse[a], reverse[b] = b, a #setting the values of new elements of reverse appropriately

        for j, edge in enumerate((e1, e2)):
            dummy=0 # added by moi
            counter = 0.0
            crude_centroid_value=0
            face_id = n_face_old + 2*i + j #current
            while face_id_by_edge[edge] != face_id: #loops round the edges of face_id
                face_id_by_edge[edge] = face_id #sets the face_id_by_edge for each edge to face_id
                dummy +=cbye[edge] #adds concentration at root of edge to dummy
                edge = rotate[reverse[edge]]# rotate[reverse] = next, i.e. change to next edge around face_id
                counter+=1 #increment count (for average calculation after loop)
            dummy += cbycent[face_id_by_edge[edge]]# add centroid of original face
            crude_centroid_value = dummy / (counter+1) #average value of nodes around the newcentroid
                                                        #old centroid included
            cbycent[face_id] = crude_centroid_value #set concentration_by_centroid value for face_id
        re1, re2 = rotate[[e1, e2]]
        # winding (I don't understand 'winding.')
        face_id_by_edge[n] = face_id_by_edge[re1] #setting another value appropriately
        v[n] += v[re1]-v[e1] #if e1 not in grid_bdy, this is zero.  Otherwise we move the root of edge n to the other side of the grid   
        face_id_by_edge[n+3] = face_id_by_edge[re2] #setting another value 
        v[n+3] += v[re2]-v[e2] #see comments about v[n] above and below.
        n += 6 #new number of edges
        #when e1 is in the boundary of the grid, v[n] becomes v[re1]+0.5(v[nxt[e1]] - v[e1])
        #Since v[nxt[e1]] - v[e1] is the vector from v[e1] to v[nxt[e1]] and 
        #the rev[e1] is just a shifted version of the same vector, then
        #v[re1]+0.5(v[nxt[e1]] - v[e1]) is the midpt of the other side of the grid.
        
        
    mesh = mesh.copy() #copy original mesh object and give the attributes 
    mesh.edges = Edges(reverse) #the appropriate values, as calculated above.
    mesh.face_id_by_edge = face_id_by_edge
    mesh.vertices = vertices
    mesh.n_face = n_face_old + 2*len(edge_pairs)
    return mesh, cbye, cbycent         
            
#def FE_transitions(fe):
#   fe=fe.divide()
#   fe.T1()
    
            

def build_FE(cells, concentration_by_edge=None, concentration_by_centroid=None,source=None, tol=None):
    """
    Args:
        cells object.
        Optional
        concentration_by_edge
        concentration_by_centroid
        source
        tol
        
    """
    if tol is None:
        tol=0.1
    grid_bdy = grid_boundary(cells,tol) #list
    print(grid_bdy)
    per_bdy = grid_bdy #toroidal periodicity by default
    etn = edges_to_vtx_nodes(grid_bdy, cells)  #array
    print(etn)
    liv_faces=living_faces(living(cells)) #array
    nds, cents = nodes_generator(cells, etn , liv_faces)
    if source is None:
        source = np.zeros(len(nds)) #replace with set_source + add default width (in Gobal constants)
        #source[0]=1 #random TO be changed to a selection
    tris=triangles_v2(cells, etn, liv_faces) #list of lists
    if concentration_by_edge is None:
        con_by_edge = np.zeros(len(cells.mesh.edges.reverse)) #array,zero initialisation
    else:
        con_by_edge = concentration_by_edge
    if concentration_by_centroid is None:
        con_by_cent = np.zeros(cells.mesh.n_face) #array, zero initialisation
    else:
        con_by_cent = concentration_by_centroid
    concent_vtx_nodes =con_vtx_nodes(con_by_edge, etn)
    concentration= np.hstack((concent_vtx_nodes, con_by_cent[liv_faces]))#array
    #concentration[20]=8.0 #random choice to watch diffusion
    p_b_map_vtx = periodic_map(etn, cells, per_bdy) #list
    mh = max(p_b_map_vtx) 
    extend = np.array(list(range(int(mh+1), int(mh+1) +len(liv_faces)))) #give centroid nodes an appropriate new index
    p_b_map = np.hstack((p_b_map_vtx,extend))
    return FE2(cells, grid_bdy,per_bdy,liv_faces,etn,cents,nds,source,concentration,
             tris, p_b_map)
"""
(self,cells, grid_boundary,periodic_boundary,living_faces,
             edges_to_nodes,centroids,nodes,concentration,triangles, periodic_boundary_map)
print 'grid_bdy', grid_bdy
    print 'per_bdy', per_bdy
    print 'liv_faces', liv_faces
    print 'etn', etn
    print 'cents', cents
    print 'nds', nds
    print 'concentration', concentration
    print 'tris', tris
    print 'p_b_map', p_b_map
"""    

def build_FE2(cells, concentration_by_edge=None, concentration_by_centroid=None, tol=None): #used in NT_full_sim_seq
    """
    Args:
        cells object.
        Optional
        concentration_by_edge
        concentration_by_centroid
        source
        tol
        
    """
    if tol is None:
        tol=0.1
    grid_bdy = grid_boundary(cells,tol) #list
    #print grid_bdy
    per_bdy = grid_bdy #toroidal periodicity by default
    etn = edges_to_vtx_nodes(grid_bdy, cells)  #array
    #print etn
    liv_faces=living_faces(living(cells)) #array
    nds, cents = nodes_generator(cells, etn , liv_faces)
    source = np.zeros(len(nds))
    for k in liv_faces:
       if cells.properties['source'][k] > 0:
           l = face_to_node(etn, liv_faces, k) #node index of centroid of face k
           source[l]=0.0 #set to zero for test
    tris=triangles_v2(cells, etn, liv_faces) #list of lists
    if concentration_by_edge is None:
        con_by_edge = np.zeros(len(cells.mesh.edges.reverse)) #array,zero initialisation
    else:
        con_by_edge = concentration_by_edge
    if concentration_by_centroid is None:
        con_by_cent = np.zeros(cells.mesh.n_face) #array, zero initialisation
        for k in liv_faces:
            if cells.properties['source'][k] > 0: 
                con_by_cent[k]=5.0 #initial condition
    else:
        con_by_cent = concentration_by_centroid
    concent_vtx_nodes =con_vtx_nodes(con_by_edge, etn)
    concentration= np.hstack((concent_vtx_nodes, con_by_cent[liv_faces]))#array
    #concentration[20]=8.0 #random choice to watch diffusion
    p_b_map_vtx = periodic_map(etn, cells, per_bdy) #list
    mh = max(p_b_map_vtx) 
    extend = np.array(list(range(int(mh+1), int(mh+1) +len(liv_faces)))) #give centroid nodes an appropriate new index
    p_b_map = np.hstack((p_b_map_vtx,extend)) 
    return FE2(cells, grid_bdy,per_bdy,liv_faces,etn,cents,nds,source,concentration,
             tris, p_b_map)
"""
(self,cells, grid_boundary,periodic_boundary,living_faces,
             edges_to_nodes,centroids,nodes,concentration,triangles, periodic_boundary_map)
print 'grid_bdy', grid_bdy
    print 'per_bdy', per_bdy
    print 'liv_faces', liv_faces
    print 'etn', etn
    print 'cents', cents
    print 'nds', nds
    print 'concentration', concentration
    print 'tris', tris
    print 'p_b_map', p_b_map
"""    

def build_FE_steps(cells, concentration_by_edge=None, concentration_by_centroid=None,source=None, tol=None):
    """
    Args:
        cells object.
        Optional
        concentration_by_edge
        concentration_by_centroid
        source
        tol
        
    """
    if tol is None:
        tol=0.1
    grid_bdy = grid_boundary(cells,tol) #list
    print("grid_bdy", grid_bdy)
    per_bdy = grid_bdy #toroidal periodicity by default
    etn = edges_to_vtx_nodes(grid_bdy, cells)  #array
    print("etn",etn)    
    liv_faces=living_faces(living(cells)) #array
    print(liv_faces)
    m=int(max(etn))
    dummy=[]
    for i in range(int(m)+1):
        dummy.append(cells.mesh.vertices.T[np.where(etn==i)][0])
    vtx_nodes=np.array(dummy) #vtx_nodes
    print("shape vtx_nodes", np.shape(vtx_nodes))
    cents = centroids2(cells)
    print("shape cents", np.shape(cents))
    liv_cents = cents[liv_faces]
    print("shape liv_cents" , np.shape(liv_cents))
    nodes = np.vstack([vtx_nodes, liv_cents])
    print("shape nodes" , np.shape(nodes))
    """
    nds, cents = nodes_generator(cells, etn , liv_faces)
    if source is None:
        source = np.zeros(len(nds)) #replace with set_source + add default width (in Gobal constants)
        #source[0]=1 #random TO be changed to a selection
    tris=triangles_v2(cells, etn, liv_faces) #list of lists
    if concentration_by_edge is None:
        con_by_edge = np.zeros(len(cells.mesh.edges.reverse)) #array,zero initialisation
    else:
        con_by_edge = concentration_by_edge
    if concentration_by_centroid is None:
        con_by_cent = np.zeros(cells.mesh.n_face) #array, zero initialisation
    else:
        con_by_cent = concentration_by_centroid
    concent_vtx_nodes =con_vtx_nodes(con_by_edge, etn)
    concentration= np.hstack((concent_vtx_nodes, con_by_cent[liv_faces]))#array
    #concentration[20]=8.0 #random choice to watch diffusion
    p_b_map_vtx = periodic_map(etn, cells, per_bdy) #list
    mh = max(p_b_map_vtx) 
    extend = np.array(range(int(mh+1), int(mh+1) +len(liv_faces))) #give centroid nodes an appropriate new index
    p_b_map = np.hstack((p_b_map_vtx,extend))
    return FE2(cells, grid_bdy,per_bdy,liv_faces,etn,cents,nds,source,concentration,
             tris, p_b_map)
    """

def grid_boundary(cells, tol):
    """
    Args:
        cells is a cells object
        tol is a constant.  If the distance between the root of a edge k and
        the tip of edges reverse[k] is > tol, then k and rev[k] belong to the
        boundary.
    Returns:
        a list of the indices of the edges which form the grid boundary.
    Tested once and seems to work.  
    Tested with a manual count of edges of a
    mesh and length of grid_boundary matched the manual count. 
    Some care required with the choice of the constant tol>0.
    
    """
    half_edges=[]
    grid_bdy=[]
    rev=cells.mesh.edges.reverse
    nxt = cells.mesh.edges.__next__
    verts = cells.mesh.vertices.T
    for i in range(len(rev)):
        if (i < rev[i]):
            half_edges.append(i)
    for w in half_edges:
        error = verts[w] - verts[nxt[rev[w]]]
        if (np.linalg.norm(error) > tol):
            grid_bdy.append(w)
            grid_bdy.append(rev[w])
    return grid_bdy 

def edges_to_vtx_nodes(grid_boundary, cells):
    """
    Determines the edges_to nodes vector from the list of grid_boundary edges
    and a cells object.
    Args:
        cells is a cells object
        grid_boundary is a list of the edges which form the boundary of the 
        cell grid.
    Returns:
        the edges_to_vtx_nodes vector, whose ith entry is the index of the
        node which corresponds to the root of edge i.
        
    Remark:  We could also call this function edges_to_nodes.  We form the 
    nodes vector by appending the centroid nodes to the vtx_nodes.  So,the 
    index of the vertex_node and the index of the node corresponding to an edge
    will be the same.
    
    Tested. Has problems.
    """
    rev=cells.mesh.edges.reverse
    rot=cells.mesh.edges.rotate
    rot2=cells.mesh.edges.rotate2
    #nxt=cells.mesh.edges.next
    n_edge = len(cells.mesh.edges.reverse) # 
    l = -1*np.ones(n_edge) #l will be our edges_to_vtx_nodes vector
    counter = 0
    edges=list(range(len(rev))) #edges.ids is the same
    dummy=[]
    for k in grid_boundary:
        edges.remove(k)
        l[k]=counter
        counter+=1
    for k in edges:
        if(rot[k] in grid_boundary): 
            dummy.append(k)
            l[k] = l[rot[k]]       
    for n in dummy:
        edges.remove(n)
    #for k in edges: ##added
    #    if(rot2[k] in grid_boundary): 
    #        dummy.append(k)
    #        l[k] = l[rot2[k]]       
    #for n in dummy:
    #    edges.remove(n) 
    while len(edges)>0: #changed indexing
        k=edges[0]  
        k2=rot[k]
        k3=rot[k2]
        l[k]=counter
        l[k2]=counter
        l[k3]=counter
        edges.remove(k)
        edges.remove(k2)
        edges.remove(k3)
        counter+=1 
    return l 
        
def pair_mat(edges_to_nodes, cells, per_bdy):
    """
    Args:
        per_bdy is a list of edges.  Precisely, it is a subset of grid_boundary.  
        It is a list of the edges through/over which Shh can flow.
        If we define closed_boundary to be the set of edges through which no Shh
        flows, then per_bdy=grid_boundary \ closed_boundary.
        cells is a cells object.
        edges_to_nodes is a list. edges_to_nodes[i] is the index of the node 
        associated with the root of edge[i].
    Returns:
        An m x m matrix a, where m is the number of nodes in the grid.
        Entry a[i,j]=1 iff nodes i and j are regarded as being the
        same for the Shh flow and  a[i,j] = 0 otherwise.  Note that a[i,i]=0 
        for all i.
    """
    n_node=int(max(edges_to_nodes))+1
    a = np.zeros((n_node , n_node)) # if nodes i,j are paired, a_{i,j} = 1
    half_pb=[]                                       #otherwise a_{i,j}=0
    rev = cells.mesh.edges.reverse
    nxt = cells.mesh.edges.__next__
    print(nxt[rev[2]])
    for k in per_bdy: #only need to use half of l
        if (k < rev[k]): #as used by D. Page
            half_pb.append(k)
    for j in half_pb: #edge j is joined to rev[j]
        r=int(edges_to_nodes[j]) # root node of edge j
        t=int(edges_to_nodes[nxt[j]]) #tip node of edge j
        rr = int(edges_to_nodes[rev[j]]) #root node of rev[j], rr is reverse root
        tr = int(edges_to_nodes[nxt[rev[j]]]) # tip node of rev[j], rt is rev tip
        a[r][tr] = 1 #root node of edge j paired with tip node of rev[j]
        a[tr][r] = 1 #symmetry
        a[t][rr] = 1 #tip node of edge j paired with root node of rev[j]
        a[rr][t] = 1 #symmetry
    return a   

def set_source(cells, centroids,living_faces, width):
    """
    Determines the face_ids of cells whose centroids lie within a vertical
    strip {(x,y) : x in (-width/2, width/2)}
    Args:
        cells object
        centroids
        living_faces
        width > 0
    Returns a list the face_ids whose centroids lie in the strip desribed
     above.
    """
    source = np.zeros(cells.mesh.n_face)
    for k in living_faces:
        if (centroids[k][0] > -width/2.0) & (centroids[k][0] < width/2.0):
            source[k] = 1
    return source


    

def h_from_mat(a):
    """
    a is a square matrix.  a[i][j] = 1 if nodes i,j are paired. a[i][j]=0
    otherwise
    returns a vector h of length n_node.
    h[i]=h[j] means that nodes i and j are the same (as far as the Shh flow 
    is concerned.)    
    """
    n_node = len(a[0])# a will always have shape (n_node, n_node)
    h=list(-1*np.ones(n_node).astype(int))
    counter=0
    while( -1 in h ):
        start = h.index(-1)
        dummy=[start] # dummy = list of nodes which are same as start
        d2=dummy   #copy dummy
        l=list(np.where(a[start]==1)[0]) #some nodes which are the same as start
        dummy=dummy+l
        diff=list(set(dummy).difference( set(d2)))
        while(len(diff)>0): #if not, no more nodes which are the same as 
            d2=dummy
            for k in diff:
                ll=list(np.where(a[k]==1)[0]) #nodes which are the same as elts of diff
                dummy = dummy + ll
            dummy = list(np.unique(dummy)) #remove repeats for the list
            diff=set(dummy).difference(set(d2))
        for k in dummy:
            h[k] = counter
        counter+=1
    return h
 
    """
    In the matrix a, if a[i][j]=1, we have i is the same as j
    We may also have a[j][k]=1.  In which case, i, j  and k are all the same.
    """
            
def periodic_map(edges_to_nodes, cells, l):
    """
    Args:
        edges_to_nodes[i] is the index of the node associated with the 
        root of edge[i]
        cells is a cells object
        l is a list of periodic boundary edges.  These are the edges of the 
        grid through/over which the morphogen flows.
    Output:
        returns  a list s of length n_node.  For i in
        0,..,max(s), np.where(s==i) is a list of nodes which are the same in
        terms of the Shh flow.
    """
    a = pair_mat(edges_to_nodes, cells , l) #form the pair matrix
    h=h_from_mat(a) # give paired nodes the same index
    return h # return the list of indices  

def living(cells): #SEEMS TO WORK
    """
    Takes in a cells object and returns an array a of length n_face.
    a[i]=1 iff cell i is alive, and a[i]=0 otherwise.
    """
    n_face=cells.mesh.n_face
    a = np.zeros(n_face)
    l= np.array(cells.mesh.face_id_by_edge)
    w= np.bincount(l)
    for i in range(len(w)):
        if (w[i]>0):
            a[i]=1
    return a  

def living_faces(living):
    return np.where(living==1)[0]

def cell_vertices(cells,i):#SEEMS TO WORK
    """
    Args:
        cells object
    Returns:
         The indices of the vertices of the cell as an array, 
         in consecutive order going clockwise? (I think).
    Remark: cells.mesh.vertices[i] is the start vertex of edge[i].
    """
    l= np.array(cells.mesh.face_id_by_edge)
    indices = np.where(l==i)
    return indices     #array
    
    
def centroid(cells,i):
    """
    Args:
        cells object 
        index i
    Returns:
        The coordinates of the centroid of the cell i.
    """
    indices= cell_vertices(cells,i)
    l=cells.mesh.vertices.T[indices] #mesh.vertices is of type float?
    if (len(l)==0):
        return -1 #return 
    else:
        return sum(l)/len(l)

def centroid2(cells,i):
    """
    Args:
        cells object 
        index i
    Returns:
        The coordinates of the centroid of the cell i.
    """
    indices= cell_vertices(cells,i)
    l=cells.mesh.vertices.T[indices] #mesh.vertices is of type float?
    if (len(l)==0):
        return np.array([-1,-1]) #return 
    else:
        return sum(l)/len(l)     


   
def centroids(cells):
    """
    Computes the centroids of all cells (or equivalently faces) living and dead
    and outputs them as a list l. l[i] is the centroid of face[i].  If face[i] 
    is dead, l[i]=[0].

    """  
    l=[] #stores the indices
    n_face= cells.mesh.n_face
    for i in range(n_face):
        if (living(cells)[i]):
            l.append(centroid(cells,i))
        else:
            l.append(np.array([0])) #used to be l.append([0])
    return np.array(l) 

def centroids2(cells):
    """
    Computes the centroids of all cells (or equivalently faces) living and dead
    and outputs them as a list l. l[i] is the centroid of face[i].  If face[i] 
    is dead, l[i]=[-1,-1].

    """  
    n_face= cells.mesh.n_face
    dummy = centroid2(cells,0)
    for i in range(1,n_face):
        dummy = np.vstack([dummy, centroid2(cells,i)]) 
    return dummy
"""
def centroids3(cells)
    living = ~cells.empty()
    m = len(living)
    centroids = np.array((m, 2))
    for k in range(m):
        if living[k]:
            bdy = cells.mesh.boundary(k)
"""           
            
            
            
            

def ind_i_centNZ2(cells , i): # index of the centroid of cell i in the list of centroids of LIVING cells
    """                      
    i is the index of a living cell
    cells is a cells object.
    centroids is a list l where l[i] = 0 if face i is dead and l[i] is an 
    element if R^2 otherwise.
    Let Reduced_centroids be the list centroids, with the zeros removed.
    This function outputs the index of the centroid of the living cell i in
    the list Reduced_centroids.
    We will use this function to set up our triangles for the FEM.
    
    """
    a = living(cells)
    b=np.cumsum(a)
    return int(b[i]-1)   

def triangles_v2(cells, edges_to_vtx_nodes, living_faces):
    """
   
    cells is a cells object.
    edges_to_vtx_nodes is the list s.t.edges_to_vtx_nodes[i] is the index of 
    the node corresponding to edge[i].
    
    
    The output is a list l
    l[i]= [a,b,c] where a,b,c are indices of nodes.
    The triple [a,b,c] corresponds to a triangle in the mesh.
    The shift 'const' comes from the way we have set up the node data.
    We first generate a list, 'vertex_nodes' of nodes generated from the vertices. 
    Then we compute the centroids of each face. centroids[i] is the coordinates
    of the centroid of face i, if face
    i is alive.  Otherwise it's zero.
    Then we reduce the zeros from the centroids list and append this list to
    the list of vertex nodes to get our list of nodes.
    Then, for a given edge i, the node index of the centroid of the corresponding
    face is given by 
    ind_i_centNZ(cells, face_id_by_edge[i]) + len(vertex_nodes).
    In Short:
    Every triangle in the mesh (a,b,c) corresponds to exactly one edge.
    The node index a, is the index of an edge, node index b is the index of the
    next edge around the (left) face and c is the node index of the centroid of
    the (left) face.
    """
    m = edges_to_vtx_nodes # renamed for convenience
    triList = [] # stores the triangles
    n = len(cells.mesh.edges.reverse) #go through all edges
     # m is the map from edges to nodes
    for i in range(n):
        face_number = cells.mesh.face_id_by_edge[i]
        k = int(np.where(living_faces==face_number)[0]) 
        dummy = [int(m[i]) , int(m[cells.mesh.edges.next[i]]), k + int(max(m)+1)]
        triList.append(dummy)
    return triList    
    
def build_vtx_nodes(edges_to_vtx_nodes, cells):
    """
    Makes the array of vertex nodes.
    Args:
        edges_to_vtx_nodes
        cells
    Returns:
        vtx_nodes
    TESTED.
    """ 
    m=int(max(edges_to_vtx_nodes))
    dummy=[]
    for i in range(int(m)+1):
        dummy.append( cells.mesh.vertices.T[np.where(edges_to_vtx_nodes==i)][0])  
    return np.array(dummy)    


def con_vtx_nodes(concentration_by_edge, edges_to_vtx_nodes):
    """
    Args:
        concentration_by_edge is the vector of concentrations by edge
        edges_to_vtx_nodes (same as edges_to_nodes)
    Returns:
        The vector of nodes which are roots of edges.  Sometimes called vertex
        nodes.
    Remark:
        The output will have the centroids of living faces appended to it to from
        the concentration vector, whose ith entry is the concentration at node i.
    """
    m=int(max(edges_to_vtx_nodes))
    concentration=np.zeros(m+1)
    concentration_by_edge
    for i in range(m+1):
        concentration[i] = concentration_by_edge[np.where(edges_to_vtx_nodes==i)][0] 
    return concentration.T
    

def nodes_generator(cells, edges_to_nodes , living_faces):
    """
    Generates the vector of nodes from the cells object and the edges_to_nodes
    map.
    Args:
        cells object
        edges_to_nodes
        living_faces
    Returns:
        the vector of nodes
    """       
    m=int(max(edges_to_nodes))
    dummy=[]
    for i in range(int(m)+1):
        dummy.append(cells.mesh.vertices.T[np.where(edges_to_nodes==i)][0])
    vtx_nodes=np.array(dummy) #vtx_nodes
    #print "shape vtx_nodes", np.shape(vtx_nodes)
    cents = centroids2(cells)
    #print "shape cents", np.shape(cents)
    liv_cents = cents[living_faces]
    #print "shape liv_cents" , np.shape(liv_cents)
    nodes = np.vstack((vtx_nodes, liv_cents))
    #print "shape nodes" , np.shape(nodes)
    return nodes, cents

def nodes_generator2(cells, edges_to_nodes , living_faces):
    """
    Generates the vector of nodes from the cells object and the edges_to_nodes
    map.
    Args:
        cells object
        edges_to_nodes
        living_faces
    Returns:
        the vector of nodes
    """       
    m=int(max(edges_to_nodes))
    dummy=cells.mesh.vertices.T[np.where(edges_to_nodes==0)]
    for i in range(1,int(m)+1):
        dummy = np.vstack([dummy, cells.mesh.vertices.T[np.where(edges_to_nodes==i)]])
    vtx_nodes=np.array(dummy) #vtx_nodes
    print("shape vtx_nodes", np.shape(vtx_nodes))
    cents = centroids2(cells)
    print("shape cents", np.shape(cents))
    liv_cents = cents[living_faces]
    print("shape liv_cents" , np.shape(liv_cents))
    nodes = np.vstack((vtx_nodes, liv_cents))
    print("shape nodes" , np.shape(nodes))
    return nodes, cents
 
def ids_by_x_pos(fe, centre, width):
    """
    Args:
        centre > 0
        width > 0
    Returns a list of the ids of faces whose centroid lies in the strip
    {(x,y): x in (centre - width/2,centre + width/2)}
    """
    ids_in_strip=[]
    liv_faces = fe.living_faces
    centroids = fe.centroids
    for k in liv_faces:
        if centroids[k] > centre - width/2.0  & centroids[k] < centre + width/2.0:
            ids_in_strip.append(k)
    return ids_in_strip

def node_ids_by_x_pos_right(fe,n):
    """
    Args:
        n> 0 is the number of bins.
        step = fe.cells.mesh.geometry.width / 2n
    Returns:
        A list l of length n
        l[i] is a list of the node_ids for which the x coord of the centroid 
        lies within [i*step, (i+1)*step]
    """
    l = []
    step = fe.cells.mesh.geometry.width /float(2*n)
    for k in range(n):
        dummy=[]
        l_lim=k * step
        r_lim = l_lim + step
        for j in range(len(fe.nodes)): #go through each node
            if (fe.nodes[j][0] > l_lim) & (fe.nodes[j][0] < r_lim):
                dummy.append(j)
        l.append(dummy)
    return l

def nodes_to_faces(fe, node_id):
    """
    node_id is the node id of a CENTROID
    """
    m = int(max(fe.edges_to_nodes))+1
    liv_faces = fe.living_faces
    return liv_faces[node_id - m]

def face_ids_by_x_pos_right(fe,n):
    """
    Args:
        n> 0 is the number of bins.
        step = fe.cells.mesh.geometry.width / 2n
    Returns:
        A list l of length n
        l[i] is a list of the living face_ids for which the x coord of the centroid 
        lies within [i*step, (i+1)*step]
    """
    l = []
    step = fe.cells.mesh.geometry.width /float(2*n)
    m = max(fe.edges_to_nodes)
    for k in range(n):
        dummy=[]
        l_lim=k * step
        r_lim = l_lim + step
        for j in range(int(m)+1,len(fe.nodes)): #go through each node which corresponds to a centroid
            if (fe.nodes[j][0] > l_lim) & (fe.nodes[j][0] < r_lim):
                if j - int(m) - 1  in fe.living_faces:
                    dummy.append(fe.living_faces[ j - int(m) - 1]) #EDITED
        l.append(dummy)
    return l


    
        
    
    
   



def level_sets_flash(fe , n):
    """
    Calculates the max and min concentration values among the nodes.
    Sets the range = max - min.
    Divides the range into n intervals of equal size.
    step = (max_c - min_c)/float(n).
    Interval k is (min + k*step, min + (k+1)*step)
    Classifies each node according to the interval to which its concentration
    belongs.
    Args:
        fe is a finite element object
        n is an integer, >=2
    RETURNS a list of arrays    
    Potentially a bit dodgy numerically.
    
    """
    dummy=[] #dummy[k] will be the array of indices belonging to interval k
    concentration = fe.concentration
    max_c= max(fe.concentration) #max concentration 
    min_c = min(fe.concentration) #min concentration
    if max_c > min_c + 0.1: #include the 0.1, so we don't divide by something small
        rescaled = (concentration - min_c) / (max_c - min_c) #rescaled[i] in [0,1]
        modified = np.modf(10 * rescaled)[0] # integer part of 10*rescaled
        for k in range(n):
            dummy.append(np.where(modified == k)[0]) #np.where(modified ==k) are the indices of nodes in bin k
    else:
        return 'error' #so we don't divide by something small.
    
def level_sets(fe , n):
    """
    Calculates the max and min concentration values among the nodes.
    Sets the range = max - min.
    Divides the range into n intervals of equal size.
    Interval k is (min + k*step, min + (k+1)*step)
    Classifies each node according to the interval to which its concentration
    belongs.
    Args:
        fe is a finite element object
        n is an integer, >=2
    RETURNS A LIST OF LISTS.  The kth list are the indices of the nodes in 
    interval k
    
    """
    dummy=[] #dummy[k] will be the array of indices belonging to interval k
    con = fe.concentration
    max_c= max(fe.concentration) #max concentration 
    min_c = min(fe.concentration) #min concentration
    step_size = (max_c - min_c)/float(n)
    for k in range(n):
        dummy2=[]
        l_lim = min_c+ k*step_size
        r_lim = min_c+ (k+1)*step_size
        for l in range(len(con)):
            if con[l] > l_lim & con[l] < r_lim:
                dummy2.append(l) #np.where(modified ==k) are the indices of nodes in kth interval
        dummy.append(dummy2)
    return dummy

def level_sets_right(fe , n):
    """
    Calculates the max and min concentration values among the nodes.
    Sets the range = max - min.
    Divides the range into n intervals of equal size.
    Interval k is (min + k*step, min + (k+1)*step)
    Classifies each node according to the interval to which its concentration
    belongs.
    Args:
        fe is a finite element object
        n is an integer, >=2
    RETURNS A LIST OF LISTS.  The kth list are the indices of the nodes in 
    interval k
    
    """
    dummy=[] #dummy[k] will be the array of indices belonging to interval k
    con = fe.concentration
    max_c= max(fe.concentration) #max concentration 
    min_c = min(fe.concentration) #min concentration
    step_size = (max_c - min_c)/float(n)
    for k in range(n):
        dummy2=[]
        l_lim = min_c+ k*step_size
        r_lim = min_c+ (k+1)*step_size
        for l in range(len(con)):
            if (con[l] > l_lim) & (con[l] < r_lim):
                if fe.nodes[l][0] > 0: #only store a node index if the node is to the right of zero
                    dummy2.append(l) #np.where(modified ==k) are the indices of nodes in kth interval
        dummy.append(dummy2)
    return dummy

def same_expo(fe,i, tol):
    """
    Args:
        i is the id of a living face
        tol > 0
    Returns the ids of the faces which are receiving a signal that is equal to
    that received by face i, up to a tolerance.
    """
    
    exposure_vector = fe.exposure()
    same_sig_ids = [] #stores ids of faces receiving the same signal
    for k in fe.living_faces:
        if exposure_vector[k] > exposure_vector[i] - tol/2 & exposure_vector[k] < exposure_vector[i] + tol/2:
            same_sig_ids.append(k)
    return same_sig_ids

def sharpness(cells,cc,a,b):
    """
    Args:
        cells is a cells object
        cc is a connected component.  That is, it is a list of face_ids which are
        path connected.
        a,b are edges in the boundary of cc.
    """
    length=cells.mesh.length
    euclid_dist=np.abs(cells.mesh.vertices.T[a],cells.mesh.vertices.T[b]) # dist root a to root b
    cw_path=path(cells,cc,a,b) #clockwise path edges
    acw_path = path(cells,cc,b,a) # anticlockwise
    cw_dist=sum(length[np.array(cw_path)])
    acw_dist=sum(length[np.array(acw_path)])
    bdy_dist = min(cw_dist, acw_dist)
    if (euclid_dist > 0.1):
        return bdy_dist / euclid_dist

def path(cells,cc,a,b):
    """
    Returns a list l of the edges joining the root of edge a to the root of edge
    b.
    the tip of edge l[i] is the root of edge l[i+1]
    l[0] is edge a and the tip of l[-1] (i.e. the last edge in the list) is 
    the root of edge b.
    
    (clockwise)
    Args:
        cells is a cells object
        cc is a connected component.  That is, it is a list of face_ids such that
        each element has another element in the list as a neighbour.
        a,b are edges in the boundary of cc.
    Edge a belongs to the boundary of the cc.  Then next[a] belongs to the boundary
    or nxt[rot[a]] belongs to the boundary.
    
    Remark:  If the path return has cycles at the end of it, the start edge
    and end edge do not belong to the same connected component of the boundary.
    """
    cc_boundary= cc_bdy(cc,cells) #edges which form the boundary of the cc
    edge_path=[] #stores the path from a to b
    count=0 #stores number of steps. If too large (> 100), we stop
    current_edge = a
    nxt= cells.mesh.edges.__next__ #shorthand
    rot= cells.mesh.edges.rotate #shorthand
    while((current_edge != b) & (count < 100) ):
        edge_path.append(current_edge)
        count+=1
        if nxt[current_edge] in cc_boundary: 
            current_edge = nxt[current_edge]
        else:
            current_edge = rot[nxt[current_edge]] #added nxt
    return edge_path

def path_finder(cells,edge_list,start_edge):
    """
    cells is a cells object
    edge_list is a list of edges
    start edge belongs to edge_list
    
    finds a sequence of edges from the edge list, one pointing to the next
    """
     #edges which form the boundary of the cc
     #stores the path
    current_edge = start_edge
    edge_path=[current_edge]
    nxt= cells.mesh.edges.__next__ #shorthand
    rot= cells.mesh.edges.rotate #shorthand
    while True:
        if nxt[current_edge] in edge_list: 
            current_edge = nxt[current_edge]
            edge_path.append(current_edge)
        elif rot[nxt[current_edge]] in edge_list:
            current_edge = rot[nxt[current_edge]] #added nxt
            edge_path.append(current_edge)
        else:
            break
    return edge_path
    
def path_distance(cells, path):
    """
    Args:
        cells object
        path, which is a list of edges.  Should also work if path is an array.
    Returns:
        The total length of the edges in path.     
    """
    edge_lengths=cells.mesh.length
    return sum(edge_lengths[np.array(path)])
        
        
"""
Connected components of the boundary.

Let A be a connected component of faces and let bdy(A) be the list of edges
that form the boundary of A.

A connected component of the bdy(A) is a subset U of bdy(A)  such that for
any i,j in U, there is a path of edges in U, joining i and j.

A connected component U of the boundary is maximal if there is no connected 
component of the boundary which contains U as a strict subset.

Claim:  Consider any connected component of faces.  If it has a boundary then it
 is the union of at most two maximal connected components.
 
 Note that if we are working with a toroidal mesh and the connected component of
 faces is the whole grid of cells, then it has no boundary

"""    
    
def cc_no_geo2(fe, prop):
    """
    Args:
        fe is a finite element object
        prop is a key to the dictionary cells.properties
        type(prop)=str
    Returns:
        The connected components of the faces having the property prop.
        In this function, connected components are in the sense of domains 
        in R^2.  The cell grid is interpreted as a domain of R^2.  That is
        the periodic boundary is taken to be empty.
        In this situation, the boundary of any connected component of faces
        will itself consist of only one maximal connected component.
    """
    a = fe.cells.properties[prop] #a[k]=1 if face k is of type A
    A_edges = []
    f_by_e = fe.cells.mesh.face_id_by_edge #shorthand
    type_A_faces = np.where(a==1)[0]
    counter = 0
    rev=fe.cells.mesh.edges.reverse #shorthand
    grid_bdy = fe.grid_boundary ##new
    for w in type_A_faces:
        face_bdy = fe.cells.mesh.boundary(w)
        for k in face_bdy:
            A_edges.append(k)
    for h in grid_bdy:
        if h in A_edges:   
            A_edges.remove(h) # 
    type_by_face= -1*np.ones(fe.cells.mesh.n_face) # vector to update
    while(len(A_edges) > 0):                #indices with the same value are in same cc
        l = A_edges.pop()
        if rev[l] in A_edges: #then f_by_e[l] and f_by_e[rev[l]] are connected
            g = f_by_e[l]
            h = f_by_e[rev[l]]
            #type_by_face[g]
            #type_by_face[h]
            if type_by_face[g] >=0:
                if type_by_face[h]>=0:
                    type_by_face[np.where(type_by_face == type_by_face[h])] = type_by_face[g]
                else:
                    type_by_face[h] = type_by_face[g]
            elif type_by_face[h]>=0: # give g h's type
                type_by_face[g] = type_by_face[h]
            else: #create a new type and give h and g the 
                type_by_face[g]=counter
                type_by_face[h] = counter
                counter += 1
            A_edges.remove(rev[l])
    cc = []
    for j in range(int(max(type_by_face)+1)):
        dummy = np.where(type_by_face == j)[0]
        if(len(dummy)>0):
            cc.append(dummy)
    for k in type_A_faces:
        if type_by_face[k]<0: #these are isolated type A cells
            cc.append([k])
    return cc # 
"""
def grid_bdyv2(cells):
    living=living(cells)
    liv_faces = living_faces(living)
    edges=[]
    for k in living_faces:
        bdy = cells.mesh.boundary(k)
        for l in bdy:
            edges.append(l)
"""   
def cc(fe_obj, prop):
    """
    Args:
        if type(prop)==str
        prop is a property.  type(prop)=str
        self.cells.properties is a dictionary
        prop is a key for the dictionary
        self.cells.properties[prop] [i] = 0 or 1
        face [i] is of type A iff  self.cells.properties[prop] [i] = 1
        len(self.cells.properties[prop]) = n_face
        Otherwise, prop is just the an array of zeros and ones.
    Returns:
        A list of arrays.  Each array is the collection of face_ids of a 
        connected component of faces which have the particular property in
        question.
    untested
    """
    fe=fe_obj.copy()  #use has_key(prop) for error message???
    if type(prop)==str:
        a = fe.cells.properties[prop] #a[k]=1 if face k is of type A
    else:
        a = prop    
    A_edges = []
    f_by_e = fe.cells.mesh.face_id_by_edge #shorthand
    type_A_faces = np.where(a==1)[0]
    counter = 0
    closed_boundary=[]
    rev=fe.cells.mesh.edges.reverse #shorthand
    for w in type_A_faces:
        face_bdy = fe.cells.mesh.boundary(w)
        for k in face_bdy:
            A_edges.append(k)
    for k in fe.grid_boundary:
        if (k not in fe.periodic_boundary):
            closed_boundary.append(k) 
    for m in closed_boundary: #when flow is toroidal, does nothing
        if m in A_edges:
            A_edges.remove(m) #after this, A_edges will be a list of all edges k for which face_id_by_edge(rev[k]) may be of type A
    type_by_face= -1*np.ones(fe.cells.mesh.n_face) # vector to update
    while(len(A_edges) > 0):     #indices in type_by_face with the same value are in the same cc
        l = A_edges.pop() #the elt l is removed from A_edges
        if rev[l] in A_edges: #then f_by_e[l] and f_by_e[rev[l]] are connected
            g = f_by_e[l] #shorthand
            h = f_by_e[rev[l]] #shorthand
            #type_by_face[g]
            #type_by_face[h]
            if type_by_face[g]>=0:
                if type_by_face[h]>=0: #if type_by_face[h]>0, reset all entries k with type_by_face[k] = type_by_face[h], to the value of type_by_face[g]
                    type_by_face[np.where(type_by_face == type_by_face[h])] = type_by_face[g]
                else:
                    type_by_face[h] = type_by_face[g]
            elif type_by_face[h]>=0:
                type_by_face[g] = type_by_face[h]
            else:
                type_by_face[g]=counter
                type_by_face[h] = counter
                counter += 1
            A_edges.remove(rev[l])   
    cc = []
    for j in range(int(max(type_by_face)+1)):
        dummy = np.where(type_by_face == j)[0] 
        if(len(dummy)>0): #we may have len(dummy)==0
            cc.append(dummy)
    for k in type_A_faces:
        if type_by_face[k]<0:
            cc.append(np.array([k])) #these will be the isolated type A cells with no type A neighbours 
    return cc #  
    """
    Consider the grid of cells without any assumed periodicity.  Then, connected
    components are just path connected regions in R^2.

    When we introduce periodicity, i.e. when the periodic boundary atribute is
    not empty, then we can have cells whcih are not in the same connected component
    of R^2, but are connected in our chosen geometry.  Eg. Consider two cells A,B 
    on opposite  sides of the grid which are disconnected in R2 but s.t we can find 
    w in bdy(A), rev(w) in bdy(B) and w in periodic boundary.
    Then, cells A and B would belong to the same connected component.
    """       
def cc_of_bdy(cells, bdy):
    """
    Args:
        cells is a cells object
        bdy is a list of boundary edges of a connected component of faces.
    Returns
        connected components of boundary edges as  list l.  l[i] is an array,
        the entries of which are a connected component of bdy edges.  The 
        array l[i] is ordered in the sense that for all suitable k,
        the tip of l[i][k] is the root of l[i][k+1]
    
    """  
    start_edges=[] # will store edges which are the start of maximal bdy components
    nxt=cells.mesh.edges.__next__ #shorthand
    rev=cells.mesh.edges.reverse #shorthand
    rot = cells.mesh.edges.rotate
    prev=cells.mesh.edges.prev
    edge_v=-1*np.ones_like(rev)
    for k in bdy:
        #if nxt[rev[k]] not in bdy: #does this make sense - maybe rev[nxt[rev[k]]]
        if (prev[k] not in bdy) & (rev[rot[k]] not in bdy): #WAS rev[rot[k]]
            start_edges.append(k)
    counter=-1      
    for i in start_edges:
        st_ed = i #start edge for this loop
        current_edge=i
        counter+=1
        edge_v[current_edge]=counter
        while True:
            if (nxt[current_edge] in bdy) & (nxt[current_edge]!=st_ed): #second part makes sure we're not back at the start
                current_edge=nxt[current_edge]
                edge_v[current_edge]=counter
            elif (rot[nxt[current_edge]] in bdy) & (rot[nxt[current_edge]]!=st_ed):
                current_edge=rot[nxt[current_edge]]
                edge_v[current_edge]=counter
            else:
                break
    ccs_of_bdy=[]
    for j in range(max(edge_v) +1):
        dummy  = np.where(edge_v==j)[0]
        if len(dummy)>0:
            ccs_of_bdy.append(dummy)
    return ccs_of_bdy


"""
cc_of_bdy will be used in tandem with cc_bdy2, from the file FELT.
For a given cc of faces (and correspoding fe object), cc_bdy2 returns
a list of the boundary edges of cc.  Then, the function cc_of_bdy will be used 
to return a list l s.t. l[i] is the array of edges which form a maximal connected
component of the boundary.

"""  

def cc_of_bdy2(cells, bdy):
    """
    Args:
        cells is a cells element object
        bdy is a list of boundary edges of a connected component of faces.
    Returns
        connected components of boundary edges as  list l.  l[i] is an array,
        the entries of which are a connected component of bdy edges.  The 
        array l[i] is ordered in the sense that for all suitable k,
        the tip of l[i][k] is the root of l[i][k+1]
    There's a problem.  In some situations, you never get a start edge even though 
    you should.
    """  
    start_edges=[] # will store edges which are the start of maximal bdy components
    nxt=cells.mesh.edges.__next__ #shorthand
    rev=cells.mesh.edges.reverse #shorthand
    rot = cells.mesh.edges.rotate
    prev=cells.mesh.edges.prev
    edge_v=-1*np.ones_like(rev)
    for k in bdy:
        #if nxt[rev[k]] not in bdy: #does this make sense - maybe rev[nxt[rev[k]]]
        if (prev[k] not in bdy):
            if (rev[rot[k]] not in bdy): #WAS rev[rot[k]]
                start_edges.append(k)
    counter=-1
    ccs_of_bdy=[]     
    print("start_edges", start_edges)
    for i in start_edges:
         dummy = path_finder(cells,bdy,i)
         ccs_of_bdy.append(dummy)
         for k in dummy:
             bdy.remove(k)
    return ccs_of_bdy

def cc_of_bdy3(fe, bdy):
    """
    Args:
        fe is a finite element object
        bdy is a list of boundary edges of a connected component of faces.
    Returns
        connected components of boundary edges as  list l.  l[i] is an array,
        the entries of which are a connected component of bdy edges.  The 
        array l[i] is ordered in the sense that for all suitable k,
        the tip of l[i][k] is the root of l[i][k+1]
    HACK.
    """  
    grid_boundary = fe.grid_boundary
    start_edges=[] # will store edges which are the start of maximal bdy components
    nxt=fe.cells.mesh.edges.__next__ #shorthand
    rev=fe.cells.mesh.edges.reverse #shorthand
    rot = fe.cells.mesh.edges.rotate
    prev=fe.cells.mesh.edges.prev
    edge_v=-1*np.ones_like(rev)
    for k in bdy:
        #if nxt[rev[k]] not in bdy: #does this make sense - maybe rev[nxt[rev[k]]]
        if (prev[k] in grid_boundary):
                start_edges.append(k)
    counter=-1
    ccs_of_bdy=[]     
    print("start_edges", start_edges)
    for i in start_edges:
         dummy = path_finder(fe.cells,bdy,i)
         ccs_of_bdy.append(dummy)
         for k in dummy:
             bdy.remove(k)
    return ccs_of_bdy
        
        
        
        
        
        
        
        
        
        
        
    
    
def cluster_data(fe,prop): 
    """
    Computes data for clusters of cells.  
    Treats faces as connected if they share a common edge (geometry taken into account)
    Args:
        fe is a finite element object.
        prop (str) may be a key for the dictionary fe.cells.properties.  
        fe.cells.properties[prop] is a vector, entry k is 1 if face k belongs
        has the property and it's 0 otherwise.
        Otherwise, prop can just be the array of zeros and ones.
        
        prop is either an array of zeros and ones of length n_face or it's
        the key to a property which has the same form.
        
    Returns:
        Two lists, face_ccs and bdy_lengths
        face_ccs is a list of arrays
        face_ccs[i] is an array whose entries are the faces which make up the
        ith connected component of faces having property prop.
        bdy_lengths[i] is a list of the lengths of the pieces of the boundary
        of the ith connected component of faces.  A cc of faces may have several
        pieces of boundary which are not connected.  So, len(bdy_lengths[i])
        can be >1.
    """
    #cluster_ids = np.array(fe.cells.properties['cluster']) #shorthand and make sure it's an array
    #area = fe.cells.mesh.area() #areas of each face in the grid
    face_ccs = cc(fe, prop) #connected components of faces.  
    bdy_lengths=[]
    #face_ccs_centroids maybe
    for k in face_ccs:
        bdy_edges_of_cc = cc_bdy2(k,fe) #computes the bdy edges the face_cc k.
        ccs_of_bdy =cc_of_bdy(fe.cells, bdy_edges_f_cc)
        temp = [] #temp[i] will be a list of lengths of the bdy pieces of the ith cc (i.e. face_ccs[i])
        for l in ccs_of_bdy:
            temp.append(path_distance(fe.cells,l))    
        bdy_lengths.append(temp)    
    return face_ccs, bdy_lengths   

def frontier_len(fe, prop):
    """
    prop has type string.
    prop is the name of a key in the dictionary fe.cells.properties
    prop will (at least initially) be s.t. fe.cells.properties['prop'][k]=1
    if face k is left of some suitable vertical line x=x_0
    Returns:
        the length of the boundary of region with all faces having the 
        property prop.
        also returns 1 if no cells have detached from the block and 0 otherwise    
    """
    #block_ids= np.array(fe.cells.properties[prop])
    face_ccs = cc(fe, prop)
    flag=1
    if len(face_ccs)==0:
        return -100, -100 #nonsense values as there are no connected components
    if len(face_ccs)>1:
        flag=0
    grid_bdy = fe.grid_boundary
    block_cc=face_ccs[0]
    for k in face_ccs:
        if len(k) > len(block_cc):
            block_cc=k #find the biggest connected component
    block_bdy = cc_bdy2(block_cc,fe) #should have only one ( maximal ) cc of bdy
    frontier_edges = block_bdy # we will remove edges which belong to the grid_bdy to leave the frontier edges
    for k in grid_bdy:
        if k in frontier_edges:
            frontier_edges.remove(k)
    return path_distance(fe.cells,frontier_edges), flag  

def living_relatives(fe):
    """
    Args:
        cells is a cells object.
    Returns:
        a list living_relatives of arrays of face ids.  living_relatives[i] is 
        an array of the face_ids of living faces which are descended from face[i]
    
    """
    props = fe.cells.properties['parent_group']
    m = max(props)
    relatives=[]
    for k in range(int(m+1)):
        relatives.append(np.where(props==k)[0])
    living_relatives=[]    
    for l in relatives:
        dummy=[]
        for s in l:
            if s in fe.living_faces:
                dummy.append(s)
        dummy=np.array(dummy) #convert to array
        living_relatives.append(dummy)
    return living_relatives

def current_relatives_data(fe, living_relatives):
    """
    All faces in the grid are descended from one of the original faces
    when we set up the parent groups.
    
    There may be a problem if we run the motion for a while to randomise and
    starting time after a certain number of iterations.  Provided we set the 
    parent groups at this time, it should not be a problem.  Need to be a bit
    careful though because living_relatives[i] may not be  the list of descendents
    of face i.  It will be if we do not randomize.

    ARGS:
        fe is a finite element object
        living_relatives is a list. living_relatives[i] is a list of the living
        face-ids which have the same ancestor.  We will refer to
        living_relatives[i] as group i.  
        
        I think living_relatives[i] are the
        decendents of f
    Returns:
        a list rel_data
        rel_data[i] = [a,b,c,d,e,f,g]
        where:
            a is the list of the centroids of the cells in living_relatives[i]
                i.e. a[k] is the centroid of cell living_relatives[i][k]
            b is the mean of a
            c is the  variance of a
            d if the list of exposures of cells in living_relatives[i]
            e is the mean of d
            f is the variance of d
            g is the list of connected components of faces among the relatives 

    """
    number_of_groups = len(living_relatives)
    n_face = fe.cells.mesh.n_face
    expo_by_face=fe.exposure_by_face()
    rel_data=[]#stores data about relatives
    for k in range(number_of_groups):
        dummy=[]
        prop = np.zeros(n_face)
        for l in living_relatives:
            for s in l:
                prop[s]=1 #build an appropriate property vector to pass to cluster_data
            dummy.append(fe.centroids[l]) #centroids 
            dummy.append(np.mean(fe.centroids[l],dtype=np.float64))
            dummy.append(np.var(fe.centroids[l],dtype=np.float64))
            dummy.append(expo_by_face[l])  #exposure to morphogen
            dummy.append(np.mean(fe.expo_by_face[l],dtype=np.float64))
            dummy.append(np.var(fe.expo_by_face[l],dtype=np.float64))
            dummy.append(cluster_data(fe, prop)[0]) #ccs_of_faces
        rel_data.append(dummy)
    return rel_data
    
            
def face_to_node(edges_to_nodes, living_faces, face_id):
    """
    edges_to_nodes[k] is the node index of the root of edge k
    living_faces is the array of faces which are alive
    face_id is the index of a living face
    The formula for node_id is a consequence of our method of constructing the
    node vector.
    """
    l = int(np.where(living_faces == face_id)[0])
    node_id = int(max(edges_to_nodes))+1 +l
    return node_id

#START OF  FILE   "FELT"      

def updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt):
    """
    TAKEN FROM FE_Sandbox2.
    h is the map from the indices of nodes to indices of free nodes.
    A is an  (m x m) empty matrix where m is the number of free nodes.
    This is the matrix which this function updates.
    old_alpha[i] is the concentration at node i, from the previous step.
    old_alpha may be referred to as long_alpha elsewhere. 
    nodes is the vector of coordinates of the nodes at the current time step.
    prev_nodes is the coordinates of the nodes at the previous time step.
    
    """
    for i in range(3):
        bv[h[triangle[i]]] += b(i,triangle,f,old_alpha,nodes,prev_nodes,dt)
        for j in range(3):
            A[h[triangle[i]]][h[triangle[j]]]+= I(i,j,triangle,nodes)
            A[h[triangle[i]]][h[triangle[j]]]+= K(i,j,triangle,nodes,v) 
            A[h[triangle[i]]][h[triangle[j]]]+=W(i,j,triangle,nodes, prev_nodes)   
            
            
def matrixM(triangle , nodes):
    """
    FROM FiniteElement
    Given a triangle and the list of node coordinates, this function returns
    the matrix required for the FEM.
    
    """
    va = [nodes[triangle[1]][0] - nodes[triangle[0]][0] , nodes[triangle[1]][1] - nodes[triangle[0]][1]] 
    vb = [nodes[triangle[2]][0] - nodes[triangle[0]][0] , nodes[triangle[2]][1] - nodes[triangle[0]][1]]
    M=np.array([va , vb])
    return M.T
    
def I(i,j,triangle ,nodes):
    """
    FROM FiniteElement
    i and i are two indices of triangle
    M is a matrix of numbers.
    Outputs the value of a certain integral in the FEM.
    
    """
    M = matrixM(triangle, nodes)
    d = np.linalg.det(M)
    if ( i - j ):
        return (1.0/24)* np.abs(d)
    else:
        return (1.0/12)* np.abs(d)
    
def K(i,j,triangle,nodes,v):
    """
    FROM FiniteElement
    i,j are indices of triangle.  So, i,j belong to {0,1,2}.
    M is a matrix
    v is a constant (the diffusion coefficient)
    This is a contribution to A[triangle[i],triangle[j]] when updating the 
    matrix with the part of the integral from triangle.
    """   
    M=matrixM(triangle,nodes)
    d = np.abs(np.linalg.det(M))
    return (1.0/2)*v*np.inner(nabPhi(M)[i],nabPhi(M)[j])*d      

def nabPhi(M):
    """
    FROM FiniteElement
    Args:
        M is the 2x2 FEM matrix for a triangle (p,q,r).
    Returns spatial gradients of phi_{p},  phi_{q}, phi_{r}, in that order 
     nabPhi[i] is the spatial gradient of Phi_{triangle[i]} (in the triangle p,q,r)
        
    """
    N = np.linalg.inv(M).T
    nabP_p = np.matmul(N , np.array([-1,-1]))
    nabP_q = np.matmul(N , np.array([1,0]))
    nabP_r = np.matmul(N , np.array([0,1]))
    return nabP_p, nabP_q , nabP_r


def W(i,j,triangle,nodes, previous_nodes):
    """
    FROM FiniteElement
    Args:
        i,j are indices of triangle.
        
    """
    M=matrixM(triangle, nodes)
    P0=(nodes[triangle[0]]-previous_nodes[triangle[0]]).T
    P1 = (nodes[triangle[1]]-previous_nodes[triangle[1]]).T
    P2 = (nodes[triangle[2]]-previous_nodes[triangle[2]]).T
    dummy = P0 + P1 + P2 + (nodes[triangle[j]]-previous_nodes[triangle[j]]).T
    return (1.0 / 24 )*np.abs(np.linalg.det(M))*np.inner(nabPhi(M)[i].T, dummy)
    

def b(i,triangle,f,old_alpha,nodes, prev_nodes,dt): 
    """
    FROM FiniteElement
    i is an index of triangle.  That is, i belongs to {0,1,2}
    f is the source vector of length = len(nodes)
    f is the vector coefficients of the source expressed in terms of hat functions
    old_alpha is the vector of coefficients alphas from the previous time step
    nodes are the current nodes
    prev_nodes are the nodes from the previous time step
    dt is the time step
    """
    dummy = I(i,0,triangle, nodes)*f[triangle[0]]
    dummy +=I(i,1,triangle, nodes)*f[triangle[1]]
    dummy +=I(i,2,triangle, nodes)*f[triangle[2]]
    dummy +=I(i,0,triangle, prev_nodes)*old_alpha[triangle[0]]
    dummy +=I(i,1,triangle, prev_nodes)*old_alpha[triangle[1]]
    dummy +=I(i,2,triangle, prev_nodes)*old_alpha[triangle[2]]
    return dummy  

def short_to_long(short_alpha, n_nodes ,h):
    """
    From FE_Sandbox
    Args:
        short_alpha is the vector of concentrations for the free nodes.
        n_nodes is the number of nodes 
        h is the periodicity map hfrom the indices of nodes to the indices of 
        the true_nodes.
    Returns:
        the vector of concentrations for the list of nodes.
    """
    long_alpha=np.zeros(n_nodes)
    for w in range(n_nodes):
        long_alpha[w] = short_alpha[h[w]]
    return long_alpha 

def periodicFEMstep_v2(nodes, prev_nodes, triangles,f, old_alpha, h , v ,dt):
    """
    From FE_Sandbox
    updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt)
    Args:
        nodes is the current list of node coordinates.
        prev_nodes is the list of the nodes at the previous time step.
        triangles is the list of triangles.  Each element is a list of 3
        node indices.
        f is the source.  f[i] is the rate? of morphogen production of face i
        old_alpha[i] is the concentration at node [i] at the previous time
        step.  Note that len(old_alpha) = len(nodes).
        h is the periodicity map
        v is the diffusion coefficient
        dt is the length of the time step.
    Returns:
        the concentration vector at the current time step
    """
    m = max(h)+1 # size of square matrix A and length of b
    A=np.zeros((m,m))
    bv=np.zeros(m)
    n_nodes = len(nodes)
    for w in triangles:
        #updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt):
        updater_Periodic_v2(w,A,bv,nodes,prev_nodes,f,h,old_alpha,v,dt)
    short_alpha= np.linalg.solve(A,bv)  #concentration vector for the free nodes
    long_alpha = short_to_long(short_alpha, n_nodes ,h) #concentration for the nodes
    return long_alpha   

def periodicFEMstep_v2_parallel(nodes, prev_nodes, triangles,f, old_alpha, h , v ,dt):
    """
    Does not work!!!!!!!!!!
    From FE_Sandbox
    updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt)
    Args:
        nodes is the current list of node coordinates.
        prev_nodes is the list of the nodes at the previous time step.
        triangles is the list of triangles.  Each element is a list of 3
        node indices.
        f is the source.  f[i] is the rate? of morphogen production of face i
        old_alpha[i] is the concentration at node [i] at the previous time
        step.  Note that len(old_alpha) = len(nodes).
        h is the periodicity map
        v is the diffusion coefficient
        dt is the length of the time step.
    Returns:
        the concentration vector at the current time step
    """
    m = max(h)+1 # size of square matrix A and length of b
    A=np.zeros((m,m))
    bv=np.zeros(m)
    n_nodes = len(nodes)
    processes=[]
    for w in triangles:
        #updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt):
        p = multiprocessing.Process(target=updater_Periodic_v2,args=[w,A,bv,nodes,prev_nodes,f,h,old_alpha,v,dt])
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    short_alpha= np.linalg.solve(A,bv)  #concentration vector for the free nodes
    long_alpha = short_to_long(short_alpha, n_nodes ,h) #concentration for the nodes
    return long_alpha

def periodicFEMstep_v2_parallel_v2(nodes, prev_nodes, triangles,f, old_alpha, h , v ,dt):
    """
    Does not work!!!!!!!!!!
    From FE_Sandbox
    updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt)
    Args:
        nodes is the current list of node coordinates.
        prev_nodes is the list of the nodes at the previous time step.
        triangles is the list of triangles.  Each element is a list of 3
        node indices.
        f is the source.  f[i] is the rate? of morphogen production of face i
        old_alpha[i] is the concentration at node [i] at the previous time
        step.  Note that len(old_alpha) = len(nodes).
        h is the periodicity map
        v is the diffusion coefficient
        dt is the length of the time step.
    Returns:
        the concentration vector at the current time step
    """
    m = max(h)+1 # size of square matrix A and length of b
    A=np.zeros((m,m))
    bv=np.zeros(m)
    n_nodes = len(nodes)
    processes=[]
    for w in triangles:
        #updater_Periodic_v2(triangle, A, bv , nodes,prev_nodes, f, h, old_alpha , v ,dt):
        p = multiprocessing.Process(target=updater_Periodic_v2,args=[w,A,bv,nodes,prev_nodes,f,h,old_alpha,v,dt])
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    short_alpha= np.linalg.solve(A,bv)  #concentration vector for the free nodes
    long_alpha = short_to_long(short_alpha, n_nodes ,h) #concentration for the nodes
    return long_alpha

def tri_exposure(triangle,nodes,concentration):
    """
    Args:
        triangle is a list of nodes which form a triangle in the fe mesh
        nodes is the list of node coordinates
        concentration[i] is the concentration at nodes[i]
    Returns:
        The amount of morphogen above triangle
    
    """
    M=matrixM(triangle, nodes) #map to canonical triangle
    dummy = np.abs(np.linalg.det(M))
    c0=concentration[triangle[0]]
    c1=concentration[triangle[1]]
    c2=concentration[triangle[2]]
    d2= (c2 - c1)/(4.0*np.sqrt(2)) + (c1 - c0)/(8.0) + c0/(3.0)+c1/(6.0)
    exposure = d2 * dummy
    return exposure

def tri_cover(cells,i):
    """
    Args:
        i is the index of a living_face.
    Returns:
        The indices of the triangles which cover face i.
    Note:
        The list of triangles is such that triangles[i] has edge i as its
        base.
        TESTED: SEEMS TO WORK.    
    """
    dummy = cells.mesh.boundary(i) #the edges which bound face i
    return dummy


def cc_bdy(cc,cells):
    """
    Args:
        cc is a connected component of faces.  An array of indices of
        face_ids s.t. all faces have the same property and are adjacent.
    This is meant for situations without periodicity.
    Tested with a toy example and it seems to work.
    Note:
        The boundary may be disconnected.  For arb a, b in boundary
        we may not always be able to go from a to b by following boundary edges.
        a path,
    """    
    edges=[] #will store all the edges of cells in the cc
    for w in cc:
        dummy = cells.mesh.boundary(w)
        for k in dummy:
            edges.append(k) 
    uni_edges = np.unique(edges) #get rid of repeats (I think this is pointless)
    bdy_of_cc=[]
    for m in uni_edges:
        if cells.mesh.edges.reverse[m] not in uni_edges:
            bdy_of_cc.append(m)
    return bdy_of_cc
            
def cc_bdy2(cc,fe):
    """
    Args:
        cc is a connected component of faces.  An array of indices of
        face_ids s.t. all faces have the same property and are adjacent.
    Is periodicity a problem for this function?
    Tested with a toy example and it seems to work.
        fe is a finite element object
    Returns:
        a list of the boundary edges of the cc of faces.
    Note:
        The boundary list may be disconnected in the following sense.  For arb
        a, b in boundary we may not always be able to go from a to b by 
        following boundary edges.
    """    
    edges=[] #will store all the edges of cells in the cc
    for w in cc:
        dummy = fe.cells.mesh.boundary(w)
        for k in dummy:
            edges.append(k) 
    uni_edges = np.unique(edges) #get rid of repeats
    bdy_of_cc=[]
    for m in uni_edges:
        if fe.cells.mesh.edges.reverse[m] not in uni_edges:
            bdy_of_cc.append(m)
        elif (m in fe.grid_boundary) & (m not in fe.periodic_boundary):
            bdy_of_cc.append(m)
    return bdy_of_cc

def cc_bdy3(cc,fe):
    """
    Args:
        cc is a connected component of faces.  An array of indices of
        face_ids s.t. all faces have the same property and are adjacent.
    Is periodicity a problem for this function?
    Tested with a toy example and it seems to work.
        fe is a finite element object
    Returns:
        a list of the boundary edges of the cc of faces.
    Note:
        The boundary list may be disconnected in the following sense.  For arb
        a, b in boundary we may not always be able to go from a to b by 
        following boundary edges.
    """    
    edges=[] #will store all the edges of cells in the cc
    for w in cc:
        dummy = fe.cells.mesh.boundary(w)
        for k in dummy:
            edges.append(k) 
    uni_edges = np.unique(edges) #get rid of repeats
    bdy_of_cc=[]
    for m in uni_edges:
        if fe.cells.mesh.edges.reverse[m] not in uni_edges:
            bdy_of_cc.append(m)
    return bdy_of_cc
           

    

def divideFE(fe):
        #PROBABLY JUNK
        cells = fe.cells
        cbyEdge=fe.concentration_by_edge()
        cbyCent = fe.concentration_by_centroid()
        properties = cells.properties 
        if 'age' in properties:
            ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0] 
        else:
            ready = np.where(~cells.empty() & (cells.mesh.area>=0.5*A_c))[0]
        #print "len(ready)" ,len(ready) , "n_face", cells.mesh.n_face
        edge_pairs=[]
        if len(ready)==0: #do nothing
            return fe
        for cell_id in ready:
            edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
        #print edge_pairs
        mesh, cbye, cbycent= _add_edges(cells.mesh,edge_pairs,cbyEdge,cbyCent) #Add new edges in the mesh
        #print "add edges"
        props = property_update(cells, ready) 
        cells2 = Cells(mesh,props) #form a new cells object with the updated properties. WHen we update properties, change to cells2 =Cells(mesh,prop)
        #print cells2.mesh.n_face, len(props['parent'])
        return build_FE(cells2, cbye, cbycent)       
       
        
#Rest from "divStuff2"
    

             
def same_ancestor(fe, cell_id):
    """
    For a given cell_id, this function returns the id's of living cells which 
    have a common ancestor.
    """
    parents = fe.cells.properties['parent']
    id_of_ancestor = parents[cell_id]
    same_ancestor = np.where(parents == id_of_ancestor)
    dummy=[]
    for w in same_ancestor: #if an elt of same_ancestor is alive, it is added
        if w in fe.living:  #to the list which we will return
            dummy.append(w)
    return dummy

def lewis(fe):
    """
    Lewis' law predicts that the average area of cells with j >=3 neighbours 
    divided by the average of all cell areas should equal (j-2)/4.
    This function computes the difference between these two quantities for
    appropriate values of j.
    Tested once on hexagonal grid and seems to work.
    
    Args:
        fe is a finite element object
    
    Returns:
        An array 'lewis' length max_nbs+1, where max_nbs be the largest number of 
        neighbours that any cell has.
        lewis[j] = ave area of cells with j neighbors / ave_area - (j-2)/4
        When j <3 or when there are no cells with j neighbours, we set
        lewis[j]=100.

    
    """
    edge_count = np.bincount(fe.cells.mesh.face_id_by_edge) #edge_count[i] is the number of neighbours cell i has  
    areas = fe.cells.mesh.area # array of areas of all cells (dead face ids have value zero)
    liv_face = fe.living_faces
    mean_area = sum(areas)/len(liv_face) # mean face area
    max_nbs = int(max(edge_count)) #max number of neighbours any cell has.
    lewis = np.zeros(max_nbs +1) 
    for k in range(max_nbs+1):
        dummy = areas[np.where(edge_count == k)] #np.where(edge_count == k) are the indices of faces with k neighbours
        if len(dummy)>0:
            lewis[k] = sum(dummy) / (len(dummy) * mean_area) -(k+1)/4.0 #error in Lewis' law
        else:
            lewis[k] = 100 #nonsense value
    return lewis

def a_w(fe):
    """
    Suppose that neighbours(j) is the number of neighbours that cell j has.
    Let j_neighbours be a list of the ids of j's neighbours.
    Suppose that for all k in j_neighbours, we compute neighbours(j) and take 
    the average and call this number mn(j).
    AW law (version 1) claims  that for all faces j
            neighbours(j) * mn(j) = 5n+8
    AW law (version 2) claims that
            neighbours(j)*mn(j) = 5n+6 + Var( neighbours(j) )
            where Var(n) = sum_{k = 3}^{infinty} (k - n_bar)^{2} f_{k}
            with n_bar being the mean neighbour number for all cells
            and f_k is the frequency of neighbour number k in the population. 
    Remark:  Works for toroidal meshs.  We assume that a for k in mesh.boundary(i),
    face_id_by_edge[reverse[k]] is the id of a neighbouring cell.  Depending on
    the geometry of the mesh, this may not be true at the grid_boundary.
    
    Remark: Tested for hexagonal faces and the first array is out by 2 everywhere
    and the second (corrected) aw array is zero everywhere, as expected.
    
    Args:
        fe is a finite element object.
        
    Returns:
        2 arrays of length n_face.
        The ith entry in the first array is the error in AW law version 1, for 
        face i.
        The ith entry in the second array is the error in AW law version 2, for 
        face i.
        
        
    """
    f_ids= fe.cells.mesh.face_ids  
    rev=fe.cells.mesh.edges.reverse #shorthand
    face_id_by_edge = fe.cells.mesh.face_id_by_edge # shorthand
    neighbours_by_face = np.bincount(fe.cells.mesh.face_id_by_edge)
    mean_neighbour_number = sum(neighbours_by_face) / float(len(fe.living_faces))
    max_neighbours = max(neighbours_by_face)
    f = np.zeros(max_neighbours+1)
    ff=np.bincount(neighbours_by_face)
    v = np.zeros(max_neighbours+1)
    for k in range(max_neighbours + 1):
        #f[k] = len((np.where(neighbours_by_face) == k)[0]) #number of cells with k neighbours
        v[k] = (k - mean_neighbour_number)*(k - mean_neighbour_number)*ff[k]
    a_w1 = np.zeros_like(f_ids)
    a_w2 = np.zeros_like(f_ids)
    ave_nbs_of_nbs = np.zeros_like(f_ids) #kth entry is average no. of neighbours of adjacent cells
    for k in fe.living_faces:
        dummy=[] #will store face_ids of face k's neighbours 
        mn=0.0 # will store the 
        bdy = fe.cells.mesh.boundary(k)
        for m in bdy:
            dummy.append(face_id_by_edge[rev[m]]) #f_ids of k's neighbours
        for g in dummy:
            mn+=neighbours_by_face[g]
        ave_nbs_of_nbs[k] = mn / len(dummy) #len(dummy ) never zero since k in living_faces
    for k in range(fe.cells.mesh.n_face):
        a_w1[k] = neighbours_by_face[k]*ave_nbs_of_nbs[k] - 5*neighbours_by_face[k] -8 #error in AW1 for each cell
        a_w2[k] = neighbours_by_face[k]*ave_nbs_of_nbs[k] - 5*neighbours_by_face[k] - 6  - v[neighbours_by_face[k]]
    return a_w1, a_w2

#Analysis by strip from here.
def sig_by_strip(FE_history, n):
    """
    Args:
        FE_history (list) is the output of a FE simulation.  FE_history[k] is
        a FE object.
        We will look at the nodes which have positive x coordinate.
        We split the range of possible x coords into n equally sized bins.
        We compute the average signal in each bin 
    Returns:
        Two arrays, M and V of shape len(FE_history) , n 
        M[i , j] is the mean signal in strip j at step i.
        V[i,j] is the variance in signal in strip j at step i
        
    """
    m = len(FE_history)
    M = np.ones((m,n))
    V =  np.ones((m,n))
    for k in range(m):
        fe = FE_history[k]
        binned_ids = node_ids_by_x_pos_right( fe  , n)
        for i in range(n):
            node_ids = np.array(binned_ids[i]) #ids in strip i, as an array
            node_ids=node_ids.astype(int)
            con_vect = fe.concentration[node_ids] # vector of concentrations in the strip
            M[k][i ] = np.mean(con_vect)
            V[k][ i ] = np.var(con_vect)
    return M, V

def lev_sets_mv(FE_history, n):
    """
    Args:
        At any time, we divide the range of concentration values into n equal
        pieces.
        We then bin the nodes (with positive x coord) by concentration level.
    Returns:
        Two arrays M,V of shape len(FE_history) , n 
        M[i , j] is the mean x coordinate for nodes in bin j at step i
        V[i, j] is the variance of the x coordinates of nodes in bin j, at step i.
    """
     
    m = len(FE_history)
    M = np.ones((m,n))
    V =  np.ones((m,n))
    for k in range(m):
        fe = FE_history[k]
        binned_ids = level_sets_right(fe , n)
        for i in range(n):
            node_ids = np.array(binned_ids[i]) #ids in strip i, as an array
            node_ids=node_ids.astype(int)
            coords= fe.nodes[node_ids] # coordinates of nodes in the strip
            x_coords = coords.T[0]
            M[k][i ] = np.mean(x_coords)
            V[k][ i ] = np.var(x_coords)
    return M, V