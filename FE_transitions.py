#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:49:34 2020

@author: andrewg
"""
from mesh import _T1, Edges
import numpy as np
from cells import *
from cells_extra import property_update
from Global_Constant import *
from run_select import division_axis, mod_division_axis

import matplotlib.pyplot as plt
# 2d array with a and b as columns 0 and 1, respectively
styles = {
        'seg': [{'color': 'blue'},
                {'color': 'red', 'dashes': [5,5]}],
        'nxt': [{'color': 'green'},
                {'color': 'orange', 'dashes': [5,5]}]
    }
def plot_edge(edges, vertices, reverse, ax=plt, **kwargs):
    for edge in edges:
        a = vertices[:,edge]
        b = vertices[:,reverse[edge]]
        pts = np.array([a,b]).T
        ax.plot(pts[0],pts[1],**kwargs)

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
    es = np.unique(edges//3*3) #for all j in edges, we obtain j , rot[j] , rot[rot[j]] in  
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
    


def T1(cells,eps=None):
        """
        Performs t1 transitions on the mesh.  Almost entirely the same as
        the first part of _transition from mesh.  
        Args:
            cells is a cells object
            eps >0 is a tolerance.  We perform type 1 transitions on edges which are
            shorter than eps.
        Returns:
            (updated) cells object.
            
        """
        props = cells.properties
        if eps==None:
            eps=0.1 #include a default epsilon value
        mesh = cells.mesh
        edges = mesh.edges #shorthand
        half_edges = edges.ids[edges.ids < edges.reverse]  # array of indices of edges with id < reverse_id
        dv = mesh.edge_vect.take(half_edges, 1) #dv[i] is the vector from root of edge half_edges[i] to its tip. 
        short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps]) #why did they make this a set?
        ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps] #edges which are shorter than the tolerance eps>0
            
        # if the set is empty (no T1 transition occurring), then return the cells as they are
        if not short_edges:
            return cells, ids_t1

        # make a copy of the lists to be changed
        reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()

        # edges obtained from `edges` by CW rotation around their roots
        rotate = edges.rotate

        # Do T1 transitions
        # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
        # and delay to the next timestep if necessary.
        # A better approach would be to take multiple partial timesteps.

        # (A) how about copies here???
        boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
        while short_edges:
            edge = short_edges.pop()    # (A) picks the last element of 'short_edges' and removes it from list
            if edge in boundary_edges:
                edge = reverse[edge]

            # (A) _T1 returns all the half-edges adjacent to the one being rotated
            (neighbours,_) = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
            # (A) these are excluded from list of short edges
            for x in neighbours:
                short_edges.discard(x)
        mesh = mesh.copy() # copy mesh 
        
        # set attributes of cells
        mesh.edges = Edges(reverse) # (A) why the reverse edges?
        mesh.vertices = vertices
        mesh.face_id_by_edge = face_id_by_edge
        cells = Cells(mesh, props) # make a cells object
        # T1 transitions do not change the concentration_by_edge or concentration_by_centroid
        # (A) and this is wrong, isn't it?
        return cells, ids_t1
 
       
def rem_collapsed(cells,c_by_e):
    """
    Removes collapsed faces:
    
        Copied overall structure from function in mesh.
        Uses modified _remove function.
        
    Args:
        cells is a cells object
        c_by_e is the concentration by edge
    Returns:
        (updated )cells, (updated ) c_by_e
        
    Remark:
        We do not need to update the concentration_by_centroid.  We would only
        be changing values which are not used when we call build_FE.
        WHAT DOES THIS MEAN?
        I UNDERSTAND THAT C_BY_C IS NOT NEEDED, WHAT IS THE BUILD_FE THING?
    
    """
    props=cells.properties #to pass on to the new object
    edges = cells.mesh.edges
    reverse= cells.mesh.edges.reverse
    vertices = cells.mesh.vertices
    face_id_by_edge = cells.mesh.face_id_by_edge
    rotate=cells.mesh.edges.rotate
    try:
        reverse = cells.mesh.edges.reverse
        reverse.setflags(write = True)
    except ValueError:
        reverse=cells.mesh.edges.reverse.copy()
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        if not len(two_sided):
            break
        count = 0
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            print("two_sided = ", two_sided, "    reverse[two_sided] = ", reverse[two_sided], "    nxt[two_sided] = ", nxt[two_sided], "\n")
            print("rotate[two_sided] =", rotate[two_sided])
            print("reverse[rotate[two_sided]] =", reverse[rotate[two_sided]])
            print("reverse[reverse[rotate[two_sided]]] =", reverse[reverse[rotate[two_sided]]])
            print("")
            print("nxt[two_sided] = ", nxt[two_sided])
            print("rotate[nxt[two_sided]] = ", rotate[nxt[two_sided]])
            print("reverse[rotate[nxt[two_sided]]] = ", reverse[rotate[nxt[two_sided]]])
            print("")
            fig, ax = plt.subplots(1,2)
            ax[0].set_title("Before")
            for i in [0,1]:
                segments = np.array([
                        reverse[two_sided[i]],
                        # reverse[rotate[two_sided[i]]],
                        # reverse[nxt[two_sided[i]]],
                        reverse[rotate[nxt[two_sided[i]]]]
                    ]).T
                segments_nxt = np.array([
                        nxt[reverse[two_sided[i]]],
                        # nxt[reverse[rotate[two_sided[i]]]],
                        # nxt[reverse[nxt[two_sided[i]]]],
                        nxt[reverse[rotate[nxt[two_sided[i]]]]]
                    ])
                plot_edge(segments, vertices, reverse, ax=ax[0], **styles['seg'][i])
                plot_edge(segments_nxt, vertices, reverse, ax=ax[0], **styles['nxt'][i])
            count += 1
            print("Do something here...")
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]

            ax[1].set_title("After")
            nxt = rotate[reverse]
            for i in [0,1]:
                segments = np.array([
                        reverse[two_sided[i]],
                        # reverse[rotate[two_sided[i]]],
                        # reverse[nxt[two_sided[i]]],
                        reverse[rotate[nxt[two_sided[i]]]]
                    ]).T
                segments_nxt = np.array([
                        nxt[reverse[two_sided[i]]],
                        # nxt[reverse[rotate[two_sided[i]]]],
                        # nxt[reverse[nxt[two_sided[i]]]],
                        nxt[reverse[rotate[nxt[two_sided[i]]]]]
                    ])
                plot_edge(segments, vertices, reverse, ax=ax[1], **styles['seg'][i])
                plot_edge(segments_nxt, vertices, reverse, ax=ax[1], **styles['nxt'][i])
            plt.show()
            plt.savefig("T2_step-%d.png"%(count))
        prev_face_id_by_edge = face_id_by_edge
        reverse, vertices, face_id_by_edge,c_by_e = _remove(two_sided, reverse, vertices, face_id_by_edge, c_by_e)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        exit()
            #print "ids_removed", ids_removed
    # if ~(ids_t1==np.delete(ids_t1,ids_removed)):
    #     print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)
    # reverse.setflags(write = False)
    # cells.mesh.edges.reverse.setflags(write = False)
    mesh = cells.mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    cells = Cells(mesh , props)
    return cells, c_by_e #added ()

def divide(cells, c_by_e, c_by_c,ready):#Tested.  Add division of seq object if necessary #copied from divide2 and modified
    """
    Args:
        cells is a cells object
        c_by_e is the concentration by edge
        c_by_c is the concentration by centroid
        ready is a list of cell ids which are ready to divide.
    Returns:
        (updated) cells, (updated) c_by_e, (updated) c_by_c
    Note that the cells.properties dictionary is also updated.
    
    """
    properties = cells.properties
    edge_pairs=[]
    #if 'age' in properties:
    #    ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0] 
    #else:
    #    ready = np.where(~cells.empty() & (cells.mesh.area>=0.95*A_c))[0] #added 0.5 
    if len(ready)==0: #do nothing
        return cells, c_by_e, c_by_c
    for cell_id in ready:
        edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
            #edge_pairs.append(mod_division_axis(cells.mesh,cell_id))
            #print edge_pairs 
    mesh, c_by_e, c_by_c= _add_edges(cells.mesh,edge_pairs,c_by_e,c_by_c) #Add new edges in the mesh
    #print "add edges
    props = property_update(cells,ready) 
    cells = Cells(mesh,props)
    return cells, c_by_e, c_by_c
    