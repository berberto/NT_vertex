# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np


class Cells(object):
    def __init__(self, mesh, properties=None):
        self.mesh = mesh
        self.properties = properties or {}

    def copy(self):
        # mesh is 'immutable' (so we can cache shared computations) => no need to copy it
        return Cells(self.mesh, self.properties.copy())

    def __len__(self):
        return self.mesh.n_face

    def empty(self):
        # self.mesh._edge_lookup[self.mesh.area<0]=-1 ###########anyadido por mi!!!!!!!!! 
        return self.mesh._edge_lookup == -1

    def by_face(self,property_name, boundary_property_name=None): #modified these because we use vectors for all properties
        value = self.properties[property_name]
        if self.mesh.has_boundary():
            #value = make_array(value, len(self)) #old way #see below for definition of make array
            value = _modified_make_array2(value, self.mesh.n_face) # I added this
            boundary_value = self.properties[boundary_property_name] if boundary_property_name else 0.0
            value[self.mesh.boundary_faces] = boundary_value
        return value
    
    def by_edge(self, property_name, boundary_property_name=None):
        value_by_face = self.by_face(property_name, boundary_property_name)
        return value_by_face.take(self.mesh.face_id_by_edge)
    """
    def by_face(self, property_name, boundary_property_name=None):
        value = self.properties[property_name]
        if self.mesh.has_boundary():
            #value = make_array(value, len(self)) #old way #see below for definition of make array
            value = _modified_make_array(value, self.mesh.n_face) # I added this
            boundary_value = self.properties[boundary_property_name] if boundary_property_name else 0.0
            value[self.mesh.boundary_faces] = boundary_value
        return value #looks like 

    def by_edge(self, property_name, boundary_property_name=None): #PROBLEM WITH THIS
        value_by_face = self.by_face(property_name, boundary_property_name)
        #if not hasattr(value_by_face, 'shape'):  # old way.  What's wrong with this?
           #return value_by_face
        if not hasattr(value_by_face, 'length'):  # CHANGED THIS BIT
            return value_by_face 
        return value_by_face.take(self.mesh.face_id_by_edge)
# old? by_edge does not seem to work on the mac. 
    """ 

def make_array(value, n):
    if hasattr(value, 'shape'): #False if value is a scalar, True if it's a vector
        return value
    expanded = np.empty(n, type(value)) #create a vector, entries garbage of same type as value 
    expanded.fill(value) #set each entry equal to value
    return expanded
#takes in an array and an integer.  Returns the array if it is not a scalar.  Otherwise
    #it returns an array of n entries, each one equal to the input scalar.
    """
    I think there is perhaps a problem with make_array().
     
    The function make array should, when value is a number, return an array of length n
    with each entry equal to value.  
    However, it does not seem to do this.
    
    Suppose we call
    make_array(cells.properties['K'], cells.mesh.n_face). 
     A cells object has an attribute properties,
    which is a dictionary of constants associated with the mesh. Let 'K' be
    a key for the dictionary, i.e. we are saying one of the constants is called 'K'.  
    When we call cells.properties['K'] we get a number. This number has type
    numpy.float64.
    
    When w is of type np.float64, the call hasattr(w,'shape') returns True.  This 
    explains why make_array(w,18) for example returns a number instead of an array
    when w has type numpy.float64.  We would like an array of 18 copies of w.
    
    """
def _modified_make_array(value,n):
    """
    Doesn't work properly.
    
    """
    if hasattr(value,'length'):
        return value
    expanded = np.empty(n, type(value)) #create a vector, entries garbage of same type as value 
    expanded.fill(value) #set each entry equal to value
    return expanded

def _modified_make_array2(value,n):
    """
    A modified vesrion of make array which performs correctly when value is of
    type np.float64.

    """
    if hasattr(value, 'shape') and bool(np.shape(value)) : #False if value is a scalar, True if it's a vector
        return value
    expanded = np.empty(n, type(value)) #create a vector, entries garbage of same type as value 
    expanded.fill(value) #set each entry equal to value
    return expanded
    



