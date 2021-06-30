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
            value = make_array(value, self.mesh.n_face) # I added this
            boundary_value = self.properties[boundary_property_name] if boundary_property_name else 0.0
            value[self.mesh.boundary_faces] = boundary_value
        return value
    
    def by_edge(self, property_name, boundary_property_name=None):
        value_by_face = self.by_face(property_name, boundary_property_name)
        return value_by_face.take(self.mesh.face_id_by_edge)


def make_array(value,n):
    """
    Returns `value` if it is an array,
    otherwise return an array of lenth `n` with all
    elements equal to `value`.

    """
    if hasattr(value, 'shape') and bool(np.shape(value)) : #False if value is a scalar, True if it's a vector
        return value
    expanded = np.empty(n, type(value)) #create a vector, entries garbage of same type as value 
    expanded.fill(value) #set each entry equal to value
    return expanded
    



