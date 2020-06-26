# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from scipy.spatial import Voronoi,voronoi_plot_2d

from permutations import cycles, inverse
from mesh import Edges, Mesh, Torus, Cylinder, Plane


def swap_if(condition, a, b):
    for i, cond in enumerate(condition):
        if cond:
            a[i], b[i] = b[i], a[i]
"""
Arguments are two n dimensional arrays and an n dimensional array
 of boolean values. The statement 'i, cond in enumerate(condition):' gives us 
 access to the index and the corresponding value
 of condition.  If, for an index i, condition[i] is True, we swap the ith 
 elements in the arrays a,b.  Note, each element in a,b can itself be an array.
"""

def _correct_orientation(vor):
    face_centres = vor.points[vor.ridge_points] #pairs of voronoi pts between
    #which there's an edge. (Edge is perp to line joining pts)
    vertex_id_pairs = vor.ridge_vertices
    vertex_positions = vor.vertices[vertex_id_pairs] #not always well defined.
    d = lambda pair: pair[:, 1] - pair[:, 0] 
    swap_if(np.cross(d(face_centres), d(vertex_positions)) < 0,
            vertex_id_pairs[:, 0], vertex_id_pairs[:, 1])
"""
function 'd' takes an array of shape(n,2,2), i.e. n pairs of vectors in R^2 and
 for each pair of vectors, it computes the vector joining the 0th vector to the
 first vector. It returns an array of shape (n,2).
 
 When applied to two 2d vectors a,b, np.cross(a,b) returns a the determinant of
 the matrix with a as the first row and b as the second row. Given an edge in a 
 voronoi diagram, there is a corresponding pair of points [p_{i},p_{j}]and a 
 pair of vertices [v_[k],v_[l]].  Note that each 
 p_{i}, v_{j} are in R^2).
'vor.points[vor.ridge_points]' is the list of all pairs of pts (vectors in R^2)
 between which there is an edge.
'vor.vertices[vor.ridge_vertices]' is the list of all pairs of vertices (again 
vectors in R^2) which are joined by an edge.
It seems that the numbering of these lists is such that for an index k
 the pair 'vor.points[vor.ridge_points][k]' and the pair 
 'vor.vertices[vor.ridge_vertices][k]' correspond to the same edge.

 
Given an edge, the pairs of vertices define the edge completely.  Each one of 
the Voronoi points will be inside one and only one cell.  Given a Voronoi 
point p=(x,y), we may want to identify the edges which form the boundary of
the cell which contains it.  The _correct_orientation function allows us to 
do this.
The function _correct_orientation ensures that for any index k, if we travel 
from the first vector in the pairing  'vor.vertices[vor.ridge_vertices][k]' to 
the second vector, then the second component of 'vor.points[vor.ridge_points]' 
will always be on the left.  This means that we can easily access the edges
 which form the boundary of the cell containing a given point p=(x,y), say.  We
 just have to determine the set of indices U for which p is in the second
position of 'vor.points[vor.ridge_points][k]' for all k in U.  The set 
{vor.vertices[vor.ridge_vertices][k]: k in U } is a collection of all pairs of 
vertices s.t. for any pair, if you travel from the first vertex to the second,
then p will be on your left.  For each pair of vertices in
{vor.vertices[vor.ridge_vertices][k]: k in U }
there is an edge, and these edges form the cell's boundary.
"""



def attach(data, from_cols, to_cols):
    ix = data[from_cols].argsort().argsort()
    return data[to_cols].argsort()[ix]
"""

This function acts on an array data in the form of a matrix, i.e. shape(data)=
(a,b).  In fact, when this function is used,data will be a record array, with 
its columns named so that 
that column zero can be accessed as an array with the command data.col0name,
where 'col0name' is the name of column 0.
The arguments 'from_cols' and 'to_cols' are lists of names of columns.  When 
execute
data[from_cols]
an array of shape (a,len(from_cols)) is returned.  This array is made up of the
 columns in the list 'from_cols'
Executing  data[to_cols] gives us an array of shape (a, len(to_cols)), made up
of the expected columns.
The call data[from_cols].argsort() returns the indices that would sort the
array.  This will be a permutation of (0,..,a-1), p, say.  The sort is 
performed by the
first column, then by the second etc.
ix in the function code is just the inverse p^{-1} of p.
data[to_cols].argsort() gives the permutation q that sorts the second array.
The function attach returns the composition q ( p^{-1} ).



When data = edge table, from_cols=['face','vertex2'] and 
to_cols=['face','vertex'], the attach(data,from_cols,to_cols) is a permutation,
the index at the ith place will be the index of the 'next' edge which comes 
after edge[i] around a face.

'face': A face is a domain in the voronoi diagram separated from other domains
by ridges.

'next':
    Each edge has a direction.  Around a given face, the edge is either anti-
    clockwise or clockwise.  If edge i is a clockwise edge around face k, 
    with start vertex u and end vertex v.  The next edge is the 'clockwise' 
    edge starting at vertex v.
    
For a given directed edge, the 'next' edge is unambiguously defined.


With p and q as defined above, edge_table[p] and edge_table[q] will be such 
that edge_table[q][k] will be the next edge after edge_table[p][k]. 
This follows directly from the structure of each edge element.
( Note that
p and q are permutations a k is an index.)

Consider an edge element
e=(3,1,5,9,0,4).
e.face=3
e.face2=1
e.vertex=5
e.vertex2=9
e.region=0 #not important here
e.region=4. # not important here
e.face is the index of the voronoi point (also what we're calling face number) 
on the right of the directed edge e.
e.face2 is the index of the voronoi point (also what we're calling face number) 
on the left of the directed edge e.
e.vertex is the index of the starting vertex.  
e.vertex is the index of the end vertex of the edge.
Given this structure, it follows that edge_table[q][k] will be the next edge 
after edge_table[p][k]. 

"""


def voronoi(centres):
    vor = Voronoi(centres)
    vor.ridge_vertices = np.array(list(zip(*vor.ridge_vertices))).T #simple np.array 
    #call is too slow..
    # edges are not consistently oriented so we fix this
    _correct_orientation(vor) 
    return vor
"""For a given input of points in R^2, this function returns Voronoi data
wherein the pairs of vertices and pairs of points are consistently orientated, 
so that we can easily determine the boundary of a cell which contains a given 
input point.
"""





def _edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs):
    if region_id_pairs is None:
        region_id_pairs = np.zeros_like(face_id_pairs)
    edge_data = [face_id_pairs, vertex_id_pairs, region_id_pairs]
    # add reverse orientation of each edge
    flip = lambda pair: np.roll(pair, 1, axis=-1)
    edges = np.vstack([np.hstack(edge_data), np.hstack(list(map(flip, edge_data)))])
    dtype = {'names': ['face', 'face2', 'vertex', 'vertex2', 'region', 'region2'], 'formats': ['int64']*6} #was 'int64'
    return edges.view(dtype).view(np.recarray)
"""
Confused.
The point of this function seems to be to collect Voronoi data into a record 
array format so that we can access the eg. face_id_pairs by typing _.face.
When I create a Voronoi object (vor) with some example input points (pts) and 
I compute
w=_edge_table(vor.ridge_points,vor.ridge_vertices,None),
w.face does not give me the vor.ridge_points back, which I expect it to.

The problem in edge_table is that when edges is created in the function
_edge_table, edges[i][k] (k in 0,1,..,5 , i in 0,..,Nedges-1) is converted to 
'int64' type.  The output of the corresponding number from voronoi is 'int32'.

Then, when we use the dtype to create the recarray, we should use 'int64' as the
format instead of 'int32'.  Using 'int32' results in an edge_table which is 
filled with garbage.

When the recarray is formed, all the bits in the array "edges" are 'lined up.' When the format is
int32,the first 32 bits will be (the first element of ) face, the second 32 bits
 will be the first element of 'face2'.  Given that the type of edges[i][k] is int64,
 the bits that make up edges[i][k] are split in two and each is interpreted as
an integer.  So we two numbers from one.  Hence the 'padding' we see when we run 
_edge_table with an example.  When we try the function on voronoi output from a 
small number of cells, one of the 'new' numbers may actually be correct and the 
other seems to be zero or -1 depending on the sign of the old number.



"""

def _modified_edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs):
    if region_id_pairs is None:
        region_id_pairs = np.zeros_like(face_id_pairs)

    edge_data = [face_id_pairs, vertex_id_pairs, region_id_pairs]
    # print(edge_data[:4])
    # add reverse orientation of each edge
    flip = lambda pair: np.roll(pair, 1, axis=-1)
    edges = np.vstack([np.hstack(edge_data), np.hstack(list(map(flip, edge_data)))]) 
    dtype = {'names': ['face', 'face2', 'vertex', 'vertex2', 'region', 'region2'], 'formats': ['int64']*6}
    # # print(dtype)
    # print(edges[:4], "\n")
    # # print(np.shape(edges))
    # print(edges.view(dtype)[:4], "\n")
    # # print(np.shape(edges.view(dtype)))
    # print(edges.view(dtype).view(np.recarray)[:4], "\n")
    # # print(np.shape(edges.view(dtype)))
    # import sys
    # sys.exit()
    return edges.view(dtype).view(np.recarray)

def build_mesh(vertex_positions, geometry, face_id_pairs, vertex_id_pairs, region_id_pairs=None, boundary_face_ids=None):
    edge_data = _edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs)
    fundamental_region = edge_data.region == 0
    edge_data = edge_data[fundamental_region]
    # build 'half-edge' representation
    nxt = attach(edge_data, ['face', 'vertex2'], ['face', 'vertex'])  # 'next' edge
    edge_data.region = -edge_data.region2
    reverse = attach(edge_data, ['face', 'face2', 'region'], ['face2', 'face', 'region2'])  # 'reverse' edge

    order = cycles(nxt[reverse])[0]  # order so that edges around a vertex are consecutive
    reverse = inverse(order)[reverse[order]]  # group conjugation of reverse by order
    vertices = vertex_positions[edge_data.vertex[order]].T.copy()

    edges = Edges(reverse)
    face_id_by_edge = cycles(edges.next)[1] 
    
    """
    I think the following comments are irrelevant....
    
    Why don't we use a new version of nxt, conjugated by order as for reverse?  
    i.e. nxt = inverse(order) nxt[order].
    Then face_id_by_edge would be defined without any problems at the boundary.
    In the above method, we pass the 'reverse' permutation to edges, we calculate the 
    next edge in a clever way.  This method assumes that all vertices have the 3 outward
    edges and 3 inward edges. (Very likely to be true for internal vertices.)  The method 
    maygo wrong (in theory) when we come upon a vertex on the
    boundary, where are not in the same situation as internal vertices. 
    I think that the permutation nxt should be passed to edges instead.  The 
    information does not need to be recalculated by a somewhat dubious method.
    
    """

    boundary = None
    if boundary_face_ids is not None:
        face = edge_data.face[order]
        boundary_edges = np.any([face == face_id for face_id in boundary_face_ids], axis=0)
        boundary = np.unique(face_id_by_edge[boundary_edges])

    return Mesh(edges, vertices, face_id_by_edge, geometry, boundary_faces=boundary)

def _modified_build_mesh(vertex_positions, geometry, face_id_pairs, vertex_id_pairs, region_id_pairs=None, boundary_face_ids=None):
    edge_data = _modified_edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs)
    fundamental_region = edge_data.region == 0
    edge_data = edge_data[fundamental_region]

    # build 'half-edge' representation
    nxt = attach(edge_data, ['face', 'vertex2'], ['face', 'vertex'])  # 'next' edge
    edge_data.region = -edge_data.region2
    reverse = attach(edge_data, ['face', 'face2', 'region'], ['face2', 'face', 'region2'])  # 'reverse' edge
       # region, region2 are included so that the corresponding bdy edges are identified as reverse edges.
    order = cycles(nxt[reverse])[0]  # order so that OUTWARD edges around a vertex are consecutive
    reverse = inverse(order)[reverse[order]]  # group conjugation of reverse by order
    vertices = vertex_positions[edge_data.vertex[order]].T.copy() #start 

    edges = Edges(reverse)
    face_id_by_edge = cycles(edges.next)[1] #this is a list of integers with repeats
                                            # each integer corresponds to a cycle of edges.next
                                            #    i.e. each integer corresponds to a cell.

#Possible problem.  The edges.next array is created by making the assumption that each vertex has three
#outward edges and 3 inward edges.  This is not true at the boundary.

    boundary = None
    if boundary_face_ids is not None: #boundary_face_ids is a list of indices (of voronoi pts) s.t.
        face = edge_data.face[order] #puts the edge_data in the order specified by the permutation 'order'
        boundary_edges = np.any([face == face_id for face_id in boundary_face_ids], axis=0) #index of boundary edges
        boundary = np.unique(face_id_by_edge[boundary_edges])#see comment below 

#boundary_face_ids is a list of indices (of voronoi points) such that if for an
#edge its 'face' number belongs to the list, the the edge is on the boundary.
#face_id_by_edge[boundary_edges] gives us a list of numbers (with repeats), each corresponding
#to a cell, such that each cell has a boundary edge.  The np.unique command gets 
#rid of the repeats.


    return Mesh(edges, vertices, face_id_by_edge, geometry, boundary_faces=boundary)

def toroidal_voronoi_mesh(centres, width, height):
    """Returns a Mesh data structure on a torus constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions in the interval [-width/2,width/2]*[-height/2,height/2]
        width: a float giving periodicity in the x-direction
        height: a float giving periodicity in the y-direction
        Returns:
        A Mesh data structure.
    """
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    vor = voronoi(centres_3x3)

    N_cell = len(vor.points)//9
    region_id_pairs = vor.ridge_points//N_cell-4  # fundamental region is 0 WHY 4!!!!!!
    face_id_pairs = vor.ridge_points % N_cell  # idx mapped to fundamental region

    return build_mesh(vor.vertices, Torus(width, height), face_id_pairs, vor.ridge_vertices, region_id_pairs)

def _modified_toroidal_voronoi_mesh(centres, width, height):
    """Returns a Mesh data structure on a torus constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions in the interval [-width/2,width/2]*[-height/2,height/2]
        width: a float giving periodicity in the x-direction
        height: a float giving periodicity in the y-direction
        Returns:
        A Mesh data structure.
    """
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    vor = voronoi(centres_3x3)


    N_cell = len(vor.points)/9
    region_id_pairs = (vor.ridge_points//N_cell-4).astype(int)  # fundamental region is 0 
    face_id_pairs = (vor.ridge_points % N_cell).astype(int)  # idx mapped to fundamental region
    vertex_id_pairs = (vor.ridge_vertices).astype(int)

    return _modified_build_mesh(vor.vertices, Torus(width, height), face_id_pairs, vertex_id_pairs, region_id_pairs)

def cylindrical_voronoi_mesh(centres, width, height):
    """Returns a Mesh data structure on a cylinder constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions in the interval [-width/2,width/2]*[-height/2,height/2]
        width: a float giving periodicity in the x-direction
        height: a float giving height in the y-direction (for constructing boundary)
        Returns:
        A Mesh data structure.
    """
    translated = np.vstack([centres+[dx, 0.0] for dx in [-width, 0, width]])
    reflected = [1.0, -1.0]*translated

    all_centres = np.vstack([translated, [0.0, -height]+reflected, [0.0, height]+reflected])
    vor = voronoi(all_centres)

    N = len(centres)
    region_id_pairs = ((vor.ridge_points // N) % 3) - 1
    face_id_pairs = vor.ridge_points
    mask = face_id_pairs < 3*N
    face_id_pairs[mask] %= N
    # round to multiple of 3*N
    face_id_pairs[~mask] //= 3*N
    face_id_pairs[~mask] *= 3*N

    boundary_face_ids = [3*N, 6*N]

    mask = np.any(mask, 1)
    face_id_pairs = face_id_pairs[mask]
    vertex_id_pairs = vor.ridge_vertices[mask]
    region_id_pairs = region_id_pairs[mask]

    # This is a hack to get the correct 'next' edges along the boundaries of the cylinder.
    # The logic for computing 'next' matches edges ordered by ('face', 'vertex') to edges ordered by ('face', 'vertex2').
    # By labelling the vertices as below, this will work also where the boundaries wind around the cylinder...
    permutation = np.argsort(vor.vertices[:, 0] % width)
    vertices = vor.vertices[permutation]
    vertex_id_pairs = inverse(permutation)[vertex_id_pairs]

    return build_mesh(vertices, Cylinder(width), face_id_pairs, vertex_id_pairs, region_id_pairs, boundary_face_ids)

def _modified_cylindrical_voronoi_mesh(centres, width, height):
    """Returns a Mesh data structure on a cylinder constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions in the interval [-width/2,width/2]*[-height/2,height/2]
        width: a float giving periodicity in the x-direction
        height: a float giving height in the y-direction (for constructing boundary)
        Returns:
        A Mesh data structure.
    """
    translated = np.vstack([centres+[dx, 0.0] for dx in [-width, 0, width]])
    reflected = [1.0, -1.0]*translated

    all_centres = np.vstack([translated, [0.0, -height]+reflected, [0.0, height]+reflected])
    vor = voronoi(all_centres)

    N = len(centres)
    region_id_pairs = ((vor.ridge_points // N) % 3) - 1
    face_id_pairs = vor.ridge_points
    mask = face_id_pairs < 3*N
    face_id_pairs[mask] %= N
    # round to multiple of 3*N
    face_id_pairs[~mask] //= 3*N
    face_id_pairs[~mask] *= 3*N

    boundary_face_ids = [3*N, 6*N]

    mask = np.any(mask, 1)
    face_id_pairs = face_id_pairs[mask]
    vertex_id_pairs = vor.ridge_vertices[mask]
    region_id_pairs = region_id_pairs[mask]

    # This is a hack to get the correct 'next' edges along the boundaries of the cylinder.
    # The logic for computing 'next' matches edges ordered by ('face', 'vertex') to edges ordered by ('face', 'vertex2').
    # By labelling the vertices as below, this will work also where the boundaries wind around the cylinder...
    permutation = np.argsort(vor.vertices[:, 0] % width)
    vertices = vor.vertices[permutation]
    vertex_id_pairs = inverse(permutation)[vertex_id_pairs]

    return _modified_build_mesh(vertices, Cylinder(width), face_id_pairs, vertex_id_pairs, region_id_pairs, boundary_face_ids)


def planar_voronoi_mesh(centres, reflected_centres):
    """Returns a Mesh data structure on a plane constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions
        reflected centres: an (N,2) float array of x,y positions reflected through the boundary
        Returns:
        A Mesh data structure.
    """
    vor = voronoi(np.vstack([centres, reflected_centres]))
    N_cell = len(centres)

    face_id_pairs = vor.ridge_points
    mask = np.any(face_id_pairs < N_cell, 1)
    face_id_pairs = np.clip(face_id_pairs, 0, N_cell)[mask]
    vertex_id_pairs = vor.ridge_vertices[mask]

    boundary_face_ids = [N_cell]

    return build_mesh(vor.vertices, Plane(), face_id_pairs, vertex_id_pairs, None, boundary_face_ids)


def random_centres(N_cell_across, N_cell_up, rand):
    N_cell, width, height = N_cell_across * N_cell_up, N_cell_across, N_cell_up
    a = rand.rand(N_cell, 2)-np.array([0.5, 0.5])  # uniform [-0.5,0.5]*[-0.5,0.5]
    b = (a*np.sqrt(N_cell/25)).astype(int)  # location on a coarse grid
    centres = a[np.lexsort((a[:, 0], b[:, 1], b[:, 0]))]  # sort by grid ref to improve locality
    centres = centres*np.array([width, height])
    return centres, width, height


def hexagonal_centres(N_cell_across, N_cell_up, noise, rand):
    assert(N_cell_up % 2 == 0)  # expect even number of rows
    dx, dy = 1.0/N_cell_across, 1.0/(N_cell_up/2)
    x = np.arange(-0.5+dx/4, 0.5, dx)
    y = np.arange(-0.5+dy/4, 0.5, dy)
    centres = np.zeros((N_cell_across, N_cell_up//2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    x += dx/2
    y += dy/2
    centres[:, :, 1, 0] += x[:, np.newaxis]
    centres[:, :, 1, 1] += y[np.newaxis, :]

    ratio = np.sqrt(2/np.sqrt(3))
    width = N_cell_across*ratio
    height = N_cell_up/ratio

    centres = centres.reshape(-1, 2)*np.array([width, height])
    centres += rand.rand(N_cell_up*N_cell_across, 2)*noise
    return centres, width, height

def modified_hexagonal_centres(N_cell_across, N_cell_up, noise, rand):
    assert(N_cell_up % 2 == 0)  # expect even number of rows
    dx, dy = 1.0/N_cell_across, 1.0/(N_cell_up/2)
    print(dx,dy)
    x = np.arange(-0.5+dx/4, 0.5, dx)
    y = np.arange(-0.5+dy/4, 0.5, dy)
    print(x,y)
    centres = np.zeros((N_cell_across, N_cell_up//2, 2, 2))
    print(x[:,np.newaxis])
    print(centres)
    centres[:, :, 0, 0] += x[:, np.newaxis]
    print(centres)
    print(y[np.newaxis,:])
    centres[:, :, 0, 1] += y[np.newaxis, :]
    print(centres)
    x += dx/2
    y += dy/2
    centres[:, :, 1, 0] += x[:, np.newaxis]
    print(centres)
    centres[:, :, 1, 1] += y[np.newaxis, :]
    print(centres)
    ratio = np.sqrt(2/np.sqrt(3))
    print(ratio)
    width = N_cell_across*ratio
    print(width)
    height = N_cell_up/ratio
    print(height)
    centres = centres.reshape(-1, 2)*np.array([width, height])
    print(centres)
    centres += rand.rand(N_cell_up*N_cell_across, 2)*noise
    print(centres)
    return centres, width, height

def toroidal_random_mesh(N_cell_across, N_cell_up, rand):
    return toroidal_voronoi_mesh(*random_centres(N_cell_across, N_cell_up, rand))

def _modified_toroidal_random_mesh(N_cell_across, N_cell_up, rand):
    return _modified_toroidal_voronoi_mesh(*random_centres(N_cell_across, N_cell_up, rand))
    # return toroidal_voronoi_mesh(*random_centres(N_cell_across, N_cell_up, rand))

def toroidal_hex_mesh(N_cell_across, N_cell_up, noise=None, rand=None):
    return toroidal_voronoi_mesh(*hexagonal_centres(N_cell_across, N_cell_up, noise, rand))

def _modified_toroidal_hex_mesh(N_cell_across, N_cell_up, noise=None, rand=None):
    return _modified_toroidal_voronoi_mesh(*hexagonal_centres(N_cell_across, N_cell_up, noise, rand))
    # return toroidal_voronoi_mesh(*hexagonal_centres(N_cell_across, N_cell_up, noise, rand))


def cylindrical_random_mesh(N_cell_across, N_cell_up, rand):
    return cylindrical_voronoi_mesh(*random_centres(N_cell_across, N_cell_up, rand))


def cylindrical_hex_mesh(N_cell_across, N_cell_up, noise=None, rand=None):
    # hexagonal centres doesn't work if you don't specify a noise value and rand.
    return cylindrical_voronoi_mesh(*hexagonal_centres(N_cell_across, N_cell_up, noise, rand))


def circular_random_mesh(N_cell, rand):
    R = np.sqrt(N_cell/np.pi)
    rand = np.random.RandomState(123456)
    r = R*np.sqrt(rand.rand(N_cell))
    theta = 2*np.pi*rand.rand(N_cell)
    centres = np.array((r*np.cos(theta), r*np.sin(theta))).T
    reflected_centres = (R*R/r/r)[:, None]*centres

    return planar_voronoi_mesh(centres, reflected_centres)

