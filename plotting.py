
# coding: utf-8

# In[2]:

import itertools
import numpy as np
from matplotlib.collections import PolyCollection
#import matplotlib#added by G for cluster
#matplotlib.use('Agg')#added  by G for cluster
import matplotlib.pyplot as plt
import os
from permutations import cycles
from mpl_toolkits.mplot3d import Axes3D #Added by me
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.colors as mcol
from matplotlib.animation import FuncAnimation
from matplotlib import use
use('Agg')

color = {
    "Shh":"#000000",    # black
    "Ptch": "#8D8D8D",  # grey
    "GliFL": "#8900FF", # purple
    "GliA": "#00AD2F",  # green
    "GliR": "#D02B09",  # red
    "Pax": "#0040FF",   # blue
    "Olig": "#FF00EB",  # magenta
    "Nkx": "#02DDD3",   # cyan
    "Irx": "#DDB002"    # gold
}

def _draw_edges(mesh, ax):
    w = mesh.vertices - mesh.vertices.take(mesh.edges.rotate, 1)  # winding
    to_draw = mesh.edges.ids[(mesh.edges.ids < mesh.edges.reverse) | (np.abs(w[0])+np.abs(w[1]) > 0.1)]
    start, end = mesh.vertices.take(to_draw, 1), mesh.vertices.take(mesh.edges.next[to_draw], 1)

    n = np.empty(len(start[0]))
    n.fill(np.nan)
    x = np.dstack([start[0], end[0], n]).ravel()
    y = np.dstack([start[1], end[1], n]).ravel()

    ax.plot(x, y, 'k-', linewidth=0.5)

def _draw_edges_non(mesh, ax):
    w = mesh.vertices - mesh.vertices.take(mesh.edges.rotate, 1)  # winding
    to_draw = mesh.edges.ids[(mesh.edges.ids < mesh.edges.reverse) | (np.abs(w[0])+np.abs(w[1]) > 0.1)]
    start, end = mesh.vertices.take(to_draw, 1), mesh.vertices.take(mesh.edges.next[to_draw], 1)

    n = np.empty(len(start[0]))
    n.fill(np.nan)
    x = np.dstack([start[0], end[0], n]).ravel()
    y = np.dstack([start[1], end[1], n]).ravel()

def _draw_midpoints(cells, ax):
    s = cells.vertices/(np.maximum(np.bincount(cells.edges.cell), 1)[cells.edges. cell])
    sx, sy = np.bincount(cells.edges.cell, weights=s[0]), np.bincount(cells.edges. cell, weights=s[1])
    mx, my = sx[cells.edges.cell], sy[cells.edges.cell]

    ax.scatter(mx, my, s=3.0, c='b', edgecolors='none', marker='o')


_PALETTES = {  # .<=3      4       5       6       7       8     >=9    edges
    'Default': '#000000 #33cc33 #ffff19 #ccccb2 #005cb8 #cc2900 #4a0093 #b0fc3e',
    'CB':      '#edf8fb #edf8fb #bfd3e6 #9ebcda #8c96c6 #8856a7 #810f7c #000000',
}
_PALETTES = {name: np.array([clr.split()[0]]*4+clr.split()[1:])
             for name, clr in _PALETTES.items()}


def _draw_faces(mesh, ax, facecolors, edgecolor='k',alpha=1.):
    order, labels = cycles(mesh.edges.next)
    counts = np.bincount(labels)

    vs = mesh.vertices.T.take(order, 0)

    faces, face_ids = [], []
    cell_ids = mesh.face_id_by_edge[order]
    boundary = mesh.boundary_faces if mesh.has_boundary() else []

    for (i, c) in zip(counts, np.cumsum(counts)):
        cell_id = cell_ids[c-i]
        if cell_id in boundary:
            continue
        faces.append(vs[c-i:c])
        face_ids.append(cell_ids[c-i])

    coll = PolyCollection(faces, facecolors=facecolors[face_ids], edgecolors=edgecolor, linewidths=0.5, alpha=alpha)
    ax.add_collection(coll)

def _draw_faces_no_edge(mesh, ax, facecolors):
    order, labels = cycles(mesh.edges.next)
    counts = np.bincount(labels)

    vs = mesh.vertices.T.take(order, 0)

    faces, face_ids = [], []
    cell_ids = mesh.face_id_by_edge[order]
    boundary = mesh.boundary_faces if mesh.has_boundary() else []

    for (i, c) in zip(counts, np.cumsum(counts)):
        cell_id = cell_ids[c-i]
        if cell_id in boundary:
            continue
        faces.append(vs[c-i:c])
        face_ids.append(cell_ids[c-i])

    coll = PolyCollection(faces, facecolors=facecolors[face_ids])
    ax.add_collection(coll)

def _draw_geometry(geometry, ax=None):
    # Torus
    if hasattr(geometry, 'width') and hasattr(geometry, 'height'):
        w, h = geometry.width, geometry.height
        # ax.add_patch(plt.Rectangle((-0.5*w, -0.5*h), w, h, fill=False, linewidth=2.0))

def draw_cells(cells,final_width=None,final_height=None, ax=None, colored=True):
    """
    Modified version of draw from plotting.
    
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
    facecolors = cells.properties.get('color', None)
    
    if final_width is None:
        final_width= cells.mesh.geometry.width
    if final_height is None:
        if hasattr(cells.mesh.geometry,'height'):
            final_height = cells.mesh.geometry.height
        else:
            final_height =max(np.abs( cells.mesh.vertices[1]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-0.55*final_width,0.55*final_width])
    ax.set_ylim([-0.55*final_height,0.55*final_height])
        
    mesh = cells.mesh.recentre()

    if facecolors is None or not colored:
        _draw_edges(mesh, ax)
    else:
        _draw_faces(mesh, ax, facecolors)
    # _draw_midpoints(cells,ax)
    # _draw_geometry(mesh.geometry, ax)

    #size = size or 2.0*np.max(mesh.vertices[0])

    plt.draw()


def set_colour_poni_state(cells,poni_state):
    n_face = cells.mesh.n_face
    source =cells.properties['source']
    cells.properties['color']=np.ones((n_face, 3)) #to store RGB number for each face
    for k in range(n_face):
        m = np.argmax(poni_state[k])
        if source[k]==1:
            cells.properties['color'][k] = np.array([1,1,1]) #source
        elif m==0:
            cells.properties['color'][k] = mcol.to_rgb(color['Pax'])# np.array([0,0,1]) #Blue, pax high
        elif m==1:
            cells.properties['color'][k] = mcol.to_rgb(color['Olig'])# np.array([1,0,0]) #Red, Olig2 high
        elif m==2:
            cells.properties['color'][k] = mcol.to_rgb(color['Nkx'])# np.array([0,0,1]) #Green, NKx22 high
        elif m==3:
            cells.properties['color'][k] = mcol.to_rgb(color['Irx'])# np.array([0,1,1]) # ?, Irx high 

        if cells.properties['leaving'][k] == 1:
            cells.properties['color'][k] = np.array([0,0,0]) # differentating cells


def drawShh(nodes, alpha, z_high, z_low, ax=None, final_width=None, final_height=None, size=None, heatmap=True):
    l=[]
    r=[]
    # if not size:
    #     d_size=10.0
    # else:
    #     d_size = size
    for i in range(len(nodes)):
        l.append(nodes[i][0])
        r.append(nodes[i][1])
    if not ax:
        fig = plt.figure()
        if heatmap:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

    ax.cla() # clear the axis
    ax.set_xlim([-0.55*final_width,0.55*final_width])
    ax.set_ylim([-0.55*final_height,0.55*final_height])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlim([-0.55*d_size,0.55*d_size])
    # ax.set_ylim([-0.55*d_size,0.55*d_size])
    if heatmap:
        tri = Triangulation(nodes[:,0],nodes[:,1])
        # mask out elongated triangles at the borders
        mask = TriAnalyzer(tri).get_flat_tri_mask(.05)
        tri.set_mask(mask)
        ax.tricontourf(tri, alpha, levels=np.linspace(0.,z_high, 20), cmap=plt.get_cmap('Greens'))
        # refiner = UniformTriRefiner(tri)
        # tri_refi, z_test_refi = refiner.refine_field(alpha, subdiv=3)
        # ax.tricontourf(tri_refi, z_test_refi, levels=np.linspace(0.,z_high, 20))
    else:
        ax.set_zlim([z_low -0.1 , z_high])
        Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()  

def morphogen_video(cells_list, nodes_array, concs_list, outputdir, name_file, zmin=None, zmax=None, heatmap=True):

    v_max = np.max((np.max(nodes_array[0]) , np.max(nodes_array[-1])))
    size = 2*v_max
    if zmax is None:
        dummy_max=[]
        for i in range(len(concs_list)):
            dummy_max.append(np.max(concs_list[i]))
        z_high = max(dummy_max)
    else:
        z_high = zmax
    if zmin is None:
        dummy_min=[]
        for i in range(len(concs_list)):
            dummy_min.append(np.min(concs_list[i]))
        z_low = min(dummy_min)
    else:
        z_low = zmin

    fig = plt.figure()
    ax = fig.add_subplot(nrows=2)
    # fig.set_size_inches(6,6)
    i=0
    frames=[]
    final_width = cells_list[-1].mesh.geometry.width
    if hasattr(cells_list[-1].mesh.geometry,'height'):
        final_height = cells_list[-1].mesh.geometry.height
    else:
        final_height =max(np.abs(cells_list[-1].mesh.vertices[1]))
    for k in range(len(cells_list)):
        drawShh(nodes_array[i], concs_list[i], z_high, z_low, ax, size, heatmap=heatmap)
        if heatmap:
            _draw_edges(cells_list[k].mesh, ax)
        i=i+1
        frame=outputdir+"/image%03i.png" % i
        fig.savefig(frame,dpi=500,bbox_inches="tight")
        frames.append(frame)  
    os.system("cd ")
    os.system("ffmpeg -framerate 5/1 -i "+outputdir+"/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer  

    # for frame in frames: os.remove(frame)  

def cells_state_video(cells_list, poni_list, outputdir, name_file):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig.set_size_inches(6,6)
    i=0
    frames=[]
    final_width = cells_list[-1].mesh.geometry.width
    if hasattr(cells_list[-1].mesh.geometry,'height'):
        final_height = cells_list[-1].mesh.geometry.height
    else:
        final_height =max(np.abs(cells_list[-1].mesh.vertices[1]))
    for k in range(len(cells_list)):
        set_colour_poni_state(cells_list[k],poni_list[k])
        draw_cells(cells_list[k],final_width,final_height, ax)
        i=i+1
        frame=outputdir+"/image%03i.png" % i
        fig.savefig(frame,dpi=500,bbox_inches="tight")
        frames.append(frame)

    os.system("cd ")
    os.system("ffmpeg -framerate 5/1 -i "+outputdir+"/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer

    # for frame in frames: os.remove(frame)  


def combined_video(NT_list, filename=None,
            zmin=None, zmax=None, heatmap=True, ffmpeg=False):
    #def combined_video(cells_list, nodes_list, concs_list, poni_list, outputdir, name_file,
    #         zmin=None, zmax=None, heatmap=True, ffmpeg=False):

    if filename is None:
        raise ValueError("Provide name for output video file")

    nodes_list = [
                np.vstack([
                    nt.FE_vtx.cells.mesh.vertices.T[::3],
                    nt.FE_vtx.centroids[~nt.FE_vtx.cells.empty()]
                ]) for nt in NT_list]
    concs_list = [nt.FE_vtx.concentration   for nt in NT_list]
    poni_list = [nt.GRN.state[:,-4:]   for nt in NT_list]
    cells_list = [nt.FE_vtx.cells   for nt in NT_list]
    # verts_list = [nt.FE_vtx.cells.mesh.vertices.T[::3] for nt in NT_list]
    concs_tri_list = [nt.FE_vtx.concentration_triangles   for nt in NT_list]

    # setup

    v_max = np.max((np.max(nodes_list[0]) , np.max(nodes_list[-1])))
    size = 2*v_max
    if zmax is None:
        dummy_max=[]
        for i in range(len(concs_list)):
            dummy_max.append(np.max(concs_list[i]))
        z_high = max(dummy_max)
    else:
        z_high = zmax
    if zmin is None:
        dummy_min=[]
        for i in range(len(concs_list)):
            dummy_min.append(np.min(concs_list[i]))
        z_low = min(dummy_min)
    else:
        z_low = zmin

    fig, ax = plt.subplots(nrows=2)
    for a in ax:
        for side in ['top', 'bottom', 'left', 'right']:
            a.spines[side].set_visible(False)

    final_width = cells_list[-1].mesh.geometry.width
    if hasattr(cells_list[-1].mesh.geometry,'height'):
        final_height = cells_list[-1].mesh.geometry.height
    else:
        final_height =max(np.abs(cells_list[-1].mesh.vertices[1]))
    
    def plot_frame(k):
        plt.cla()
        # top panel
        # 1., 0. to be replaced in general by z_high, z_low
        drawShh(nodes_list[k], concs_list[k], 1., 0., ax[0],
            final_width=final_width,final_height=final_height, heatmap=heatmap)
        if heatmap:
            draw_cells(cells_list[k], final_width, final_height, ax[0], colored=False)

        # bottom panel
        set_colour_poni_state(cells_list[k],poni_list[k])
        draw_cells(cells_list[k], final_width, final_height, ax[1])

    if ffmpeg:
        i=0
        frames=[]
        for k in range(len(cells_list)):
            plot_frame(k)
            plt.show()
            exit()
            i=i+1
            frame=outputdir+"/image%03i.png" % i
            fig.savefig(frame,dpi=100,bbox_inches="tight")
            frames.append(frame)  
        os.system("cd ")
        os.system("ffmpeg -framerate 30 -i "+outputdir+"/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+filename+".mp4") #for Mac computer  

        # for frame in frames: os.remove(frame)  

    else:
        frames=range(len(cells_list))
        dt = 60000./len(cells_list)
        ani = FuncAnimation(fig, plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
        ani.save(f'{filename}.mp4')