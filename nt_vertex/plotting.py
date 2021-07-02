
# coding: utf-8

# In[2]:

import itertools
import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D #Added by me
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.colors as mcol
from matplotlib.animation import FuncAnimation
from matplotlib import use
use('Agg')

from .permutations import cycles

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
colors_array = np.array([mcol.to_rgb(color[gene]) for gene in ['Pax', 'Olig', 'Nkx', 'Irx']])

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

def draw_cells(cells, xlim=[0,1], ylim=[0,1], ax=None, colored=True):
    """
    Modified version of draw from plotting.
    
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
    facecolors = cells.properties.get('color', None)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # if final_width is None:
    #     final_width= cells.mesh.geometry.width
    # if final_height is None:
    #     if hasattr(cells.mesh.geometry,'height'):
    #         final_height = cells.mesh.geometry.height
    #     else:
    #         final_height =max(np.abs( cells.mesh.vertices[1]))

        
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
    source = cells.properties['source']
    
    # set alpha channel as soft-max
    # alpha channel for each gene (N_cells, genes)
    alpha = np.exp(poni_state/.3)
    alpha /= np.sum(alpha, axis=1)[:,None]

    cells.properties['color'] = np.matmul(alpha, colors_array)

    # set particular color for differentiating / source cells
    cells.properties['color'][cells.properties['leaving']==1] = np.array([0,0,0])
    cells.properties['color'][cells.properties['source']==1] = np.array([1,1,1])


def drawShh(coord_tri, concs_tri, xlim=[0,1], ylim=[0,1], zlim=[0,1], ax=None, heatmap=True, log=True):

    z_low, z_high = zlim
    if not ax:
        fig = plt.figure()
        if heatmap:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

    ax.cla() # clear the axis
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if heatmap:
        if np.allclose(concs_tri.ravel(), 0):
            return
        tri = Triangulation(coord_tri[:,:,0].ravel(), coord_tri[:,:,1].ravel(), triangles=np.arange(len(coord_tri)*3).reshape(-1,3))
        if log:
            z = np.log10(concs_tri.ravel() + 1e-9)
            levels=np.linspace(-8,np.log10(z_high), 20)
        else:
            z = concs_tri.ravel()
            levels=np.linspace(0,z_high, 20)

        im = ax.tricontourf(tri, z,
                                levels=levels,
                                cmap=plt.get_cmap('Greens'),
                                extend='max'
                            )
    else:
        raise NotImplementedError("surface plot not implemented")
        # l=[]
        # r=[]
        # for i in range(len(nodes)):
        #     l.append(nodes[i][0])
        #     r.append(nodes[i][1])
        # ax.set_zlim([z_low -0.1 , z_high])
        # Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()



def combined_video(NT_list, filename=None,
            xlim=None, ylim=None, zlim=None, heatmap=True, log=True,
            duration=60.,
            ffmpeg=False):

    use('Agg')

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
    coord_tri_list = [nt.FE_vtx.cells.mesh.recentre().triangles   for nt in NT_list]


    # calculate bounds for plotting
    x_min, y_min, z_min = 3*[np.inf]
    x_max, y_max, z_max = 3*[np.NINF]
    for coord_tri, conc_tri in zip(coord_tri_list, concs_tri_list):
        x_min = min(coord_tri[:,:,0].ravel().min(), x_min)
        y_min = min(coord_tri[:,:,1].ravel().min(), y_min)
        z_min = min(conc_tri.ravel().min(), z_min)

        x_max = max(coord_tri[:,:,0].ravel().max(), x_max)
        y_max = max(coord_tri[:,:,1].ravel().max(), y_max)
        z_max = max(conc_tri.ravel().max(), z_max)

    if xlim is not None:
        x_min, x_max = xlim
    if ylim is not None:
        y_min, y_max = ylim
    if zlim is not None:
        z_min, z_max = zlim
        
    ax_lims = {'xlim': [x_min, x_max], 'ylim': [y_min, y_max]}

    fig, ax = plt.subplots(nrows=2)
    for a in ax:
        for side in ['top', 'bottom', 'left', 'right']:
            a.spines[side].set_visible(False)
    
    def plot_frame(k):
        plt.cla()
        # top panel
        # 1., 0. to be replaced in general by z_high, z_low
        drawShh(coord_tri_list[k], concs_tri_list[k],
            **ax_lims,
            zlim=[z_min, z_max], ax=ax[0], heatmap=heatmap, log=log)
        if heatmap:
            draw_cells(cells_list[k], **ax_lims, ax=ax[0], colored=False)

        # bottom panel
        set_colour_poni_state(cells_list[k],poni_list[k])
        draw_cells(cells_list[k], **ax_lims, ax=ax[1])

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
        dt = duration*1000./len(cells_list)
        ani = FuncAnimation(fig, plot_frame,
                            interval=dt,
                            frames=frames,
                            blit=False)
        ani.save(f'{filename}.mp4')