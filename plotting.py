
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
#from matplotlib import animation


def _draw_edges(mesh, ax):
    w = mesh.vertices - mesh.vertices.take(mesh.edges.rotate, 1)  # winding
    to_draw = mesh.edges.ids[(mesh.edges.ids < mesh.edges.reverse) | (np.abs(w[0])+np.abs(w[1]) > 0.1)]
    start, end = mesh.vertices.take(to_draw, 1), mesh.vertices.take(mesh.edges.next[to_draw], 1)

    n = np.empty(len(start[0]))
    n.fill(np.nan)
    x = np.dstack([start[0], end[0], n]).ravel()
    y = np.dstack([start[1], end[1], n]).ravel()

    ax.plot(x, y, 'k-', linewidth=1.0)

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


def _draw_faces(mesh, ax, facecolors, edgecolor='k'):
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

    coll = PolyCollection(faces, facecolors=facecolors[face_ids], edgecolors=edgecolor,
                          linewidths=2.0)
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


def draw(cells, ax=None, size=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()  
    ax.cla()
    facecolors = cells.properties.get('color', None)

    mesh = cells.mesh.recentre()

    if facecolors is None:
        _draw_edges(mesh, ax)
    else:
        _draw_faces(mesh, ax, facecolors)
# _draw_midpoints(cells,ax)
    _draw_geometry(mesh.geometry, ax)

    ax.set_xticks([])
    ax.set_yticks([])

    size = size or 2.0*np.max(mesh.vertices[0])
    lim = [-0.55*size, 0.55*size]
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    plt.draw()
    
def draw_tor_graeme(cells, ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()  
    ax.cla()
    facecolors = cells.properties.get('color', None)

    mesh = cells.mesh.recentre()

    if facecolors is None:
        _draw_edges(mesh, ax)
    else:
        _draw_faces(mesh, ax, facecolors)
# _draw_midpoints(cells,ax)
    _draw_geometry(mesh.geometry, ax)

    ax.set_xticks([])
    ax.set_yticks([])

    #size = size or 2.0*np.max(mesh.vertices[0])
    x_lims = [-0.55*cells.mesh.geometry.width, 0.55*cells.mesh.geometry.width]
    y_lims = [-0.55*cells.mesh.geometry.height, 0.55*cells.mesh.geometry.height]
    
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    plt.draw()
    
def draw_Graeme(cells,name,ax=None, size=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()  
    ax.cla()
    facecolors = cells.properties.get('color', None)

    mesh = cells.mesh.recentre()

    if facecolors is None:
        _draw_edges(mesh, ax)
    else:
        _draw_faces(mesh, ax, facecolors)
# _draw_midpoints(cells,ax)
    _draw_geometry(mesh.geometry, ax)

    ax.set_xticks([])
    ax.set_yticks([])

    size = size or 2.0*np.max(mesh.vertices[0])
    lim = [-0.55*size, 0.55*size]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.savefig(name)
    

def drawShh(nodes,alpha, zl, ax=None, size=None):
    l=[]
    r=[]
    if not size:
        d_size=10.0
    else:
        d_size = size
    for i in range(len(nodes)):
        l.append(nodes[i][0])
        r.append(nodes[i][1])
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_xlim([-0.55*d_size,0.55*d_size])
    ax.set_ylim([-0.55*d_size,0.55*d_size])
    ax.set_zlim([-0.1 , zl])
    Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()
    
def drawShh2(nodes,alpha,z_high, z_low, ax=None, size=None):
    l=[]
    r=[]
    if not size:
        d_size=10.0
    else:
        d_size = size
    for i in range(len(nodes)):
        l.append(nodes[i][0])
        r.append(nodes[i][1])
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_xlim([-0.55*d_size,0.55*d_size])
    ax.set_ylim([-0.55*d_size,0.55*d_size])
    ax.set_zlim([z_low -0.1 , z_high])
    Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()  
    
def drawShh3(nodes,alpha,z_high, z_low, ax=None, size=None):
    l=[]
    r=[]
    if not size:
        d_size=10.0
    else:
        d_size = size
    for i in range(len(nodes)):
        l.append(nodes[i][0])
        r.append(nodes[i][1])
    l = np.array(l)
    r=np.array(r)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_xlim([-0.55*d_size,0.55*d_size])
    ax.set_ylim([-0.55*d_size,0.55*d_size])
    ax.set_zlim([z_low -0.1 , z_high])
    Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()

def drawShh4(nodes,alpha,z_low,z_high,final_length,height, ax=None):
    l=[]
    r=[]
    for i in range(len(nodes)):
        l.append(nodes[i][0])
        r.append(nodes[i][1])
    l = np.array(l)
    r=np.array(r)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_xlim([-0.1,final_length+0.1])
    ax.set_ylim([-0.1,height+0.1])
    ax.set_zlim([z_low -0.1 , z_high])
    Axes3D.plot_trisurf(ax,l,r,alpha)
    plt.draw()

    
"""
l=[]
r=[]
for i in range(len(nodes)):
    l.append(nodes2[i][0])
    r.append(nodes2[i][1])
fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
Axes3D.plot_trisurf(ax2,l,r,dummy2[1])
plt.show()


"""    

def animate(cells_array, facecolours='Default'):
    v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    size = 2.0*v_max
    #to don't draw 
    fig = plt.figure()#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
    ax = fig.gca()#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
    fig.set_size_inches(6, 6)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
    for cells in cells_array:#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
        draw(cells, ax, size)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
  #hasta aqui to don't draw        

def draw1(cells, ax=None, size=None):
    if not ax:
        fig = plt.figure()
        ax = fig.gca()  
    ax.cla()
    facecolors = cells.properties.get('color', None)

    mesh = cells.mesh.recentre()

    if facecolors is None:
        _draw_edges_non(mesh, ax)
    else:
        _draw_faces_no_edge(mesh, ax, facecolors)
    _draw_geometry(mesh.geometry, ax)

    ax.set_xticks([])
    ax.set_yticks([])

    size = size or 2.0*np.max(mesh.vertices[0])
    lim = [-0.55*size, 0.55*size]
    ax.set_xlim(lim)
    ax.set_ylim(lim)

def draw_cells(cells,final_width=None,final_height=None, ax=None):
    """
    Modified version of draw from plotting.
    
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()  
    ax.cla()
    facecolors = cells.properties.get('color', None)
    
    if final_width is None:
        final_width= cells.mesh.geometry.width
    if final_height is None:
        if hasattr(cells.mesh.geometry,'height'):
            final_height = cells.mesh.geometry.height
        else:
            final_height =max(np.abs( cells.mesh.vertices[1]))
        
    mesh = cells.mesh.recentre()

    if facecolors is None:
        _draw_edges(mesh, ax)
    else:
        _draw_faces(mesh, ax, facecolors)
# _draw_midpoints(cells,ax)
    _draw_geometry(mesh.geometry, ax)

    ax.set_xticks([])
    ax.set_yticks([])

    #size = size or 2.0*np.max(mesh.vertices[0])

    ax.set_xlim([-0.55*final_width,0.55*final_width])
    ax.set_ylim([-0.55*final_height,0.55*final_height])
    plt.draw()

def cell_growth_video_mpg2(cells_array, name_file):    
    #v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    widths=[]
    heights=[]
    for i in range(len(cells_array)):
        widths.append(cells_array[i].mesh.geometry.width)
        if hasattr(cells_array[i].mesh.geometry,'height'):
            heights.append(cells_array[i].mesh.geometry.height)
    max_width = max(widths)
    if len(heights)>0:
        max_height = max(heights)
    else:
        max_height =max( max(cells_array[0].mesh.vertices[1]), max(cells_array[-1].mesh.vertices[1]) ) 
    #z_low = min(dummy_min)
    #size = 10.0
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.add_subplot(111); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for i in range(len(cells_array)):
        #drawShh(nodes_array[i],alpha_array[i],27.0, ax , size)
        draw_cells(cells_array[i],max_width,max_height, ax)
        #drawShh4(nodes_array[i],alpha_array[i], z_low,z_high,final_length, height,ax) #drawShh4(nodes,alpha,z_low,z_high,final_length,width, ax=None):
        i=i+1
        frame="images/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    #os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    #os.system("cd /usr/local/bin")
    os.system("cd ")
    #os.system("cd /opt")
    os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+"surf.mp4") #for Mac computer
    print(os.system("pwd"))
    #os.system("cd ")
    #os.system("cd Desktop/vertex_model/images")
    os.system("cd images")
    for frame in frames: os.remove(frame)  
        
def animate_video_mpg(cells_array,name_file,facecolours='Default'):    
    v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    size = 2.0*v_max
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.gca(); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for cells in cells_array:
        draw(cells,ax,size)
        i=i+1
        frame="images/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    # os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p test.mp4") #for Mac computer
    for frame in frames: os.remove(frame)



    
def _modified_animate_video_mpg(cells_array,name_file,facecolours='Default'):    
    v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    size = 2.0*v_max
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.gca(); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for cells in cells_array:
        draw(cells,ax,size)
        i=i+1
        frame="images/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    #os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    #os.system("cd /usr/local/bin")
    os.system("cd ")
    os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer
    os.system("cd images")
    for frame in frames: os.remove(frame) 



def _modified_animate_video_mpg2(cells_array,name_file,facecolours='Default'):    
    v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    size = 2.0*v_max
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.gca(); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for cells in cells_array:
        draw(cells,ax,size)
        i=i+1
        frame="images/image%03d.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    #os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    os.system("cd /usr/local/bin")
    os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer
    os.system("cd images")
    for frame in frames: os.remove(frame) 
    
    
def animate_surf_video_mpg(nodes_array,alpha_array, outputdir, name_file):    
    #v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    v_max = np.max((np.max(nodes_array[0]) , np.max(nodes_array[-1])))
    size = 2*v_max
    dummy_max=[]
    dummy_min=[]
    for i in range(len(alpha_array)):
        dummy_max.append(np.max(alpha_array[i]))
        dummy_min.append(np.min(alpha_array[i]))
    z_high = max(dummy_max)
    z_low = min(dummy_min)
    #size = 10.0
    # outputdir="images"
    # if not os.path.exists(outputdir): # if the folder doesn't exist create it
    #     os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.add_subplot(111, projection='3d'); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for i in range(len(nodes_array)):
        #drawShh(nodes_array[i],alpha_array[i],27.0, ax , size)
        drawShh2(nodes_array[i],alpha_array[i],z_high, z_low,ax,size)
        i=i+1
        frame=outputdir+"/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    os.system("ffmpeg -framerate 5/1 -i "+outputdir+"/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+"surf.mp4") #for Mac computer
    # print(os.system("pwd"))
    #os.system("cd ")
    #os.system("cd Desktop/vertex_model/images")
    #os.system("cd images")
    for frame in frames: os.remove(frame)  
    
def animate_surf_video_mpg2(nodes_array,alpha_array, final_length, height, name_file):    
    #v_max = np.max((np.max(cells_array[0].mesh.vertices), np.max(cells_array[-1].mesh.vertices)))
    dummy_max=[]
    dummy_min=[]
    for i in range(len(alpha_array)):
        dummy_max.append(np.max(alpha_array[i]))
        dummy_min.append(np.min(alpha_array[i]))
    z_high = max(dummy_max)
    z_low = min(dummy_min)
    #size = 10.0
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.add_subplot(111, projection='3d'); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for i in range(len(nodes_array)):
        #drawShh(nodes_array[i],alpha_array[i],27.0, ax , size)
        drawShh4(nodes_array[i],alpha_array[i], z_low,z_high,final_length, height,ax) #drawShh4(nodes,alpha,z_low,z_high,final_length,width, ax=None):
        i=i+1
        frame="images/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    #os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    #os.system("cd /usr/local/bin")
    os.system("cd ")
    #os.system("cd /opt")
    os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+"surf.mp4") #for Mac computer
    print(os.system("pwd"))
    #os.system("cd ")
    #os.system("cd Desktop/vertex_model/images")
    #os.system("cd images")
    for frame in frames: os.remove(frame)      

def animate_video_mpg_zoom(cells_array,name_file,facecolours='Default'):    
    v_max = np.max((np.max(cells_array[0].mesh.vertices[1]), np.max(cells_array[-1].mesh.vertices[1])))
    size = 2.0*v_max
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.gca(); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    for cells in cells_array:
        draw(cells,ax,size)
        i=i+1
        frame="images/zoom_image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
    os.system("mencoder 'mf://images/zoom_image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + name_file)  
    # os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p test.mp4") #for Mac computer
    for frame in frames: os.remove(frame)

def set_colour_poni_state(cells,poni_state):
    n_face = cells.mesh.n_face
    source =cells.properties['source']
    cells.properties['color']=np.ones((n_face, 3)) #to store RGB number for each face
    for k in range(n_face):
        m = np.argmax(poni_state[k])
        if source[k]==1:
            cells.properties['color'][k] = np.array([1,1,1]) #source
        elif m==0:
            cells.properties['color'][k] = np.array([0,0,1]) #Blue, pax high
        elif m==1:
            cells.properties['color'][k] = np.array([1,0,0]) #Red, Olig2 high
        elif m==2:
            cells.properties['color'][k] = np.array([0,0,1]) #Green, NKx22 high
        elif m==3:
            cells.properties['color'][k] = np.array([0,1,1]) # ?, Irx high 

def cells_state_video(cells_history, poni_state_history, outputdir, name_file):
    #time = 0
    #history=[tissue.cells]
    # outputdir="images"
    # if not os.path.exists(outputdir): # if the folder doesn't exist create it
    #     os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.add_subplot(111); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    final_width = cells_history[-1].mesh.geometry.width
    if hasattr(cells_history[-1].mesh.geometry,'height'):
        final_height = cells_history[-1].mesh.geometry.height
    else:
        final_height =max(np.abs(cells_history[-1].mesh.vertices[1]))
    for k in range(len(cells_history)):
        #tissue = normalised_simulate_tissue_step(surface_function,tissue,bind_rate,time,dt,expansion)
        set_colour_poni_state(cells_history[k],poni_state_history[k])
        draw_cells(cells_history[k],final_width,final_height, ax) #draw_cells(cells,final_width=None,final_height=None, ax=None)
        #drawShh4(nodes_array[i],alpha_array[i], z_low,z_high,final_length, height,ax) #drawShh4(nodes,alpha,z_low,z_high,final_length,width, ax=None):
        i=i+1
        frame=outputdir+"/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
        #print tissue.cells.properties['color']
        #history.append(copy.deepcopy(tissue))
        #print tissue.cells.mesh.geometry.width
    os.system("cd ")
    #os.system("cd /opt")
    os.system("ffmpeg -framerate 5/1 -i "+outputdir+"/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer
    print((os.system("pwd")))
    #os.system("cd ")
    #os.system("cd Desktop/vertex_model/images")
    #os.system("cd images")
    #for frame in frames: os.remove(frame)  