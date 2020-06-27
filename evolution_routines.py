def evolve(self,new_cells,v,prod_rate,dt):
    """
    Performs one step of the FE method.  Uses np.linalg.solve which is slow.
    Args:
        new_cells is the new cells object after movement
        v is the diffusion coefficient
        prod_rate is the morphogen production rate.
        dt is the time step
    
    """
    m = len(self.concentration)
    A = np.zeros((m,m))
    bv = np.zeros(m) #bv stands for b vector
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_verts = new_cells.mesh.vertices.T
    new_cents = centroids2(new_cells)
    f = self.cells.properties['source']*prod_rate #source
    count=0

    for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
        #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
        new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
        prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
        node_id_tri = np.array([self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ],dtype=int)
        #print "node_id_tri", node_id_tri
        reduced_f = [0,0,f[f_by_e[e]]]
        old_alpha = self.concentration[node_id_tri]
        new_M = M(new_nodes)
        old_M=M(prev_nodes)
        d = np.abs(np.linalg.det(new_M))
        d_old = np.abs(np.linalg.det(old_M))           
        nabla_Phi = nabPhi(new_M)
        for i in range(3):
            bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
            for j in range(3):
                A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)

    A[np.where(A < 1.0e-8)] = 0
    if scipy.sparse.issparse(A):
        print("is sparse")
        sys.exit()
        self.concentration = scipy.sparse.linalg.spsolve(A,bv)
    else:
        print("is not sparse")
        sys.exit()
        self.concentration = np.linalg.solve(A,bv)

    self.cells = new_cells
    self.centroids = new_cents
    #return A,bv
    
def evolve2(self,v,prod_rate,dt):
    """
    Performs one step of the FE method. Computes the new cells object itself.
    Uses np.linalg.solve
    Args:
        new_cells is the new cells object after movement
        v is the diffusion coefficient
        prod_rate is the morphogen production rate.
        dt is the time step
    
    """
    m = len(self.concentration)
    A = np.zeros((m,m))
    bv = np.zeros(m) #bv stands for b vector
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_cells = cells_evolve(self.cells,dt)[0]
    new_verts = new_cells.mesh.vertices.T
    new_cents = centroids2(new_cells)
    f = self.cells.properties['source']*prod_rate #source
    count=0
    for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
        new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
        prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
        node_id_tri = np.array([self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ],dtype=int)
        reduced_f = [0,0,f[f_by_e[e]]]
        old_alpha = self.concentration[node_id_tri]
        new_M = M(new_nodes)
        old_M=M(prev_nodes)
        d = np.abs(np.linalg.det(new_M))
        d_old = np.abs(np.linalg.det(old_M))           
        nabla_Phi = nabPhi(new_M)
        for i in range(3):
            bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
            for j in range(3):
                A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)

    A = coo_matrix(A)
    self.concentration = scipy.sparse.linalg.spsolve(A,bv)
    self.cells = new_cells
    self.centroids = new_cents

    
def mat_and_vect(self,v,prod_rate,dt):
    """
    Returns the FE matrix and vector for the purposes of checking.
    
    """
    m = len(self.concentration)
    A = np.zeros((m,m))
    bv = np.zeros(m) #bv stands for b vector
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_cells = cells_evolve(self.cells,dt)[0]
    new_verts = new_cells.mesh.vertices.T
    new_cents = centroids2(new_cells)
    f = self.cells.properties['source']*prod_rate #source
    count=0
    for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
        #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
        #t1=time.time()
        new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
        prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
        node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
        #print "node_id_tri", node_id_tri
        reduced_f = [0,0,f[f_by_e[e]]]
        old_alpha = self.concentration[np.array(node_id_tri)]
        new_M = M(new_nodes)
        old_M=M(prev_nodes)
        d = np.abs(np.linalg.det(new_M))
        d_old = np.abs(np.linalg.det(old_M))           
        nabla_Phi = nabPhi(new_M)
        for i in range(3):
            bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
            for j in range(3):
                A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
        return A, bv
    
    
def evolve_original(self,v,prod_rate,dt):
    """
    Performs one step of the FE method. Computes the new cells object itself.
    Uses np.linalg.solve
    Args:
        new_cells is the new cells object after movement
        v is the diffusion coefficient
        prod_rate is the morphogen production rate.
        dt is the time step
    
    """
    m = len(self.concentration)
    A = np.zeros((m,m))
    bv = np.zeros(m) #bv stands for b vector
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_cells = cells_evolve(self.cells,dt)[0]
    new_verts = new_cells.mesh.vertices.T
    new_cents = centroids2(new_cells)
    f = self.cells.properties['source']*prod_rate #source
    count=0
    for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
        #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
        #t1=time.time()
        new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
        prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
        node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
        #print "node_id_tri", node_id_tri
        reduced_f = [0,0,f[f_by_e[e]]]
        old_alpha = self.concentration[np.array(node_id_tri)]
        new_M = M(new_nodes)
        old_M=M(prev_nodes)
        d = np.abs(np.linalg.det(new_M))
        d_old = np.abs(np.linalg.det(old_M))           
        nabla_Phi = nabPhi(new_M)
        for i in range(3):
            bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
            for j in range(3):
                A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
        #t2=time.time()
        #print t2 - t1
        #count+=1  MATRIX A IS SYMMETRIC.
        #print count
        #if count ==300:
            #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #print old_alpha
    #for i in range(m):
     #   for j in range(m):
       #     print A[i][j]-A[j][i]
    #print np.shape(A), np.shape(bv)
    #print "shape",np.shape(A)
    #print A[:1]
    #print A[:2], "A2"
    #print A
    #print scipy.sparse.issparse(A), " that A is sparse"
    self.concentration = np.linalg.solve(A,bv) #could change to scipy.linalg.solve(A,bv,assume_a='sym')
    self.cells = new_cells
    self.centroids = new_cents

def evolve_v2(self,v,prod_rate,dt):
    """
    DOESN'T WORK.
    
    Attempt at using a shared array and parallel processing
    to update the matrix.  
    Performs one step of the FE method. Computes the new cells object itself.
    Args:
        new_cells is the new cells object after movement
        v is the diffusion coefficient
        prod_rate is the morphogen production rate.
        dt is the time step
    """
    con=self.concentration
    m = len(con)
    A = np.zeros((m,m)) #set up as Array from multiprocessing
    bv = np.zeros(m) #bv stands for b vector, set up as Array from multiprocessing
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_cells = cells_evolve(self.cells,dt)[0]
    new_verts = new_cells.mesh.vertices.T
    new_cents = centroids2(new_cells) #replace with the cython version.
    f = self.cells.properties['source']*prod_rate #source
    ftn = self.faces_to_nodes
    etn = self.edges_to_nodes
    bv = Array(ctypes.c_double, m)        
    A = Array(ctypes.c_double, m*m)
    processes=[]
    for e in self.cells.mesh.edges.ids: 
        p = Process(target=updater, args=(A,bv,m,e, new_verts, new_cents, old_verts, old_cents, nxt, f_by_e, etn, ftn, con, f, v, dt))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    self.concentration = scipy.sparse.linalg.spsolve(A,bv)#could change to scipy.linalg.solve(A,bv,assume_a='sym')
    self.cells = new_cells
    self.centroids = new_cents

def evolve_cy(self,v,prod_rate,dt):
    """
    Performs one step of the FE method. Computes the new cells object itself.
    Uses np.linalg.solve
    Args:
        new_cells is the new cells object after movement
        v is the diffusion coefficient
        prod_rate is the morphogen production rate.
        dt is the time step
    
    """
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    new_cells = cells_evolve(self.cells,dt)[0]
    new_verts = new_cells.mesh.vertices.T
    new_cents = cen2(new_cells)#centroids2(new_cells)
    f = self.cells.properties['source']*prod_rate #source
    n_edge = self.cells.mesh.edges.ids[-1]+1
    #ev2(np.ndarray old_verts, np.ndarray new_verts, np.ndarray old_cents,  np.ndarray new_cents, np.ndarray old_con, np.ndarray nx,np.ndarray f_by_e, np.ndarray  e_t_n, np.ndarray f_t_n, np.ndarray f , int n_edge , double v, double dt ):
    self.concentration = ev5(old_verts.astype(np.float64), new_verts.astype(np.float64), old_cents.astype(np.float64),new_cents.astype(np.float64), self.concentration.astype(np.float64), nxt.astype(np.intc) ,f_by_e.astype(np.intc), self.edges_to_nodes.astype(np.intc), self.faces_to_nodes.astype(np.intc), f.astype(np.float64) , np.intc(n_edge) , np.float64(v), np.float64(dt) )
    self.cells = new_cells
    self.centroids = new_cents

def transitions_faster(self,ready=None):
    if ready is None:
        ready = ready_to_divide(self.cells)
    c_by_e = self.concentration[self.edges_to_nodes]
    c_by_c = self.concentration[self.faces_to_nodes]
    self.cells = T1(self.cells) #perform T1 transitions - "neighbour exchange"
    self.cells,c_by_e = rem_collapsed(self.cells,c_by_e) #T2 transitions-"leaving the tissue"
    self.cells,c_by_e, c_by_c = divide(self.cells,c_by_e,c_by_c,ready)
    self.centroids = cen2(self.cells)
    eTn = self.cells.mesh.edges.ids//3
    n = max(eTn)
    cTn=np.cumsum(~self.cells.empty())+n
    con_part=c_by_e[::3]
    cent_part = c_by_c[~self.cells.empty()]
    self.concentration = np.hstack([con_part,cent_part])
    self.edges_to_nodes = self.cells.mesh.edges.ids//3
    self.faces_to_nodes = cTn



        
def evolve_modified(fe_vtx,v,prod_rate,dt):
        """
        Performs one step of the FE method. Computes the new cells object itself.
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        
        """
        m = len(fe_vtx.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=fe_vtx.cells.mesh.edges.next
        f_by_e = fe_vtx.cells.mesh.face_id_by_edge
        old_verts = fe_vtx.cells.mesh.vertices.T
        old_cents = fe_vtx.centroids
        new_cells = cells_evolve(fe_vtx.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = fe_vtx.cells.properties['source']*prod_rate #source
        count=0
        for e in fe_vtx.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [fe_vtx.edges_to_nodes[e],fe_vtx.edges_to_nodes[nxt[e]] , fe_vtx.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = fe_vtx.concentration[np.array(node_id_tri,dtype=int)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #count+=1  MATRIX A IS SYMMETRIC.
            #print count
            #if count ==300:
                #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                #print old_alpha
        #for i in range(m):
         #   for j in range(m):
           #     print A[i][j]-A[j][i]
        #print np.shape(A), np.shape(bv)
        #print "shape",np.shape(A)
        #print A[:1]
        #print A[:2], "A2"
        fe_vtx.concentration = np.linalg.solve(A,bv)#could change to scipy.linalg.solve(A,bv,assume_a='sym')
        fe_vtx.cells = new_cells
        fe_vtx.centroids = new_cents
        return fe_vtx

def edge_to_updates(nxt,concentration ,new_verts,new_cents,old_verts,old_cents,edges_to_nodes, faces_to_nodes ,f, f_by_e,v,dt, e):
    new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
    prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
    node_id_tri = [edges_to_nodes[e],edges_to_nodes[nxt[e]] , faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
    reduced_f = [0,0,f[f_by_e[e]]]
    old_alpha = concentration[np.array(node_id_tri)]
    new_M = M(new_nodes)
    old_M=M(prev_nodes)
    d = np.abs(np.linalg.det(new_M))
    d_old = np.abs(np.linalg.det(old_M))           
    nabla_Phi = nabPhi(new_M)
    b_updates=[]
    A_updates=[]
    for i in range(3):
        b_updates.append([node_id_tri[i],b(i,d,d_old,reduced_f,old_alpha,dt)])
        for j in range(3):
            A_updates.append([node_id_tri[i] , node_id_tri[j],I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)])
    return b_updates, A_updates


def updater(A,bv,dim, e, new_verts, new_cents, old_verts, old_cents, nxt, f_by_e, etn, ftn, con, f, v, dt):
    new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
    prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
    node_id_tri = [etn[e],etn[nxt[e]] , ftn[f_by_e[e]] ]
    #print "node_id_tri", node_id_tri
    reduced_f = [0,0,f[f_by_e[e]]]
    old_con = con[np.array(node_id_tri)]
    new_M = M(new_nodes)
    old_M = M(prev_nodes)
    d = np.abs(np.linalg.det(new_M))
    d_old = np.abs(np.linalg.det(old_M))           
    nabla_Phi = nabPhi(new_M)
    update_matrix(A,dim,node_id_tri,d,nabla_Phi,v,new_nodes,prev_nodes)
    update_vect(bv,dim,node_id_tri,d,d_old, reduced_f, old_con, dt)
    

def update_matrix(A,dim ,node_id_tri,d,nabla_Phi,v,new_nodes,prev_nodes):
    with A.get_lock(): # synchronize access
        a = np.frombuffer(A.get_obj())
        aa = a.reshape(dim,dim)
        for i in range(3):
            for j in range(3):
                aa[node_id_tri[i]][node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                
def update_vect(bv,dim,node_id_tri,d,old_d, red_f, old_con, dt):
    with bv.get_lock():
        c = np.frombuffer(bv.get_obj())
        cc=c.reshape(dim)
        for i in range(3):
            c[node_id_tri[i]] += b(i,d,old_d,red_f,old_con,dt)