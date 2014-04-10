import matplotlib
from mayavi.scripts import mayavi2
matplotlib.use('WxAgg')
matplotlib.interactive(True)
from numpy import mgrid, empty, zeros, sin, cos, pi
from tvtk.api import tvtk
from mayavi import mlab

def create_structured_grid(s=None,sname=None,v=None,vname=None,maxr=500):
    maxi = iofr(maxr)
    # Compute Cartesian coordinates of the grid
    x = (r*sin(h)*cos(ph))[:maxi]
    y = (r*sin(h)*sin(ph))[:maxi]
    z = (r*cos(h))[:maxi]

    # The actual points.
    pts = empty((3,) + z.shape, dtype=float)
    pts[0,...] = x
    pts[1,...] = y
    pts[2,...] = z

    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.T.reshape(pts.size/3,3)

    sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)

    if s is not None:
        sg.point_data.scalars = s[:maxi].T.ravel()
        sg.point_data.scalars.name = sname
    if v is not None:
        vec = v[:,:maxi]
        sg.point_data.vectors = vec.T.reshape(vec.size/3,3)
        sg.point_data.vectors.name = vname

    return( sg )
    
def create_unstructured_grid(s=None,sname=None,v=None,vname=None,minr=1,maxr=500,npts=100,dj=1,dk=1):
    if minr < rhor: minr = rhor
    mini = iofr(minr)
    maxi = iofr(maxr)
    rlist = np.linspace(minr,maxr,npts)
    ilist = iofr(rlist)
    slc = lambda f: f[...,ilist,::dj,::dk]
    # Compute Cartesian coordinates of the grid
    x = slc(r*sin(h)*cos(ph))
    y = slc(r*sin(h)*sin(ph))
    z = slc(r*cos(h))
    nx, ny, nz = z.shape

    #ti, tj, tk = mgrid[0:nx,0:ny,0:nz]
    ti = np.arange(nx)[:,None,None]
    tj = np.arange(ny)[None,:,None]
    tk = np.arange(nz)[None,None,:]

    ind = (ti+nx*(tj+tk*ny))

    # The actual points.
    pts = empty((3,) + z.shape, dtype=float)
    pts[0,...] = x
    pts[1,...] = y
    pts[2,...] = z

    num_points = pts.size/3

    #ind = np.arange(num_points).reshape(nx,ny,nz)

    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.T.reshape(num_points,3)

    tets = np.array(
        [0, 1, nx+1, nx,   #bottom of cube
         0, 1, nx+1, nx,   #will be top of cube (after addtion of nx*ny)
         ], 'd')
    
    tets[4:8] += nx*ny
    #pdb.set_trace()
    #peel off a layer of one cell thick
    ind1 = ind[:nx-1,:ny-1,:].T.ravel()

    #define the array of cube's vertices of shape ((nx-1)*(ny-1)*nz,8)
    tets_array = (ind1[:,None] + tets[None,:]) % num_points
        
    tet_type = tvtk.Hexahedron().cell_type
    ug = tvtk.UnstructuredGrid(points=pts)
    ug.set_cells(tet_type, tets_array)

    if s is not None:
        ug.point_data.scalars = slc(s).T.ravel()
        ug.point_data.scalars.name = sname
    if v is not None:
        vec = slc(v)
        ug.point_data.vectors = vec.T.reshape(vec.size/3,3)
        ug.point_data.vectors.name = vname

    return( ug )

def wraparound(v):
    """ wraparound the phi-direction """
    return( np.concatenate((v,v[...,0:1]),axis=-1) )

def interp3d(xmax=100,ymax=100,zmax=100,ncellx=100,ncelly=100,ncellz=100):
    #first, construct 1d arrays
    x3d = np.linspace(-xmax, xmax, ncellx,endpoint=1)[:,None,None]
    y3d = np.linspace(-ymax, ymax, ncelly,endpoint=1)[None,:,None]
    z3d = np.linspace(-zmax, zmax, ncellz,endpoint=1)[None,None,:]
    rmax = (xmax**2+ymax**2+zmax**2)**0.5
    Rmax = (xmax**2+ymax**2)**0.5
    #make the arrays 3d
    x3d = x3d + 0*x3d+0*y3d+0*z3d
    y3d = y3d + 0*x3d+0*y3d+0*z3d
    z3d = z3d + 0*x3d+0*y3d+0*z3d
    #construct meridional 2d grid:
    R2d = np.linspace(0, 2*Rmax, ncellx*2); dR2d = R2d[1]-R2d[0];
    z2d = np.linspace(-zmax, zmax, ncellz,endpoint=1); dz2d = z2d[1]-z2d[0];
    #compute i,j-indices on meridional grid:
    ph = 0
    x = (r*sin(h))[...,0][r[...,0]<1.1*rmax]
    z = (r*cos(h))[...,0][r[...,0]<1.1*rmax]
    i = (ti)[...,0][r[...,0]<1.1*rmax]
    j = (tj)[...,0][r[...,0]<1.1*rmax]
    #get i,j-indices on the 2d meridional grid grid
    i2d = griddata((x, z), i, (R2d[:,None], z2d[None,:]), method="linear",fill_value=0)
    j2d = griddata((x, z), j, (R2d[:,None], z2d[None,:]), method="linear",fill_value=0)
    #
    R3d = (x3d**2+y3d**2)**0.5
    i3d = np.int32(i2d[np.int32(R3d/dR2d),np.int32((z3d+zmax)/dz2d)])
    j3d = np.int32(j2d[np.int32(R3d/dR2d),np.int32((z3d+zmax)/dz2d)])
    #pdb.set_trace()
    dphi = dxdxp[3,3][0,0,0]*_dx3
    k3d = np.int32(np.arctan2(x3d, y3d)/dphi-0.5)
    k3d[k3d<0] = k3d[k3d<0] + nz
    return i3d, j3d, k3d, x3d, y3d, z3d   
    
#@mayavi2.standalone    
def visualize_data(doreload=1,no=5468,xmax=100,ymax=100,zmax=1000,ncellx=100,ncelly=100,ncellz=1000):
    if doreload:
        grid3d("gdump.bin",use2d=1)
        #rfd("fieldline9000.bin")
        rfd("fieldline%04d.bin"%no)
        cvel()
    #pdb.set_trace()
    #choose 1.8 > sqrt(3):
    # grid the data.
    # var = lrho[...,0][r[...,0]<1.8*rmax]
    # vi = griddata((x, z), var, (xi, zi), method="linear")
    # pdb.set_trace()
    i3d,j3d,k3d,xi,yi,zi = interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    lrhoi = lrho[i3d,j3d,k3d]
    mlab_lrho = mlab.pipeline.scalar_field(xi,yi,zi,lrhoi)
    mlab_lrho1 = mlab.pipeline.scalar_field(xi,yi,zi,lrhoi)
    bsqorhoi = (bsq/rho)[i3d,j3d,k3d]
    mlab_bsqorho = mlab.pipeline.scalar_field(xi,yi,zi,bsqorhoi)
    # pdb.set_trace()
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    if 0:
        sg = create_structured_grid(s=lrho,sname="density",v=None,vname=None)
        # Now visualize the data.
        d = mlab.pipeline.add_dataset(sg)
        gx = mlab.pipeline.grid_plane(d)
        # gy = mlab.pipeline.grid_plane(d)
        # gy.grid_plane.axis = 'y'
        gz = mlab.pipeline.grid_plane(d)
        gz.grid_plane.axis = 'z'
        iso = mlab.pipeline.iso_surface(d)
    if 1:
        #sg = create_unstructured_grid(s=B[1]*dxdxp[1,1],sname="density",v=None,vname=None)
        #sg_bsqorho = create_unstructured_grid(s=bsq/rho,sname="density",v=None,vname=None)
        #sg_d = create_unstructured_grid(s=rho,sname="density",v=None,vname=None)
        # Now visualize the data.
        #pl_d = mlab.pipeline.add_dataset(sg_d)
        #pl_bsqorho = mlab.pipeline.add_dataset(sg_bsqorho)
        # gx = mlab.pipeline.grid_plane(d)
        # gy = mlab.pipeline.grid_plane(d)
        # gy.grid_plane.axis = 'y'
        # gz = mlab.pipeline.grid_plane(d)
        # gz.grid_plane.axis = 'z'
        #iso_bsqorho = mlab.pipeline.iso_surface(pl_bsqorho,contours=[10.])
        #iso_d = mlab.pipeline.iso_surface(pl_d,contours=[0.1,1,10.],color=(255./255., 255./255., 0./255.))
        #vol = mlab.pipeline.volume(d,vmin=-4,vmax=-2)
        vol = mlab.pipeline.volume(mlab_lrho,vmin=-6,vmax=1)
        vol = mlab.pipeline.volume(mlab_lrho1,vmin=-6,vmax=1)
    myr = 20
    myi = iofr(myr)
    if 0:
        #mesh
        s = wraparound((np.abs(B[1])*dxdxp[1,1]))[myi,:,:]
        x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
        y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
        z = wraparound(r*cos(h))[myi,:,:]
        mlab.mesh(x, y, z, scalars=s, colormap='jet')
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    mlab.view(focalpoint=[0,0,0],distance=50)
    #mlab.show()
