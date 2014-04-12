import matplotlib
from mayavi.scripts import mayavi2
matplotlib.use('WxAgg')
matplotlib.interactive(True)
from numpy import mgrid, empty, zeros, sin, cos, pi
from tvtk.api import tvtk
from mayavi import mlab
from scipy import ndimage

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
    i3d = (i2d[np.int32(R3d/dR2d),np.int32((z3d+zmax)/dz2d)])
    j3d = (j2d[np.int32(R3d/dR2d),np.int32((z3d+zmax)/dz2d)])
    #pdb.set_trace()
    dphi = dxdxp[3,3][0,0,0]*_dx3
    k3d = (np.arctan2(y3d, x3d)/dphi-0.5)
    k3d[k3d<0] = k3d[k3d<0] + nz
    return i3d, j3d, k3d, x3d, y3d, z3d   

def trilin(a,i,j,k,order=3):
    #a is the array to be interpolated
    #i,j,k are indices, which are 3d arrays; can be non-integers (floats)
    #returns interpolated values of a[i,j,k]
    nbnd = 3
    a_with_bnd_cells = np.concatenate((a[...,-nbnd:],a,a[...,:nbnd]),axis=-1)
    return ndimage.map_coordinates(a_with_bnd_cells,np.array([i,j,k+nbnd]),order=order)
    #####################
    # x = i-np.floor(i)
    # i0 = np.int32(np.floor(i))
    # j0 = np.int32(np.floor(j))
    # k0 = np.int32(np.floor(k))
    # c00 = a[i0][j0][k0]*(1-x) + a[i0+1][j0][k0]*x
    # c10 = a[i0][j0+1][k0]*(1-x) + a[i0+1][j0+1][k0]*x
    # #
    # x = None; del x
    # i0 = None; del i0
    # j0 = None; del j0
    # k0 = None; del k0
    # gc.collect()
    # #
    # y = j-np.floor(j)
    # c0 = c00*(1-y)+c10*y
    # c1 = c01*(1-y)+c11*y
    # #
    # c00 = None; del c00
    # c01 = None; del c01
    # c10 = None; del c10
    # c11 = None; del c11
    # y = None; del y
    # gc.collect()
    # #
    # z = k-np.floor(k)
    # c = c0*(1-z)+c1*z
    # #
    # z = None; del z
    # c0 = None; del c0
    # c1 = None; del c1
    # gc.collect()
    # #
    # return( c )
    #################
    # x = i-np.floor(i)
    # y = j-np.floor(j)
    # z = k-np.floor(k)
    # return( +a[np.int32(np.floor(i))][np.int32(np.floor(j))][np.int32(np.floor(k))]*(1-x)*(1-y)*(1-z)
    #         +a[np.int32(np.floor(i))][np.int32(np.floor(j))][np.int32(np.ceil(k))]*(1-x)*(1-y)*z
    #         +a[np.int32(np.floor(i))][np.int32(np.ceil(j))][np.int32(np.floor(k))]*(1-x)*y*(1-z)
    #         +a[np.int32(np.floor(i))][np.int32(np.ceil(j))][np.int32(np.ceil(k))]*(1-x)*y*z
    #         +a[np.int32(np.ceil(i))][np.int32(np.floor(j))][np.int32(np.floor(k))]*x*(1-y)*(1-z)
    #         +a[np.int32(np.ceil(i))][np.int32(np.floor(j))][np.int32(np.ceil(k))]*x*(1-y)*z
    #         +a[np.int32(np.ceil(i))][np.int32(np.ceil(j))][np.int32(np.floor(k))]*x*y*(1-z)
    #         +a[np.int32(np.ceil(i))][np.int32(np.ceil(j))][np.int32(np.ceil(k))]*x*y*z
    #         )

#@mayavi2.standalone    
def visualize_data(doreload=1,no=5468,xmax=200,ymax=200,zmax=1000,ncellx=200,ncelly=200,ncellz=1000,dosavefig=0):
    if doreload:
        grid3d("gdump.bin",use2d=1)
        #rfd("fieldline9000.bin")
        rfd("fieldline%04d.bin"%no)
        cvel()
    scene = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(210*2, 297*2))
    #pdb.set_trace()
    #choose 1.8 > sqrt(3):
    # grid the data.
    # var = lrho[...,0][r[...,0]<1.8*rmax]
    # vi = griddata((x, z), var, (xi, zi), method="linear")
    # pdb.set_trace()
    i3d_disk,j3d_disk,k3d_disk,xi_disk,yi_disk,zi_disk =\
        interp3d(xmax=500,ymax=500,zmax=100,ncellx=500,ncelly=500,ncellz=100)
    i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
        interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    #lrhoxpos = lrho*(r*sin(h)*cos(ph)>0)
    lrhoi_disk = np.float32(trilin(lrho,i3d_disk,j3d_disk,k3d_disk))
    lrhoi_jet = np.float32(trilin(lrho,i3d_jet,j3d_jet,k3d_jet))
    mlab_lrho_disk = mlab.pipeline.scalar_field(xi_disk,yi_disk,zi_disk,lrhoi_disk)
    mlab_lrho_jet = mlab.pipeline.scalar_field(xi_jet,yi_jet,zi_jet,lrhoi_jet)
    # bsqorhoi_jet = np.float32((bsq/rho)[i3d_jet,j3d_jet,k3d_jet])
    # mlab_bsqorho = mlab.pipeline.scalar_field(xi_jet,yi_jet,zi_jet,bsqorhoi_jet)
    #
    # Magnetic field
    #
    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #
    BR=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
    Bx=BR*cos(ph)-Bpnorm*sin(ph)
    By=BR*sin(ph)+Bpnorm*cos(ph)
    Bz=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
    Bxi = np.float32(Bx[np.int32(i3d_jet),np.int32(j3d_jet),np.int32(k3d_jet)])
    Byi = np.float32(By[np.int32(i3d_jet),np.int32(j3d_jet),np.int32(k3d_jet)])
    Bzi = np.float32(Bz[np.int32(i3d_jet),np.int32(j3d_jet),np.int32(k3d_jet)])
    # pdb.set_trace()
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
        vol_disk = mlab.pipeline.volume(mlab_lrho_disk) #,vmin=-6,vmax=1)
        vol_disk.volume_mapper.blend_mode = 'maximum_intensity'
        # Change the otf (opacity transfer function) of disk and jet:
        from tvtk.util.ctf import PiecewiseFunction
        otf_disk = PiecewiseFunction()
        if 0:
            otf_disk.add_point(-6, 0)
            otf_disk.add_point(-1.7, 0)
            otf_disk.add_point(-0.25, 0.8)
            otf_disk.add_point(1, 0.8)
        elif 1:
            #brighter disk
            otf_disk.add_point(-6., 0)
            otf_disk.add_point(-1.733, 0)
            otf_disk.add_point(-1.133, 0.51)
            otf_disk.add_point(-0.533, 0.796)
            otf_disk.add_point(1., 1.)
        vol_disk._otf = otf_disk
        vol_disk._volume_property.set_scalar_opacity(otf_disk)
        vol_jet = mlab.pipeline.volume(mlab_lrho_jet) #,vmin=-6,vmax=1)
        vol_jet.volume_mapper.blend_mode = 'minimum_intensity'
        otf_jet = PiecewiseFunction()
        if 0:
            otf_jet.add_point(-6., 0.429)
            otf_jet.add_point(-4.547, 0.429)
            otf_jet.add_point(-2.92, 0)
            otf_jet.add_point(1, 0.)
        elif 1:
            #less diffuse jet, so it does not touch box boundaries
            otf_jet.add_point(-6.0, 0.408)
            otf_jet.add_point(-4.8, 0.408)
            otf_jet.add_point(-3.8, 0.)
            otf_jet.add_point(1., 0.)
        vol_jet._otf = otf_jet
        vol_jet._volume_property.set_scalar_opacity(otf_jet)
    if 1:
        streamlines = []
        xpos = [1, -.77, 3.2, -1.5]
        ypos = [0,  0.25, 0,  -1]
        zpos = [0,  0, 0,  0]
        intdir = ['backward', 'backward', 'forward', 'forward']
        for sn in xrange(4):
            streamline = mlab.flow(xi_jet, yi_jet, zi_jet, Bxi, Byi, Bzi, seed_scale=0.01,
                             seed_resolution=5,
                             integration_direction=intdir[sn],
                             seedtype='point')
            streamlines.append(streamline)
            streamline.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
            streamline.streamline_type = 'tube'
            streamline.tube_filter.radius = 2.022536581333679
            if 0:
                streamline.seed.widget.phi_resolution = 3
                streamline.seed.widget.theta_resolution = 3
                streamline.seed.widget.radius = 1.0
            elif 0: 
                #more tightly wound field lines
                streamline.seed.widget.phi_resolution = 10
                streamline.seed.widget.theta_resolution = 5
                #make them more round
            streamline.tube_filter.number_of_sides = 8
            streamline.stream_tracer.progress = 1.0
            streamline.stream_tracer.maximum_number_of_steps = 10000L
            #streamline.stream_tracer.start_position =  np.array([ 0.,  0.,  0.])
            streamline.stream_tracer.maximum_propagation = 20000.0
            streamline.seed.widget.position = np.array([ xpos[sn],  ypos[sn],  zpos[sn]])
            streamline.seed.widget.enabled = False
            streamline.update_streamlines = 1
        #pdb.set_trace()
    if 0:
        myr = 20
        myi = iofr(myr)
        #mesh
        s = wraparound((np.abs(B[1])*dxdxp[1,1]))[myi,:,:]
        x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
        y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
        z = wraparound(r*cos(h))[myi,:,:]
        mlab.mesh(x, y, z, scalars=s, colormap='jet')
    if 1:
        #show the black hole
        rbh = rhor
        thbh = np.linspace(0,np.pi,128,endpoint=1)[:,None]
        phbh = np.linspace(0,2*np.pi,128,endpoint=1)[None,:]
        thbh = thbh + 0*phbh + 0*thbh
        phbh = phbh + 0*phbh + 0*thbh
        xbh = rbh*sin(thbh)*cos(phbh)
        ybh = rbh*sin(thbh)*sin(phbh)
        zbh = rbh*cos(thbh)
        mlab.mesh(xbh, ybh, zbh, scalars=1+0*thbh, colormap='gist_yarg',vmin=0, vmax = 1)
    #move camera:
    scene.scene.camera.position = [1037.2177748124982, 1411.458249001643, -510.33158924621932]
    scene.scene.camera.focal_point = [9.7224118574481291e-12, -1.2963215809930837e-10, 2.5926431619861676e-11]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.33552748243496205, -0.092397178145941311, -0.93748817059284728]
    scene.scene.camera.clipping_range = [4.235867945966219, 4235.8679459662189]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()    
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    #mlab.view(focalpoint=[0,0,0],distance=500)
    #mlab.show()
    if dosavefig:
        mlab.savefig("disk_jet_with_field_lines.png", figure=scene, magnification=6.0)
