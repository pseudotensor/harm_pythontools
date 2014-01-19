import matplotlib
matplotlib.use('WxAgg')
matplotlib.interactive(True)
from numpy import mgrid, empty, sin, cos, pi
from tvtk.api import tvtk
from mayavi import mlab

def create_structured_grid(s=None,sname=None,v=None,vname=None,maxr=100):
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

def visualize_data(doreload=1):
    grid3d("gdump.bin",use2d=1)
    rfd("fieldline9000.bin")
    sg = create_structured_grid(s=lrho,sname="density",v=None,vname=None)
    # Now visualize the data.
    d = mlab.pipeline.add_dataset(sg)
    gx = mlab.pipeline.grid_plane(d)
    # gy = mlab.pipeline.grid_plane(d)
    # gy.grid_plane.axis = 'y'
    gz = mlab.pipeline.grid_plane(d)
    gz.grid_plane.axis = 'z'
    iso = mlab.pipeline.iso_surface(d)
    #vol = mlab.pipeline.volume(d)
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    mlab.view(focalpoint=[0,0,0],distance=20)
    #mlab.show()
