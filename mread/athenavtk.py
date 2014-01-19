import numpy as np
import matplotlib
import vtk
from vtk.util import numpy_support as VN

def rdath3d(fname):
    global n1, n2, n3, t, ti, tj, tk, x1, x2, x3, rho, v1, v2, v3, pg, B1c, B2c, B3c
    if fname.endswith(".tab"):
        rdtab(fname)
    elif fname.endswith(".vtk"):
        rdvtk(fname)
    else:
        print( "rdath3d: Unknown file type: %s" % fname )

def rdvtk(fname):
    global n1, n2, n3, t, ti, tj, tk, x1, x2, x3, rho, v1, v2, v3, pg, B1c, B2c, B3c
    global reader, data, dim
    filename = fname
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    vecdim = list(dim)
    vecdim = [i-1 for i in dim]
    scalardim = vecdim[:] #make a copy
    vecdim.append(3)
    v = VN.vtk_to_numpy(data.GetCellData().GetArray('velocity'))
    Bc = VN.vtk_to_numpy(data.GetCellData().GetArray('cell_centered_B'))
    rho = VN.vtk_to_numpy(data.GetCellData().GetArray('density'))
    pg = VN.vtk_to_numpy(data.GetCellData().GetArray('pressure'))
    v1,v2,v3 = v.reshape(vecdim,order="F").transpose(3,0,1,2)
    B1c,B2c,B3c = Bc.reshape(vecdim,order="F").transpose(3,0,1,2)
    rho = rho.reshape(scalardim,order="F")
    pg = pg.reshape(scalardim,order="F")
    #xyz = VN.vtk_to_numpy(data.GetPoints().GetData())
    # x = zeros(data.GetNumberOfPoints())
    # y = zeros(data.GetNumberOfPoints())
    # z = zeros(data.GetNumberOfPoints())    
    # for i in xrange(data.GetNumberOfPoints()):
    #     x[i],y[i],z[i] = data.GetPoint(i)
    # x = x.reshape(dim,order="F")
    # y = y.reshape(dim,order="F")
    # z = z.reshape(dim,order="F")
    # Assume regular grid and get: (i) x1,x2,x3 and (ii) ti,tj,tk
    ncorn = data.GetNumberOfPoints()
    xstart,ystart,zstart = data.GetPoint(0)
    xend,yend,zend = data.GetPoint(ncorn-1)
    dx,dy,dz = data.GetSpacing()
    x1d = np.linspace(xstart+0.5*dx,xend-0.5*dx,scalardim[0])
    y1d = np.linspace(ystart+0.5*dy,yend-0.5*dy,scalardim[1])
    z1d = np.linspace(zstart+0.5*dz,zend-0.5*dz,scalardim[2])
    x1 = np.zeros(scalardim)+x1d[:,None,None]
    x2 = np.zeros(scalardim)+y1d[None,:,None]
    x3 = np.zeros(scalardim)+z1d[None,None,:]
    ti1d = np.arange(scalardim[0])
    tj1d = np.arange(scalardim[1])
    tk1d = np.arange(scalardim[2])
    ti = np.zeros(scalardim)+ti1d[:,None,None]
    tj = np.zeros(scalardim)+tj1d[None,:,None]
    tk = np.zeros(scalardim)+tk1d[None,None,:]
    
def rdtab(fname):
    global n1, n2, n3, t, ti, tj, tk, x1, x2, x3, rho, v1, v2, v3, pg, B1c, B2c, B3c
    fin = open( fname , "rb" )
    header1 = fin.readline().split()
    header2 = fin.readline().split()
    header3 = fin.readline().split()
    header4 = fin.readline().split()
    header5 = fin.readline().split()
    header6 = fin.readline().split()
    header7 = fin.readline().split()
    n1 = np.float64(header1[3].split("'")[0])
    n2 = np.float64(header3[3].split("'")[0])
    n3 = np.float64(header5[3].split("'")[0])
    t = np.float64(header7[6].split(",")[0])
    fin.close()
    res = np.loadtxt(fname,dtype=np.float64,skiprows=0,unpack=1).reshape((-1,n1,n2,n3),order="F")
    ti, tj, tk, x1, x2, x3, rho, v1, v2, v3, pg, B1c, B2c, B3c = res
