import numpy as np
def VisitScript():
    # You can run this by:
    #     Saving the script below to "script.py"
    #     Running "visit -cli -s script.py" 
    #OpenDatabase("/Users/atchekho/run/fixdt_x2_60/fieldline0073.vtk")
    OpenDatabase("/Users/atchekho/run/test3d_1cpu_16x16x8/fieldline0000.vtk")
    DefineScalarExpression("Rsq", "x*x+y*y")
    AddPlot("Contour","Rsq")
    p=ContourAttributes()
    #p.contourMethod="Value"
    #print p
    #p.contourValue=1.04
    p.singleColor=(255,0,0,128)  #red, 50% transparent
    p.SetColorType(0)
    p.SetMin(25)
    #p.SetMax(5.1)
    p.SetMinFlag(1)
    p.SetContourNLevels(1)
    #p.SetMaxFlag(1)
    SetPlotOptions(p)
    #ChangeActivePlotsVar("Rvar")
    #
    #
    DefineScalarExpression("rsphsq", "x*x+y*y+z*z")
    AddPlot("Contour","rsphsq")
    p=ContourAttributes()
    #p.contourMethod="Value"
    #print p
    #p.contourValue=1.04
    p.singleColor=(0,255,0,128)  #red, 50% transparent
    p.SetColorType(0)
    p.SetMin(1.1)
    #p.SetMax(5.1)
    p.SetMinFlag(1)
    #p.SetMaxFlag(1)
    p.SetContourNLevels(1)
    SetPlotOptions(p)
    #ChangeActivePlotsVar("Rvar")
    #
    #
    DrawPlots()
    #
    AddPlot("Streamline","B")
    #Or:
    #SetActivePlot(0)
    #p=GetPlotOptions()
    p=StreamlineAttributes()
    p.SetSourceType(1) #SpecifiedPointList
    p.SetShowSeeds(1)
    #concatenate points in a tuple (3 numbers per point)
    p.SetPointList((2,0,0,3,0,0,4,0,0))
    p.SetTermDistance(100)
    #0/1 - backward/forward
    #2   - both directions
    p.SetStreamlineDirection(2)
    p.SetShowSeeds(0)
    SetPlotOptions(p)
    # For moviemaking, you'll need to save off the image
    # SaveWindow()
    DrawPlots()
    #SetActivePlot(0)
    #see http://visitusers.org/index.php?title=Using_pick_to_create_curves
    #ZonePick((2,0,0),("TIME"))

def get_visit_time()
    SuppressQueryOutputOn() 
    Query("Time")
    t = GetQueryOutputValue()
    SuppressQueryOutputOff()
    return t

def rotate_around_y(r, th, ph, alpha=0):
    """Returns x, y, z of the rotated vector"""
    alpha = -alpha
    xp = r*(-np.cos(th)*np.sin(alpha)+np.sin(th)*np.cos(ph)*np.cos(alpha))
    yp = r*np.sin(th)*np.cos(ph)
    zp = r*(np.cos(th)*np.cos(alpha)+np.sin(th)*np.cos(ph)*np.sin(alpha))
    return xp, yp, zp

def rotate_around_z(x, y, z, alpha=0):
    x = xp * np.cos(alpha) - yp * np.sin(alpha)
    y = xp * np.sin(alpha) + yp * np.cos(alpha)
    z = zp
    return x, y, z

def rotate_around_y_z(r, th, ph, alpha_y=0, alpha_z=0):
    xp, yp, zp = rotate_around_y(r,th,ph,alpha=alpha_y)
    x, y, z = rotate_around_y(xp,yp,zp,alpha=alpha_z)
    return x, y, z 

def compute_footpoints(r0 = 1.5, Rlc=5,npts=10,whichpole="up", alpha_y=0, alpha_z=0):
    thp = (r0/Rlc)**0.5
    ph = 2*np.pi*np.arange(0,1,1./npts)
    r = r0 + np.zeros_like(ph)
    th = thp + np.zeros_like(ph)
    return rotate_around_y_z(r, th, ph,alpha_y=alpha_y, alpha_z=alpha_z)

def zip2visit(xyz):
    return np.ravel(zip(*xyz))

def get_visit_footpoints(r0 = 1.5, Rlc=5, npts=10, whichpole="up", alpha_y=0, alpha_z=0):
    zip2visit(compute_footpoints(r0=r0, Rlc=Rlc, npys=npts, whichpole=whichpole, alpha_y=alpha_y, alpha_z=alpha_z))


if __name__ == "__main__":
    testvariable = 3