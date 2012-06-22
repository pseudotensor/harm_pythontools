import numpy as np
import pdb as pdb

def mkvmov():
    for no in xrange(0,74):
        VisitScript(no=no,r0=1.05)

def VisitScript(no=0,r0=1.05,cdb=True):
    # You can run this by:
    #     Saving the script below to "script.py"
    #     Running "visit -cli -s script.py" 
    #OpenDatabase("/Users/atchekho/run2/fixdt_x2_60/fieldline0000.vtk")
    #dbname = "/home/atchekho/run2/fixdt_x2_60/fieldline%04d.vtk" % no
    dbname = "/home/atchekho/run2/fixdt_60/fieldline%04d.vtk" % no
    OpenDatabase(dbname)
    #OpenDatabase("/Users/atchekho/run/test3d_1cpu_16x16x8/fieldline0000.vtk")
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
    p.SetMin(r0)
    #p.SetMax(5.1)
    p.SetMinFlag(1)
    #p.SetMaxFlag(1)
    p.SetContourNLevels(1)
    SetPlotOptions(p)
    DrawPlots()
    vt = get_visit_time()
    #ChangeActivePlotsVar("Rvar")
    #
    #
    #
    AddPlot("Streamline","B")
    #Or:
    #SetActivePlot(0)
    #p=GetPlotOptions()
    omega = 0.2
    phi0 = 0
    fp = compute_footpoints(r0 = r0, Rlc=1/omega, npts=40, whichpole="both", alpha_y=np.pi*60./180., alpha_z=phi0 + omega*vt)
    #print fp
    p=StreamlineAttributes()
    p.sourceType = p.SpecifiedPointList  # SpecifiedPoint, SpecifiedPointList, SpecifiedLine, SpecifiedCircle, SpecifiedPlane, SpecifiedSphere, SpecifiedBox, Selection
    p.SetShowSeeds(0)
    #concatenate points in a tuple (3 numbers per point)
    p.SetPointList(tuple(fp))
    p.SetTermDistance(100)
    #0/1 - backward/forward
    #2   - both directions
    p.SetStreamlineDirection(2)
    p.SetShowSeeds(0)
    p.displayBegin = 0
    p.displayEnd = 15
    p.displayBeginFlag = 0
    p.displayEndFlag = 1
    # p.referenceTypeForDisplay = p.Distance  # Distance, Time, Step
    DefineScalarExpression("fcolor", "B_magnitude^0.25")
    # p.coloringVariable = "B_magnitude"
    # p.coloringMethod = p.ColorByVariable  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    # p.colorTableName = "Default"
    set_fieldline_attribs(p)
    SetPlotOptions(p)
    #
    # Begin spontaneous state
    View3DAtts = View3DAttributes()
    View3DAtts.viewNormal = (0, -1, 0)
    View3DAtts.focus = (0, 0, 0)
    View3DAtts.viewUp = (0, 0, 1)
    View3DAtts.viewAngle = 30
    View3DAtts.parallelScale = 173.205
    View3DAtts.nearPlane = -346.41
    View3DAtts.farPlane = 346.41
    View3DAtts.imagePan = (0, 0)
    View3DAtts.imageZoom = 14.421
    View3DAtts.perspective = 1
    View3DAtts.eyeAngle = 2
    View3DAtts.centerOfRotationSet = 0
    View3DAtts.centerOfRotation = (0, 0, 0)
    View3DAtts.axis3DScaleFlag = 0
    View3DAtts.axis3DScales = (1, 1, 1)
    View3DAtts.shear = (0, 0, 1)
    SetView3D(View3DAtts)
    # End spontaneous state
    DrawPlots()
    #SetActivePlot(0)
    #see http://visitusers.org/index.php?title=Using_pick_to_create_curves
    #ZonePick((2,0,0),("TIME"))
    # For moviemaking, you'll need to save off the image
    # Set the save window attributes.
    s = SaveWindowAttributes()
    s.format = s.PNG
    s.fileName = "frame%04d.png" % no
    SetSaveWindowAttributes(s)
    name = SaveWindow()
    print( "Saved image name = %s" % name )
    if cdb:
        SetActivePlots((0, 1, 2))
        DeleteActivePlots()
        CloseDatabase(dbname)

def set_fieldline_attribs(StreamlineAtts):
    StreamlineAtts.pointSource = (0, 0, 0)
    StreamlineAtts.lineStart = (0, 0, 0)
    StreamlineAtts.lineEnd = (1, 0, 0)
    StreamlineAtts.planeOrigin = (0, 0, 0)
    StreamlineAtts.planeNormal = (0, 0, 1)
    StreamlineAtts.planeUpAxis = (0, 1, 0)
    StreamlineAtts.radius = 1
    StreamlineAtts.sphereOrigin = (0, 0, 0)
    StreamlineAtts.boxExtents = (0, 1, 0, 1, 0, 1)
    StreamlineAtts.useWholeBox = 1
    StreamlineAtts.sampleDensity0 = 2
    StreamlineAtts.sampleDensity1 = 2
    StreamlineAtts.sampleDensity2 = 2
    StreamlineAtts.coloringMethod = StreamlineAtts.ColorByVariable  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    StreamlineAtts.colorTableName = "Default"
    StreamlineAtts.singleColor = (0, 0, 0, 255)
    StreamlineAtts.legendFlag = 1
    StreamlineAtts.lightingFlag = 1
    StreamlineAtts.streamlineDirection = StreamlineAtts.Both  # Forward, Backward, Both
    StreamlineAtts.maxSteps = 1000
    StreamlineAtts.terminateByDistance = 0
    StreamlineAtts.termDistance = 100
    StreamlineAtts.terminateByTime = 0
    StreamlineAtts.termTime = 10
    StreamlineAtts.maxStepLength = 0.1
    StreamlineAtts.limitMaximumTimestep = 0
    StreamlineAtts.maxTimeStep = 0.1
    StreamlineAtts.relTol = 0.0001
    StreamlineAtts.absTolSizeType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.absTolAbsolute = 1e-06
    StreamlineAtts.absTolBBox = 1e-06
    StreamlineAtts.fieldType = StreamlineAtts.Default  # Default, M3DC12DField, M3DC13DField, NIMRODField, FlashField
    StreamlineAtts.fieldConstant = 1
    StreamlineAtts.velocitySource = (0, 0, 0)
    StreamlineAtts.integrationType = StreamlineAtts.DormandPrince  # Euler, Leapfrog, DormandPrince, AdamsBashforth, RK4, M3DC12DIntegrator
    StreamlineAtts.streamlineAlgorithmType = StreamlineAtts.VisItSelects  # LoadOnDemand, ParallelStaticDomains, MasterSlave, VisItSelects
    StreamlineAtts.maxStreamlineProcessCount = 10
    StreamlineAtts.maxDomainCacheSize = 3
    StreamlineAtts.workGroupSize = 32
    StreamlineAtts.pathlines = 0
    StreamlineAtts.pathlinesOverrideStartingTimeFlag = 0
    StreamlineAtts.pathlinesOverrideStartingTime = 0
    StreamlineAtts.pathlinesCMFE = StreamlineAtts.POS_CMFE  # CONN_CMFE, POS_CMFE
    StreamlineAtts.coordinateSystem = StreamlineAtts.AsIs  # AsIs, CylindricalToCartesian, CartesianToCylindrical
    StreamlineAtts.phiScalingFlag = 0
    StreamlineAtts.phiScaling = 1
    StreamlineAtts.coloringVariable = "fcolor"
    StreamlineAtts.legendMinFlag = 0
    StreamlineAtts.legendMaxFlag = 0
    StreamlineAtts.legendMin = 0
    StreamlineAtts.legendMax = 1
    StreamlineAtts.displayBegin = 0
    StreamlineAtts.displayEnd = 15
    StreamlineAtts.displayBeginFlag = 0
    StreamlineAtts.displayEndFlag = 1
    StreamlineAtts.referenceTypeForDisplay = StreamlineAtts.Distance  # Distance, Time, Step
    StreamlineAtts.displayMethod = StreamlineAtts.Lines  # Lines, Tubes, Ribbons
    StreamlineAtts.tubeSizeType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.tubeRadiusAbsolute = 0.125
    StreamlineAtts.tubeRadiusBBox = 0.005
    StreamlineAtts.ribbonWidthSizeType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.ribbonWidthAbsolute = 0.125
    StreamlineAtts.ribbonWidthBBox = 0.01
    StreamlineAtts.lineWidth = 2
    StreamlineAtts.showSeeds = 0
    StreamlineAtts.seedRadiusSizeType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.seedRadiusAbsolute = 1
    StreamlineAtts.seedRadiusBBox = 0.015
    StreamlineAtts.showHeads = 0
    StreamlineAtts.headDisplayType = StreamlineAtts.Sphere  # Sphere, Cone
    StreamlineAtts.headRadiusSizeType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.headRadiusAbsolute = 0.25
    StreamlineAtts.headRadiusBBox = 0.02
    StreamlineAtts.headHeightRatio = 2
    StreamlineAtts.opacityType = StreamlineAtts.FullyOpaque  # FullyOpaque, Constant, Ramp, VariableRange
    StreamlineAtts.opacityVariable = ""
    StreamlineAtts.opacity = 1
    StreamlineAtts.opacityVarMin = 0
    StreamlineAtts.opacityVarMax = 1
    StreamlineAtts.opacityVarMinFlag = 0
    StreamlineAtts.opacityVarMaxFlag = 0
    StreamlineAtts.tubeDisplayDensity = 10
    StreamlineAtts.geomDisplayQuality = StreamlineAtts.Medium  # Low, Medium, High, Super
    StreamlineAtts.sampleDistance0 = 10
    StreamlineAtts.sampleDistance1 = 10
    StreamlineAtts.sampleDistance2 = 10
    StreamlineAtts.fillInterior = 1
    StreamlineAtts.randomSamples = 0
    StreamlineAtts.randomSeed = 0
    StreamlineAtts.numberOfRandomSamples = 1
    StreamlineAtts.forceNodeCenteredData = 0
    StreamlineAtts.issueTerminationWarnings = 1
    StreamlineAtts.issueStiffnessWarnings = 1
    StreamlineAtts.issueCriticalPointsWarnings = 1
    StreamlineAtts.criticalPointThreshold = 0.001
    StreamlineAtts.varyTubeRadius = StreamlineAtts.None  # None, Scalar
    StreamlineAtts.varyTubeRadiusFactor = 10
    StreamlineAtts.varyTubeRadiusVariable = ""
    StreamlineAtts.correlationDistanceAngTol = 5
    StreamlineAtts.correlationDistanceMinDistAbsolute = 1
    StreamlineAtts.correlationDistanceMinDistBBox = 0.005
    StreamlineAtts.correlationDistanceMinDistType = StreamlineAtts.FractionOfBBox  # Absolute, FractionOfBBox
    StreamlineAtts.selection = ""


def get_visit_time():
    SuppressQueryOutputOn() 
    Query("Time")
    t = GetQueryOutputValue()
    SuppressQueryOutputOff()
    return t

def rotate_around_y(r, th, ph, alpha=0):
    """Returns x, y, z of the rotated vector"""
    alpha = -alpha
    xp = r*(-np.cos(th)*np.sin(alpha)+np.sin(th)*np.cos(ph)*np.cos(alpha))
    yp = r*np.sin(th)*np.sin(ph)
    zp = r*(np.cos(th)*np.cos(alpha)+np.sin(th)*np.cos(ph)*np.sin(alpha))
    return xp, yp, zp

def rotate_around_z(xp, yp, zp, alpha=0):
    x = xp * np.cos(alpha) - yp * np.sin(alpha)
    y = xp * np.sin(alpha) + yp * np.cos(alpha)
    z = zp
    return x, y, z

def rotate_around_y_z(r, th, ph, alpha_y=0, alpha_z=0):
    xp, yp, zp = rotate_around_y(r,th,ph,alpha=alpha_y)
    x, y, z = rotate_around_z(xp,yp,zp,alpha=alpha_z)
    return x, y, z 

def compute_footpoints(r0 = 1.5, Rlc=5,npts=10,whichpole="both", alpha_y=0, alpha_z=0):
    thp = (r0/Rlc)**0.5
    ph = 2*np.pi*np.arange(0,1,1./npts)
    r = r0 + np.zeros_like(ph)
    th = thp + np.zeros_like(ph)
    xyz=[]
    if whichpole=="dn" or whichpole=="both":
        xyz += zip2visit(rotate_around_y_z(r, th, ph,alpha_y=alpha_y, alpha_z=alpha_z))
    if whichpole=="up" or whichpole=="both":
        xyz += zip2visit(rotate_around_y_z(r, np.pi-th, ph,alpha_y=alpha_y, alpha_z=alpha_z))
    return xyz

def zip2visit(xyz):
    return list(np.ravel(zip(*xyz)))

def get_visit_footpoints(r0 = 1.5, Rlc=5, npts=10, whichpole="up", alpha_y=0, alpha_z=0):
    zip2visit(compute_footpoints(r0=r0, Rlc=Rlc, npys=npts, whichpole=whichpole, alpha_y=alpha_y, alpha_z=alpha_z))


if __name__ == "__main__":
    testvariable = 3
