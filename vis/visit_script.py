import numpy as np
import pdb as pdb

def mkvmov():
    for no in xrange(0,74):
        VisitScript(no=no,r0=1.05)

def VisitScript(no=73,r0=1.05,cdb=True,pf=1,i=60.):
    # You can run this by:
    #     Saving the script below to "script.py"
    #     Running "visit -cli -s script.py" 
    #OpenDatabase("/Users/atchekho/run2/fixdt_x2_60/fieldline0000.vtk")
    #dbname = "/home/atchekho/run/fixdt_x2_60/fieldline%04d.vtk" % no
    # dbname = "/home/atchekho/run2/fixdt_60/fieldline%04d.vtk" % no
    dbname = "/home/atchekho/run2/fixdt_x2_60/avg_31_73.vtk"
    #dbname = "/home/atchekho/run2/fixdt_90/avg_61_120.vtk"
    OpenDatabase(dbname)
    #OpenDatabase("/Users/atchekho/run/test3d_1cpu_16x16x8/fieldline0000.vtk")
    DefineScalarExpression("Rsq", "x*x+y*y")
    DefineScalarExpression("bsqow", "bsq/(rho+4./3.*ug)")
    AddPlot("Contour","Rsq")
    p=ContourAttributes()
    #p.contourMethod="Value"
    #print p
    #p.contourValue=1.04
    #p.singleColor=(255,0,0,128)  #red, 50% transparent
    p.singleColor = (192, 192, 192, 128) #grey, 50% transparent
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
    #vt = 0
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
    fp = compute_footpoints(r0 = r0, Rlc=1/omega, npts=40, whichpole="up", alpha_y=np.pi*i/180., alpha_z=phi0 + omega*vt,pf=pf)
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
    set_fieldline_attribs(p,whichpole=0)
    SetPlotOptions(p)
    #
    AddPlot("Streamline","B")
    #Or:
    #SetActivePlot(0)
    #p=GetPlotOptions()
    omega = 0.2
    phi0 = 0
    fp = compute_footpoints(r0 = r0, Rlc=1/omega, npts=40, whichpole="dn", alpha_y=np.pi*i/180., alpha_z=phi0 + omega*vt, pf=pf)
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
    # p.coloringVariable = "B_magnitude"
    # p.coloringMethod = p.ColorByVariable  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    # p.colorTableName = "Default"
    set_fieldline_attribs(p,whichpole=1)
    SetPlotOptions(p)
    #
    # visualize current sheet
    if(0):
        PlotCurrentSheet()
    elif(1):
        PlotChargeInCurrentSheet()
    #
    # visualize velocity in current sheet
    PlotVelocityInCurrentSheet()
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

def PlotChargeInCurrentSheet():
    #Plot charge times the square of spherical polar radius (this makes charge visible throughout)
    DefineScalarExpression("rhocrsq","rhoc*rsphsq")
    AddPlot("Pseudocolor", "rhocrsq", 1, 0)
    AddOperator("Threshold", 0)
    SetActivePlots(4)
    SetActivePlots(4)
    ThresholdAtts = ThresholdAttributes()
    ThresholdAtts.outputMeshType = 0
    ThresholdAtts.listedVarNames = ("bsqow")
    ThresholdAtts.zonePortions = (0)
    ThresholdAtts.lowerBounds = (0)
    ThresholdAtts.upperBounds = (2)
    ThresholdAtts.defaultVarName = "rhocrsq"
    ThresholdAtts.defaultVarIsScalar = 1
    SetOperatorOptions(ThresholdAtts, 0)
    AddOperator("Clip", 0)
    DemoteOperator(1, 0)
    ClipAtts = ClipAttributes()
    ClipAtts.quality = ClipAtts.Fast  # Fast, Accurate
    ClipAtts.funcType = ClipAtts.Sphere  # Plane, Sphere
    ClipAtts.plane1Status = 1
    ClipAtts.plane2Status = 0
    ClipAtts.plane3Status = 0
    ClipAtts.plane1Origin = (0, 0, 0)
    ClipAtts.plane2Origin = (0, 0, 0)
    ClipAtts.plane3Origin = (0, 0, 0)
    ClipAtts.plane1Normal = (1, 0, 0)
    ClipAtts.plane2Normal = (0, 1, 0)
    ClipAtts.plane3Normal = (0, 0, 1)
    ClipAtts.planeInverse = 0
    ClipAtts.planeToolControlledClipPlane = ClipAtts.Plane1  # None, Plane1, Plane2, Plane3
    ClipAtts.center = (0, 0, 0)
    ClipAtts.radius = 15
    ClipAtts.sphereInverse = 1
    SetOperatorOptions(ClipAtts, 0)
    DrawPlots()
    # MAINTENANCE ISSUE: SetSuppressMessagesRPC is not handled in Logging.C. Please contact a VisIt developer.
    SaveSession("/Users/atchekho/.visit/crash_recovery.session")
    # MAINTENANCE ISSUE: SetSuppressMessagesRPC is not handled in Logging.C. Please contact a VisIt developer.
    PseudocolorAtts = PseudocolorAttributes()
    PseudocolorAtts.legendFlag = 1
    PseudocolorAtts.lightingFlag = 1
    PseudocolorAtts.minFlag = 0
    PseudocolorAtts.maxFlag = 0
    PseudocolorAtts.centering = PseudocolorAtts.Natural  # Natural, Nodal, Zonal
    PseudocolorAtts.scaling = PseudocolorAtts.Linear  # Linear, Log, Skew
    PseudocolorAtts.limitsMode = PseudocolorAtts.OriginalData  # OriginalData, CurrentPlot
    PseudocolorAtts.min = 0
    PseudocolorAtts.max = 1
    PseudocolorAtts.pointSize = 0.05
    PseudocolorAtts.pointType = PseudocolorAtts.Point  # Box, Axis, Icosahedron, Point, Sphere
    PseudocolorAtts.skewFactor = 1
    PseudocolorAtts.opacity = 1
    PseudocolorAtts.colorTableName = "hot_and_cold"
    PseudocolorAtts.invertColorTable = 0
    PseudocolorAtts.smoothingLevel = 0
    PseudocolorAtts.pointSizeVarEnabled = 0
    PseudocolorAtts.pointSizeVar = "default"
    PseudocolorAtts.pointSizePixels = 2
    PseudocolorAtts.lineStyle = PseudocolorAtts.SOLID  # SOLID, DASH, DOT, DOTDASH
    PseudocolorAtts.lineWidth = 0
    PseudocolorAtts.opacityType = PseudocolorAtts.Explicit  # Explicit, ColorTable
    SetPlotOptions(PseudocolorAtts)
    PseudocolorAtts = PseudocolorAttributes()
    PseudocolorAtts.legendFlag = 1
    PseudocolorAtts.lightingFlag = 1
    PseudocolorAtts.minFlag = 1
    PseudocolorAtts.maxFlag = 1
    PseudocolorAtts.centering = PseudocolorAtts.Natural  # Natural, Nodal, Zonal
    PseudocolorAtts.scaling = PseudocolorAtts.Linear  # Linear, Log, Skew
    PseudocolorAtts.limitsMode = PseudocolorAtts.OriginalData  # OriginalData, CurrentPlot
    PseudocolorAtts.min = -15
    PseudocolorAtts.max = 15
    PseudocolorAtts.pointSize = 0.05
    PseudocolorAtts.pointType = PseudocolorAtts.Point  # Box, Axis, Icosahedron, Point, Sphere
    PseudocolorAtts.skewFactor = 1
    PseudocolorAtts.opacity = 1
    PseudocolorAtts.colorTableName = "hot_and_cold"
    PseudocolorAtts.invertColorTable = 0
    PseudocolorAtts.smoothingLevel = 0
    PseudocolorAtts.pointSizeVarEnabled = 0
    PseudocolorAtts.pointSizeVar = "default"
    PseudocolorAtts.pointSizePixels = 2
    PseudocolorAtts.lineStyle = PseudocolorAtts.SOLID  # SOLID, DASH, DOT, DOTDASH
    PseudocolorAtts.lineWidth = 0
    PseudocolorAtts.opacityType = PseudocolorAtts.Explicit  # Explicit, ColorTable
    SetPlotOptions(PseudocolorAtts)


def PlotVelocityInCurrentSheet():
    AddPlot("Streamline", "v", 1, 0)
    StreamlineAtts = StreamlineAttributes()
    StreamlineAtts.sourceType = StreamlineAtts.SpecifiedPointList  # SpecifiedPoint, SpecifiedPointList, SpecifiedLine, SpecifiedCircle, SpecifiedPlane, SpecifiedSphere, SpecifiedBox, Selection
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
    StreamlineAtts.pointList = (2.7449, 5.45317, 9.90786, 12.3581,
    -2.14433, 1.39096, 0.129415, -5.27178, -9.04806, -4.75934,
    -5.5174, -2.30835, 5.31036, 4.58074, 1.84843, 9.68432, -0.237736,
    7.00192, 6.88364, -0.849017, -5.27709, -0.487008, -6.60219,
    -10.7438, 9.25125, -1.14103, 9.55298, 6.15627, -1.06821, -6.40353,
    -11.0363, 4.25715, -4.39692, 11.8996, -5.98975, 0.818459, 8.47443,
    3.75645, 1.72588, 2.94552, -3.41467, -6.93168, -4.24939, -4.92622,
    -0.560244, 10.2843, -9.79151, -3.37229, 0.34527, 4.68071, 7.61694,
    3.86373, 2.44668, -2.66537, 6.35285, 3.59884, 7.12436, 2.78533,
    5.5335, 5.47783, 4.76739, -9.47119, -9.37591, 4.92285, 3.84183,
    9.10179, 2.45108, -5.52955, -7.5573, 8.85407, -5.01577, -6.27032,
    -1.74382, -5.27864, -3.42549, -1.375, -4.96861, -4.55856, 4.07876,
    -4.72841, -9.10179, 8.02799, 4.04095, 3.59424, 5.5616, 5.29509,
    2.43274, 4.66134, 4.89596, 6.92806, 2.97181, -1.49589, -3.76263,
    -7.57463, -3.81275, -4.66739, -1.92029, 3.38978, 3.99272,
    -1.37992, 6.14508, 8.7135, 0.092553, -3.77019, -4.71212, -3.15684,
    -3.65965, 1.79651, 3.04031, 6.04005, 9.85615, 9.32095, 3.07922,
    3.64886, -5.9945, -3.79598, 1.50209, -5.99974, -5.17541, -3.62923,
    -6.06679, -4.73457, -9.14589, 5.68204, 4.90136, 5.70935, 8.11949,
    3.59912, 2.34088, 0.109966, 4.4795, 3.07421, 5.98059, 3.01038,
    -1.59008, -2.09499, -5.43106, -8.05358, 4.31567, 4.10886,
    -0.958784, 8.81016, -4.9909, -6.94695, 4.16084, 1.37456, -3.68714,
    0.335631, 4.55003, 4.91133, -9.0035, -2.0218, -2.92333, -0.126152,
    5.13886, 6.75615, -6.03599, -3.41935, -7.84542, 6.23077, 4.86254,
    5.14185, 6.73607, 4.74408, 0.709499, -2.56602, -5.0978, -1.9634,
    0.581602, -7.88456, -9.39593, -2.41367, 2.79812, 4.17908,
    -2.62576, 2.26499, 4.55765, -8.77782, 1.08264, -11.0507, 3.43908,
    2.17777, -2.94224, -0.095363, -3.88466, -4.61815, 7.81684,
    -4.94996, -6.01928, -6.21266, -4.84841, -1.66834, -10.3932,
    0.255138, 2.74015, -5.52693, -3.8925, -6.92806, -4.21552, 8.37479,
    10.6034, 8.0334, -6.92966, -6.18324, -5.11685, 4.41382, 7.64218,
    2.73652, 5.43654, 6.23767, -1.95355, -5.06437, -2.98759, -6.41408,
    1.44033, 6.41441, 7.99376, 3.08353, -1.59521, -2.77414, -4.89704,
    -6.05867, -3.35596, -3.19514, 2.26158, 0.428229, 5.80535, 8.05358,
    -8.1849, -1.42022, 3.32212, 6.88091, 3.89798, 6.01701, 4.89013,
    5.13627, 6.58801, -5.00418, -3.52434, 2.27512, -0.342248, 4.63974,
    5.81292, 9.95969, 1.72817, 1.3729, 3.00238, 5.29994, 1.13411,
    6.30632, -4.44141, -8.30329, -7.9111, -3.05165, 2.0137, 3.43838,
    -2.68334, -6.03419, 1.76746, -3.98734, -6.03419, 1.18615,
    -5.28217, -9.28887, 5.71903, -11.3618, -6.20823, 8.19996, 2.7089,
    -1.17288, -12.5143, 5.5472, -1.1788, 0.344555, 4.67102, 5.04193,
    1.33377, -3.00893, -5.05908, 3.18197, -8.24894, -9.99885,
    -6.02812, -2.67208, 3.02019, 4.84544, -3.41255, -6.70241, 4.42846,
    -0.326663, -5.02182, 0.542361, 4.39735, 5.26567, -5.46897,
    -0.134255, 5.33797, 1.80472, -4.07137, -7.22747)
    StreamlineAtts.sampleDensity0 = 2
    StreamlineAtts.sampleDensity1 = 2
    StreamlineAtts.sampleDensity2 = 2
    StreamlineAtts.coloringMethod = StreamlineAtts.Solid  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    StreamlineAtts.colorTableName = "PuOr"
    StreamlineAtts.singleColor = (255, 153, 0, 255)
    StreamlineAtts.legendFlag = 1
    StreamlineAtts.lightingFlag = 1
    StreamlineAtts.streamlineDirection = StreamlineAtts.Forward  # Forward, Backward, Both
    StreamlineAtts.maxSteps = 1000
    StreamlineAtts.terminateByDistance = 0
    StreamlineAtts.termDistance = 10
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
    StreamlineAtts.coloringVariable = ""
    StreamlineAtts.legendMinFlag = 0
    StreamlineAtts.legendMaxFlag = 0
    StreamlineAtts.legendMin = 0
    StreamlineAtts.legendMax = 1
    StreamlineAtts.displayBegin = 0
    StreamlineAtts.displayEnd = 1
    StreamlineAtts.displayBeginFlag = 0
    StreamlineAtts.displayEndFlag = 1
    StreamlineAtts.referenceTypeForDisplay = StreamlineAtts.Distance  # Distance, Time, Step
    StreamlineAtts.displayMethod = StreamlineAtts.Tubes  # Lines, Tubes, Ribbons
    StreamlineAtts.tubeSizeType = StreamlineAtts.Absolute  # Absolute, FractionOfBBox
    StreamlineAtts.tubeRadiusAbsolute = 0.05
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
    SetPlotOptions(StreamlineAtts)


def PlotCurrentSheet():
    AddPlot("Contour", "bsqow", 1, 1)
    ContourAtts = ContourAttributes()
    ContourAtts.defaultPalette.smoothing = ContourAtts.defaultPalette.None  # None, Linear, CubicSpline
    ContourAtts.defaultPalette.equalSpacingFlag = 1
    ContourAtts.defaultPalette.discreteFlag = 1
    ContourAtts.defaultPalette.externalFlag = 0
    ContourAtts.changedColors = ()
    ContourAtts.colorType = ContourAtts.ColorBySingleColor  # ColorBySingleColor, ColorByMultipleColors, ColorByColorTable
    ContourAtts.colorTableName = "Default"
    ContourAtts.invertColorTable = 0
    ContourAtts.legendFlag = 1
    ContourAtts.lineStyle = ContourAtts.SOLID  # SOLID, DASH, DOT, DOTDASH
    ContourAtts.lineWidth = 0
    ContourAtts.singleColor = (255, 0, 255, 255)
    ContourAtts.contourValue = (2)
    ContourAtts.contourMethod = ContourAtts.Value  # Level, Value, Percent
    ContourAtts.scaling = ContourAtts.Linear  # Linear, Log
    ContourAtts.wireframe = 0
    SetPlotOptions(ContourAtts)
    # Clipping
    AddOperator("Clip", 0)
    SetActivePlots(4)
    ClipAtts = ClipAttributes()
    ClipAtts.quality = ClipAtts.Fast  # Fast, Accurate
    ClipAtts.funcType = ClipAtts.Sphere  # Plane, Sphere
    ClipAtts.center = (0, 0, 0)
    ClipAtts.radius = 15
    ClipAtts.sphereInverse = 1
    SetOperatorOptions(ClipAtts, 0)



def set_fieldline_attribs(StreamlineAtts,whichpole=0):
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
    StreamlineAtts.coloringMethod = StreamlineAtts.Solid  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    #StreamlineAtts.coloringMethod = StreamlineAtts.ColorByVariable  # Solid, ColorBySpeed, ColorByVorticity, ColorByLength, ColorByTime, ColorBySeedPointID, ColorByVariable, ColorByCorrelationDistance
    StreamlineAtts.colorTableName = "Default"
    if whichpole==0:
        StreamlineAtts.singleColor = (0, 255, 0, 255)
    else:
        StreamlineAtts.singleColor = (0, 0, 255, 255)
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

def compute_footpoints(r0 = 1.5, Rlc=5,npts=10,whichpole="both", alpha_y=0, alpha_z=0,pf=1.):
    thp = pf*(r0/Rlc)**0.5
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
    VisitScript(cdb=False)
