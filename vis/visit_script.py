def VisitScript():
    # You can run this by:
    #     Saving the script below to "script.py"
    #     Running "visit -cli -s script.py" 
    OpenDatabase("/Users/atchekho/run2/fixdt_x2_60/fieldline0073.vtk")
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
    p.SetMin(1.04)
    #p.SetMax(5.1)
    p.SetMinFlag(1)
    #p.SetMaxFlag(1)
    p.SetContourNLevels(1)
    SetPlotOptions(p)
    #ChangeActivePlotsVar("Rvar")
    #
    #
    DrawPlots()

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
    SuppressQueryOutputOn()
    Query("Time")
    t = GetQueryOutputValue()
    SuppressQueryOutputOff()

def rotate_around_y(r, theta, phi, angle=0):
    """Returns x, y, z of the rotated vector"""

if __name__ == "__main__":
    testvariable = 3
