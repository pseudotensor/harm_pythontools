import numpy as np
import matplotlib.pyplot as plt

def grid3d(dumpname): #read grid dump file: header and body
    #The internal cell indices along the three axes: (ti, tj, tk)
    #The internal uniform coordinates, (x1, x2, x3), are mapped into the physical
    #non-uniform coordinates, (r, h, ph), which correspond to radius (r), polar angle (theta), and toroidal angle (phi).
    #There are more variables, e.g., dxdxp, which is the Jacobian of (x1,x2,x3)->(r,h,ph) transformation, that I can
    #go over, if needed.
    global nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
    print( "Reading grid from " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    #First line of grid dump file is a text line that contains general grid information:
    header = gin.readline().split()
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    #cell size in internal coordintes
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    #other information: 
    #polytropic index
    gam=float(header[11])
    #black hole spin
    a=float(header[12])
    #Spherical polar radius of the innermost radial cell
    Rin=float(header[14])
    #Spherical polar radius of the outermost radial cell
    Rout=float(header[15])
    #read grid dump per-cell data
    #
    if dumpname.endswith(".bin"):
        body = np.fromfile(gin,dtype=np.float64,count=-1) 
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        gin.close()
    else:
        gin.close()
        gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:,:].view() 
    #get the right order of indices by reversing the order of indices i,j(,k)
    #conn=gd[9:73].view().reshape((4,4,4,nx,ny,nz), order='F').transpose(2,1,0,3,4,5)
    #contravariant metric components, g^{\mu\nu}
    gn3 = gd[73:89].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    #covariant metric components, g_{\mu\nu}
    gv3 = gd[89:105].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    #metric determinant
    gdet = gd[105]
    ck = gd[106:110].view().reshape((4,nx,ny,nz), order='F')
    #grid mapping Jacobian
    dxdxp = gd[110:126].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    print( "Done!" )

def rfd(fieldlinefilename,**kwargs):
    #read information from "fieldline" file: 
    #Densities: rho, u, 
    #Velocity components: u1, u2, u3, 
    #Cell-centered magnetic field components: B1, B2, B3, 
    #Face-centered magnetic field components multiplied by metric determinant: gdetB1, gdetB2, gdetB3
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,rho,lrho,ug,uu,uut,uu,B,uux,gdetB
    #read image
    fin = open( "dumps/" + fieldlinefilename, "rb" )
    header = fin.readline().split()
    #time of the dump
    t = np.float64(header[0])
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    #cell size in internal coordintes
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    #other information: 
    #polytropic index
    gam=float(header[11])
    #black hole spin
    a=float(header[12])
    #Spherical polar radius of the innermost radial cell
    Rin=float(header[14])
    #Spherical polar radius of the outermost radial cell
    Rout=float(header[15])
    #read grid dump per-cell data
    #
    body = np.fromfile(fin,dtype=np.float32,count=-1)
    fin.close()
    d=body.view().reshape((-1,nx,ny,nz),order='F')
    #rho, u, -hu_t, -T^t_t/U0, u^t, v1,v2,v3,B1,B2,B3
    #matter density in the fluid frame
    rho=d[0,:,:,:]
    lrho = np.log10(rho)
    #matter internal energy in the fluid frame
    ug=d[1,:,:,:]
    #d[4] is the time component of 4-velocity, u^t
    #d[5:8] are 3-velocities, v^i
    uu=d[4:8,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldline file)
    #multiply by u^t to get 4-velocities: u^i = u^t v^i
    uu[1:4]=uu[1:4] * uu[0]
    B = np.zeros_like(uu)
    #cell-centered magnetic field components
    B[1:4,:,:,:]=d[8:11,:,:,:]
    #if the input file contains additional data
    if(d.shape[0]>=14): 
        #new image format additionally contains gdet*B^i
        gdetB = np.zeros_like(B)
        #face-centered magnetic field components multiplied by gdet
        gdetB[1:4] = d[11:14,:,:,:]
    else:
        print("No data on gdetB, approximating it.")
        gdetB = np.zeros_like(B)
        gdetB[1] = gdet * B[1]
        gdetB[2] = gdet * B[2]
        gdetB[3] = gdet * B[3]

def plco(myvar,xcoord=None,ycoord=None,**kwargs):
    plt.clf()
    plc(myvar,xcoord,ycoord,**kwargs)

def plc(myvar,xcoord=None,ycoord=None,**kwargs): #plc
    #2D plotting routine wrapper
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    if( xcoord == None or ycoord == None ):
        res = plt.contour(myvar[:,:,0].transpose(),nc,**kwargs)
    else:
        res = plt.contour(xcoord[:,:,0],ycoord[:,:,0],myvar[:,:,0],nc,**kwargs)
    if( cb == True): #use color bar
        plt.colorbar()

if __name__ == "__main__":
    #cd into the directory that contains the dumps/ directory
    #read in the grid file
    grid3d("gdump.bin")
    #read in the 0th dump file
    rfd("fieldline0000.bin")
    #plot log10(radius) vs. grid index, ti
    plt.plot( ti[:,0,0], np.log10(r[:,0,0]) )
    plt.figure((6,12))
    #theta grid lines
    plc(ti,xcoord=r*np.sin(h),ycoord=r*cos(h))
    #radial grid lines
    plc(tj,xcoord=r*np.sin(h),ycoord=r*cos(h))
    plt.xlim(0,20)
    plt.ylim(-20,20)
