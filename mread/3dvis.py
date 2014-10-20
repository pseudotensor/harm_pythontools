import matplotlib
from mayavi.scripts import mayavi2
matplotlib.use('WxAgg')
matplotlib.interactive(True)
from numpy import mgrid, empty, zeros, sin, cos, pi
from tvtk.api import tvtk
from mayavi import mlab
from scipy import ndimage
import sys
import os,glob
import numpy as np
import gc
#from mread import grid3d, rfd, cvel
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

def iofr(rval):
    rval = np.array(rval)
    if np.max(rval) < r[0,0,0]:
        return 0
    res = interp1d(r[:,0,0], ti[:,0,0], kind='linear', bounds_error = False, fill_value = 0)(rval)
    if len(res.shape)>0 and len(res)>0:
        res[rval<r[0,0,0]]*=0
        res[rval>r[nx-1,0,0]]=res[rval>r[nx-1,0,0]]*0+nx-1
    else:
        res = np.float64(res)
    return(np.floor(res+0.5).astype(int))

def amax(arg1,arg2):
    return(np.maximum(arg1,arg2))
    # arr1 = np.array(arg1)
    # arr2 = np.array(arg2)
    # #create storage array of size that's largest of arr1 and arr2
    # ret=np.zeros_like(arr1+arr2)
    # ret[arr1>=arr2]=arr1[arr1>=arr2]
    # ret[arr2>arr1]=arr2[arr2>arr1]
    # return(ret)
def amin(arg1,arg2):
    return(np.minimum(arg1,arg2))
    # arr1 = np.array(arg1)
    # arr2 = np.array(arg2)
    # #create storage array of size that's largest of arr1 and arr2
    # ret=np.zeros_like(arr1+arr2)
    # ret[arr1<=arr2]=arr1[arr1<=arr2]
    # ret[arr2<arr1]=arr2[arr2<arr1]
    # return(ret)


def mdot(a,b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
    """
    if (a.ndim == 3 and b.ndim == 3) or (a.ndim == 4 and b.ndim == 4):
          c = (a*b).sum(0)
    elif a.ndim == 5 and b.ndim == 4:
          c = np.empty(amax(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)      
          for i in range(a.shape[0]):
                c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
    elif a.ndim == 4 and b.ndim == 5:
          c = np.empty(amax(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)      
          for i in range(b.shape[1]):
                c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
    elif a.ndim == 5 and b.ndim == 5:
          c = np.empty((a.shape[0],b.shape[1],a.shape[2],a.shape[3],max(a.shape[4],b.shape[4])),dtype=a.dtype)
          for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                      c[i,j,:,:,:] = (a[i,:,:,:,:]*b[:,j,:,:,:]).sum(0)
    elif a.ndim == 5 and b.ndim == 6:
          c = np.empty((a.shape[0],b.shape[1],b.shape[2],max(a.shape[2],b.shape[3]),max(a.shape[3],b.shape[4]),max(a.shape[4],b.shape[5])),dtype=a.dtype)
          for mu in range(c.shape[0]):
              for k in range(c.shape[1]):
                  for l in range(c.shape[2]):
                      c[mu,k,l,:,:,:] = (a[mu,:,:,:,:]*b[:,k,l,:,:,:]).sum(0)
    else:
           raise Exception('mdot', 'wrong dimensions')
    return c


def set_numnprs(fname="nprlistinfo.dat"):
    fin = open( fname , "rb" )
    header1 = fin.readline().split() #numlines, numversion
    header2 = fin.readline().split() #NPR (conserved)
    NPR = len(header2)
    header3 = fin.readline().split() #NPR2INTERP
    header4 = fin.readline().split() #NPR2NOTINTERP
    header5 = fin.readline().split() #NPRBOUND
    header6 = fin.readline().split() #NPRFLUXBOUND
    header7 = fin.readline().split() #NPRDUMP
    NPRDUMP = len(header7)
    header8 = fin.readline().split() #NPRINVERT
    return((NPR, NPRDUMP))
    
def set_dumpversions(header):
    global numheaderitems, numcolumns, whichdump, whichdumpversion,_is,_ie,_js,_je,_ks,_ke
    global MBH,QBH,EP3,THETAROT
    #
    numheaderitems=len(header)
    if numheaderitems==32:
        print("Found 32 header items, reading them in")  ; sys.stdout.flush()
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=myfloatalt(float(header[21]))
        THETAROT=myfloatalt(float(header[22]))
        #
        _is=int(header[23])
        _ie=int(header[24])
        _js=int(header[25])
        _je=int(header[26])
        _ks=int(header[27])
        _ke=int(header[28])
        whichdump=int(header[29])
        whichdumpversion=int(header[30])
        numcolumns=int(header[31])
    #
    if numheaderitems==31:
        print("Found 31 header items, reading them in and setting THETAROT=0.0\n")  ; sys.stdout.flush()
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=myfloatalt(float(header[21]))
        THETAROT=0.0
        #
        _is=int(header[22])
        _ie=int(header[23])
        _js=int(header[24])
        _je=int(header[25])
        _ks=int(header[26])
        _ke=int(header[27])
        whichdump=int(header[28])
        whichdumpversion=int(header[29])
        numcolumns=int(header[30])
    #
    if numheaderitems==30:
        print("Found 30 header items, reading them in and setting EP3=THETAROT=0.0\n")  ;
        MBH=myfloatalt(float(header[19]))
        QBH=myfloatalt(float(header[20]))
        EP3=0.0
        THETAROT=0.0
        #
        _is=int(header[21])
        _ie=int(header[22])
        _js=int(header[23])
        _je=int(header[24])
        _ks=int(header[25])
        _ke=int(header[26])
        whichdump=int(header[27])
        whichdumpversion=int(header[28])
        numcolumns=int(header[29])

def myfloatalt(f):
    return( np.float64(f) )
    

def myfloat(f,acc=1):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1:
        return( np.float32(f) )
    else:
        return( np.float64(f) )

def cvel():
    global ud,etad, etau, gamma, vu, vd, bu, bd, bsq
    if 'ud' in globals():
        del ud
    if 'etad' in globals():
        del etad 
    if 'etau' in globals():
        del etau 
    if 'gamma' in globals():
        del gamma 
    if 'vu' in globals():
        del vu 
    if 'vd' in globals():
        del vd 
    if 'bu' in globals():
        del bu 
    if 'bd' in globals():
        del bd 
    if 'bsq' in globals():
        del bsq
    ud = mdot(gv3,uu)                  #g_mn u^n
    etad = np.zeros_like(uu)
    etad[0] = -1/(-gn3[0,0])**0.5      #ZAMO frame velocity (definition)
    etau = mdot(gn3,etad)
    gamma=-mdot(uu,etad)                #Lorentz factor as measured by ZAMO
    vu = uu - gamma*etau               #u^m = v^m + gamma eta^m
    vd = mdot(gv3,vu)
    bu=np.empty_like(uu)              #allocate memory for bu
    #set component per component
    bu[0]=mdot(B[1:4], ud[1:4])             #B^i u_i
    bu[1:4]=(B[1:4] + bu[0]*uu[1:4])/uu[0]  #b^i = (B^i + b^t u^i)/u^t
    bd=mdot(gv3,bu)
    bsq=mdot(bu,bd)

def rfd(fieldlinefilename,**kwargs):
    #read information from "fieldline" file: 
    #Densities: rho, u, 
    #Velocity components: u1, u2, u3, 
    #Cell-centered magnetic field components: B1, B2, B3, 
    #Face-centered magnetic field components multiplied by metric determinant: gdetB1, gdetB2, gdetB3
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,rho,lrho,ug,uu,uut,uu,B,uux,gdetB,rhor,r,h,ph,ti,tj,tk,gdetF,fdbody,OmegaNS,AlphaNS,Bstag,defcoord,numheaderitems,numcolumns
    #read image
    if 'rho' in globals():
        del rho
    if 'lrho' in globals():
        del lrho
    if 'ug' in globals():
        del ug
    if 'uu' in globals():
        del uu
    if 'uut' in globals():
        del uut
    if 'uu' in globals():
        del uu
    if 'B' in globals():
        del B
    if 'uux' in globals():
        del uux
    if 'gdetB' in globals():
        del gdetB
    if 'rhor' in globals():
        del rhor
    if 'gdetF' in globals():
        del gdetF
    if 'fdbody' in globals():
        del fdbody
    print("Opening %s ..." % fieldlinefilename)
    fin = open( "dumps/" + fieldlinefilename, "rb" )
    header = fin.readline().split()
    set_dumpversions(header)
    #time of the dump
    t = myfloat(np.float64(header[0]))
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    #cell size in internal coordintes
    _dx1=myfloat(float(header[7]))
    _dx2=myfloat(float(header[8]))
    _dx3=myfloat(float(header[9]))
    #other information: 
    #polytropic index
    gam=myfloat(float(header[11]))
    #black hole spin
    a=myfloat(float(header[12]))
    rhor = 1+(1-a**2)**0.5
    R0=myfloat(float(header[13]))
    #Spherical polar radius of the innermost radial cell
    Rin=myfloat(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloat(float(header[15]))
    defcoord = myfloat(float(header[18]))
    #
    if os.path.isfile("coordparms.dat"):
        coordparams = np.loadtxt( "coordparms.dat", 
                      dtype=np.float64, 
                      skiprows=0, 
                      unpack = False )
        if defcoord == 3010: #SNSCOORDS
            if len(coordparams)>=14:
                OmegaNS = coordparams[13]
                AlphaNS = coordparams[14]
            else:
                OmegaNS = 0
                AlphaNS = 0
        elif defcoord == 3000: #SJETCOORDS
            if len(coordparams)>=27:
                OmegaNS = coordparams[26]
                AlphaNS = coordparams[27]
            else:
                OmegaNS = 0
                AlphaNS = 0
    else:
        OmegaNS = 0
        AlphaNS = 0
    #read grid dump per-cell data
    #
    body = np.fromfile(fin,dtype=np.float32,count=-1)
    fdbody=body
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
    uu=np.copy(d[4:8,:,:,:])  #avoid modifying original; again, note uu[i] are 3-velocities (as read from the fieldline file)
    #uu=d[4:8,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldline file)
    #multiply by u^t to get 4-velocities: u^i = u^t v^i
    uu[1:4]=uu[1:4] * uu[0]
    B = np.zeros_like(uu)
    #cell-centered magnetic field components
    B[1:4,:,:,:]=d[8:11,:,:,:]
    ii=11
    #if the input file contains additional data
    if d.shape[0]==14 or d.shape[0]==23 or d.shape[0]==26:
        print("Loading gdetB data...")
        #new image format additionally contains gdet*B^i
        gdetB = np.zeros_like(B)
        #face-centered magnetic field components multiplied by gdet
        gdetB[1:4] = d[ii:ii+3,:,:,:]; ii=ii+3
    else:
        print("No data on gdetB, approximating it.")
        gdetB = np.zeros_like(B)
        gdetB[1] = gdet * B[1]
        gdetB[2] = gdet * B[2]
        gdetB[3] = gdet * B[3]
    if d.shape[0]==20 or d.shape[0]==23 or d.shape[0]==26:
        print("Loading flux data...")
        gdetF=np.zeros((4,3,nx,ny,nz))
        gdetF[1,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
        gdetF[2,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
        gdetF[3,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
    else:
        print("No data on gdetF, setting it to None.")
        gdetF = None
    #if includes Bstag in addition to all of the above
    if d.shape[0]==26:
        Bstag = np.zeros_like(B)
        Bstag[1:] =  d[ii:ii+3,:,:,:]; ii=ii+3
    else:
        print("No data on Bstag, setting it to B.")
        Bstag = B
    #     if 'gdet' in globals():
    #         #first set everything approximately (B's are at shifted locations by half-cell)
    #         B = gdetB/gdet  
    #         #then, average the inner cells to proper locations
    #         B[1,0:nx-1,:,:] = 0.5*(gdetB[1,0:nx-1,:,:]+gdetB[1,1:nx,:,:])/gdet[0:nx-1,:,:]
    #         B[2,:,0:ny-1,:] = 0.5*(gdetB[2,:,0:ny-1,:]+gdetB[2,:,1:ny,:])/gdet[:,0:ny-1,:]
    #         B[3,:,:,0:nz-1] = 0.5*(gdetB[3,:,:,0:nz-1]+gdetB[3,:,:,1:nz])/gdet[:,:,0:nz-1]
    #         #note: last cells on the grids (near upper boundaries of each dir are at
    #         #      approximate locations
    #     else:
    #         print( "rfd: warning: since gdet is not defined, I am skipping the computation of cell-centered fields, B" )
    # else:
    if 'r' in globals() and r.shape[2] != nz:
        #dynamically change the 3rd dimension size
        rnew = np.zeros((nx,ny,nz),dtype=r.dtype)
        hnew = np.zeros((nx,ny,nz),dtype=h.dtype)
        phnew = np.zeros((nx,ny,nz),dtype=ph.dtype)
        rnew += r[:,:,0:1]
        hnew += h[:,:,0:1]
        tinew = np.zeros((nx,ny,nz),dtype=ti.dtype)
        tjnew = np.zeros((nx,ny,nz),dtype=tj.dtype)
        tknew = np.zeros((nx,ny,nz),dtype=tk.dtype)
        tinew += ti[:,:,0:1]
        tjnew += tj[:,:,0:1]
        tknew += np.arange(nz)[None,None,:]
        #compute size of phi wedge assuming dxdxp[3][3] is up to date
        phiwedge = dxdxp[3][3][0,0,0]*_dx3*nz
        a_phi = phiwedge/(2.*nz)+np.linspace(0,phiwedge,num=nz,endpoint=False)
        phnew += a_phi[None,None,:]
        del r
        del h
        del ph
        r = rnew
        h = hnew
        ph = phnew
        del ti
        del tj
        del tk
        ti = tinew
        tj = tjnew
        tk = tknew
        gc.collect()
    #save file for Josh
    savenewgrid = kwargs.pop("savenewgrid",0)
    if savenewgrid:
        newRin = Rin
        newRout = 1000
        newR0 = 0
        newd = reinterpfld(d,newRin=newRin,newRout=newRout)
        print( "Saving new grid...", )
        #write out a dump with reinterpolated grid spin:
        gout = open( "dumps/" + fieldlinefilename + "newgrid", "wb" )
        header[4] = "%g" % (1.*np.log(newRin))
        header[5] = "%g" % (0.) #_startx2
        header[6] = "%g" % (0.) #_startx3
        header[7] = "%g" % (1.*np.log(newRout/newRin)/nx)
        header[8] = "%g" % (1./ny)
        header[9] = "%g" % (2*np.pi/nz)
        #Spherical polar radius of the innermost radial cell
        header[13] = "%g" % newR0
        header[14] = "%g" % newRin
        header[15] = "%g" % newRout
        for headerel in header:
            s = "%s " % headerel
            gout.write( s )
        gout.write( "\n" )
        gout.flush()
        os.fsync(gout.fileno())
        #reshape the rdump content
        gd1 = newd.reshape(-1,order='F') #view().reshape((nz,ny,nx,-1),order='C')
        gd1.tofile(gout)
        gout.close()
        print( " done!" )
    ##
    ##
    ## RADIATION
    ##
    ##
    global GGG,CCCTRUE,MSUNCM,MPERSUN,LBAR,TBAR,VBAR,RHOBAR,MBAR,ENBAR,UBAR,TEMPBAR,ARAD_CODE_DEF,XFACT,ZATOM,AATOM,MUE,MUI,OPACITYBAR,MASSCM,KORAL2HARMRHO1
    global KAPPAUSER,KAPPAESUSER
    global Erf,uradu,Tgas
    #
    global gotrad
    gotrad=0
    if numcolumns==16:
        print("Reading radiation primitives...")
        gotrad=1
        Erf=np.zeros((1,nx,ny,nz),dtype='float32',order='F')
        uradu=np.zeros((4,nx,ny,nz),dtype='float32',order='F')
        Erf=d[11,:,:,:] # radiation frame radiation energy density
        uradu=d[12:16,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldl
        #multiply by u^t to get 4-velocities: u^i = u^t v^i
        uradu[1:4]=uradu[1:4] * uradu[0]
        #
        maxErf=np.max(Erf)
        minErf=np.min(Erf)
        print("maxErf=%g minErf=%g" % (maxErf,minErf)) ; sys.stdout.flush()
        #
        rddims()
        #
        # now compute auxillary opacity related quantities since only otherwise in raddump
        KAPPA=1.0
        KAPPAES=1.0
        # KORALTODO: Put a lower limit on T~1E4K so not overly wrongly opaque in spots whe
        T1E4K=(1.0E4/TEMPBAR)
        # ideal gas assumed for Tgas
        # code pg
        pg=(gam-1.0)*ug
        # code Tgas
        Tgas=pg/rho
        KAPPAUSER=(rho*KAPPA*KAPPA_FF_CODE(rho,Tgas+T1E4K))
        KAPPAESUSER=(rho*KAPPAES*KAPPA_ES_CODE(rho,Tgas))
        #

def grid3d(dumpname,use2d=False,doface=False): #read grid dump file: header and body
    #The internal cell indices along the three axes: (ti, tj, tk)
    #The internal uniform coordinates, (x1, x2, x3), are mapped into the physical
    #non-uniform coordinates, (r, h, ph), which correspond to radius (r), polar angle (theta), and toroidal angle (phi).
    #There are more variables, e.g., dxdxp, which is the Jacobian of (x1,x2,x3)->(r,h,ph) transformation, that I can
    #go over, if needed.
    global nx,ny,nz,_startx1,_startx2,_startx3,_dx1,_dx2,_dx3,gam,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet,OmegaNS,AlphaNS,defcoord,phiwedgesize
    global tif,tjf,tkf,rf,hf,phf,rhor
    usinggdump2d = False
    if dumpname.endswith(".bin"):
        dumpnamenoext = os.path.splitext(dumpname)[0]
        dumpname2d = dumpnamenoext + "2d.bin"
        if use2d and os.path.isfile("dumps/"+dumpname2d):
            #switch to using 2d gdump if exists
            dumpname = dumpname2d
            usinggdump2d = True
    sys.stdout.write( "Reading grid from " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    #First line of grid dump file is a text line that contains general grid information:
    headerline = gin.readline()
    header = headerline.split()
    #dimensions of the grid
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    if nz == 1:
        usinggdump2d = False
    #grid internal coordinates starting point
    _startx1=myfloat(float(header[4]))
    _startx2=myfloat(float(header[5]))
    _startx3=myfloat(float(header[6]))
    #cell size in internal coordintes
    _dx1=myfloat(float(header[7]))
    _dx2=myfloat(float(header[8]))
    _dx3=myfloat(float(header[9]))
    #other information: 
    #polytropic index
    gam=myfloat(float(header[11]))
    #black hole spin
    a=myfloat(float(header[12]))
    rhor = 1+(1-a**2)**0.5
    R0=myfloat(float(header[13]))
    #Spherical polar radius of the innermost radial cell
    Rin=myfloat(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloat(float(header[15]))
    defcoord = myfloat(float(header[18]))
    if os.path.isfile("coordparms.dat"):
        coordparams = np.loadtxt( "coordparms.dat", 
                      dtype=np.float64, 
                      skiprows=0, 
                      unpack = False )
        if defcoord == 3010: #SNSCOORDS
            if len(coordparams)>=14:
                OmegaNS = coordparams[13]
                AlphaNS = coordparams[14]
            else:
                OmegaNS = 0
                AlphaNS = 0
        elif defcoord == 3000: #SJETCOORDS
            if len(coordparams)>=27:
                OmegaNS = coordparams[26]
                AlphaNS = coordparams[27]
            else:
                OmegaNS = 0
                AlphaNS = 0
    else:
        OmegaNS = 0
        AlphaNS = 0
    #read grid dump per-cell data
    #
    if use2d:
        lnz = 1
    else:
        lnz = nz
    ncols = 126
    if dumpname.endswith(".bin"):
        body = np.fromfile(gin,dtype=np.float64,count=ncols*nx*ny*lnz) 
        gd = body.view().reshape((-1,nx,ny,lnz),order='F')
        gin.close()
        if use2d and not usinggdump2d and not dumpname.endswith("2d.bin"):
            #2d cache file does not exist, create it for future speedup
            sys.stdout.write( 
                "\n Saving a 2d slice of %s as %s for future caching..." 
                % (dumpname, dumpname2d) )
            gout = open( "dumps/" + dumpname2d, "wb" )
            gout.write( headerline )
            #gout.write( "\n" )
            gout.flush()
            os.fsync(gout.fileno())
            #reshape the gdump content
            #pdb.set_trace()
            gd1 = body.view().reshape((lnz,ny,nx,-1),order='C')
            gd1.tofile(gout)
            gout.close()
            sys.stdout.write( "  done!" )
    else:
        gin.close()
        gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True ).view().reshape((126,nx,ny,lnz), order='F')
    gd=myfloat(gd)
    gc.collect()
    ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:,:].view()
    #get the right order of indices by reversing the order of indices i,j(,k)
    conn=gd[9:73].view().reshape((4,4,4,nx,ny,lnz), order='F').transpose(2,1,0,3,4,5)
    #contravariant metric components, g^{\mu\nu}
    gn3 = gd[73:89].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
    #covariant metric components, g_{\mu\nu}
    gv3 = gd[89:105].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
    #metric determinant
    gdet = gd[105]
    ck = gd[106:110].view().reshape((4,nx,ny,lnz), order='F')
    #grid mapping Jacobian
    dxdxp = gd[110:126].view().reshape((4,4,nx,ny,lnz), order='F').transpose(1,0,2,3,4)
    phiwedgesize = nz*_dx3*dxdxp[3,3,0,0,0]
    if doface:
        #CELL VERTICES:
        #RADIAL:
        #add an extra dimension to rf container since one more faces than centers
        rf = np.zeros((r.shape[0]+1,r.shape[1]+1,r.shape[2]+1))
        #operate on log(r): average becomes geometric mean, etc
        rf[1:nx,0:ny,0:lnz] = (r[1:nx]*r[0:nx-1])**0.5 #- 0.125*(dxdxp[1,1,1:nx]/r[1:nx]-dxdxp[1,1,0:nx-1]/r[0:nx-1])*_dx1
        #extend in theta
        rf[1:nx,ny,0:lnz] = rf[1:nx,ny-1,0:lnz]
        #extend in phi
        rf[1:nx,:,lnz]   = rf[1:nx,:,lnz-1]
        #extend in r
        rf[0] = 0*rf[0] + Rin
        rf[nx] = 0*rf[nx] + Rout
        #ANGULAR:
        hf = np.zeros((h.shape[0]+1,h.shape[1]+1,h.shape[2]+1))
        hf[0:nx,1:ny,0:lnz] = 0.5*(h[:,1:ny]+h[:,0:ny-1]) #- 0.125*(dxdxp[2,2,:,1:ny]-dxdxp[2,2,:,0:ny-1])*_dx2
        hf[1:nx-1,1:ny,0:lnz] = 0.5*(hf[0:nx-2,1:ny,0:lnz]+hf[1:nx-1,1:ny,0:lnz])
        #populate ghost cells in r
        hf[nx,1:ny,0:lnz] = hf[nx-1,1:ny,0:lnz]
        #populate ghost cells in phi
        hf[:,1:ny,lnz] = hf[:,1:ny,lnz-1]
        #populate ghost cells in theta (note: no need for this since already initialized everything to zero)
        hf[:,0] = 0*hf[:,0] + 0
        hf[:,ny] = 0*hf[:,ny] + np.pi
        #TOROIDAL:
        phf = np.zeros((ph.shape[0]+1,ph.shape[1]+1,ph.shape[2]+1))
        phf[0:nx,0:ny,0:lnz] = ph[0:nx,0:ny,0:lnz] - dxdxp[3,3,0,0,0]*0.5*_dx3
        #extend in phi
        phf[0:nx,0:ny,lnz]   = ph[0:nx,0:ny,lnz-1] + dxdxp[3,3,0,0,0]*0.5*_dx3
        #extend in r
        phf[nx,0:ny,:]   =   phf[nx-1,0:ny,:]
        #extend in theta
        phf[:,ny,:]   =   phf[:,ny-1,:]
        #indices
        #tif=np.zeros(ti.shape[0]+1,ti.shape[1]+1,ti.shape[2]+1)
        #tjf=np.zeros(tj.shape[0]+1,tj.shape[1]+1,tj.shape[2]+1)
        #tkf=np.zeros(tk.shape[0]+1,tk.shape[1]+1,tk.shape[2]+1)
        tif=np.arange(0,(nx+1)*(ny+1)*(lnz+1)).reshape((nx+1,ny+1,lnz+1),order='F')
        tjf=np.arange(0,(nx+1)*(ny+1)*(lnz+1)).reshape((nx+1,ny+1,lnz+1),order='F')
        tkf=np.arange(0,(nx+1)*(ny+1)*(lnz+1)).reshape((nx+1,ny+1,lnz+1),order='F')
        tif %= (nx+1)
        tjf /= (nx+1)
        tjf %= (ny+1)
        tkf /= (ny+1)*(lnz+1)
    gc.collect() #try to release unneeded memory
    print( "  done!" )


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
    zz = np.linspace(-zmax, zmax, ncellz,endpoint=1)
    x3d = np.linspace(-xmax, xmax, ncellx,endpoint=1)[:,None,None]
    y3d = np.linspace(-ymax, ymax, ncelly,endpoint=1)[None,:,None]
    z3d = zz[None,None,:]
    rmax = (xmax**2+ymax**2+zmax**2)**0.5
    Rmax = (xmax**2+ymax**2)**0.5
    #make the arrays 3d
    x3d = x3d + 0*x3d+0*y3d+0*z3d
    y3d = y3d + 0*x3d+0*y3d+0*z3d
    z3d = z3d + 0*x3d+0*y3d+0*z3d
    #construct meridional 2d grid:
    R2d = np.linspace(0, 2*Rmax, ncellx*2); dR2d = R2d[1]-R2d[0];
    z2d = zz; dz2d = z2d[1]-z2d[0];
    #compute i,j-indices on meridional grid:
    ph = 0
    R = r*sin(h)
    R[:,0,:]*=0
    R[:,ny-1,:]*=0
    x = (R)[...,0][r[...,0]<1.1*rmax]
    z = (r*cos(h))[...,0][r[...,0]<1.1*rmax]
    i = (ti)[...,0][r[...,0]<1.1*rmax]
    j = (tj)[...,0][r[...,0]<1.1*rmax]
    #get i,j-indices on the 2d meridional grid grid
    i2d = griddata((x, z), i, (R2d[:,None], z2d[None,:]), method="linear",fill_value=nx-1)
    j2d = griddata((x, z), j, (R2d[:,None], z2d[None,:]), method="linear",fill_value=0)
    #
    R3d = (x3d**2+y3d**2)**0.5
    i3d = bilin(i2d,(R3d/dR2d),(z3d+zmax)/dz2d)
    j3d = bilin(j2d,(R3d/dR2d),(z3d+zmax)/dz2d)
    #pdb.set_trace()
    dphi = dxdxp[3,3][0,0,0]*_dx3
    k3d = (np.arctan2(y3d, x3d)/dphi-0.5)
    k3d[k3d<0] = k3d[k3d<0] + nz
    return i3d, j3d, k3d, x3d, y3d, z3d   

def bilin(a,i,j,order=3):
    return ndimage.map_coordinates(a,np.array([i,j]),order=order,mode="nearest")


def trilin(a,i,j,k,order=3,fast=0):
    #a is the array to be interpolated
    #i,j,k are indices, which are 3d arrays; can be non-integers (floats)
    #returns interpolated values of a[i,j,k]
    nbnd = 3
    a_with_bnd_cells = np.concatenate((a[...,-nbnd:],a,a[...,:nbnd]),axis=-1)
    if not fast:
        return ndimage.map_coordinates(a_with_bnd_cells,np.array([i,j,k+nbnd]),order=order,mode="nearest")
    else:
        return a_with_bnd_cells[np.int32(i),np.int32(j),np.int32(k+nbnd)]

def compute_jet_head_velocity(headfile="headinfo.npz",endn=-1,dn=1):
    if not os.path.isfile( headfile ):
        print( "Precomputing head motion data..." )
        thead = []
        vhead = []
        rhead = []
        xmaxes = []
        ymaxes = []
        flist1 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9].bin") ) )
        flist2 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9][0-9].bin") ) )
        flist1.sort()
        flist2.sort()
        flist = np.concatenate((flist1,flist2))
        for fldname in flist:
            fldindex = np.int(fldname.split(".")[0].split("e")[-1])
            # if fldindex < startn:
            #     continue
            if endn>=0 and fldindex >= endn:
                break
            if fldindex < dn*100:
                if fldindex % (dn*10):
                    continue
            elif fldindex % (dn*100):
                continue
            print( "Reading " + fldname + " ..." )
            sys.stdout.flush()
            rfd("../"+fldname)
            sys.stdout.flush()
            ent1d = (ug/(rho+1e-15)**gam)[:,2:10].mean(1)
            which = (ent1d>0.1*np.max(ent1d))[:,0]
            vheadmin = 0.1
            if t < 10*Rin:
                vhead = vheadmin
                rhead = vhead * t
            elif which.sum():
                rhead = r[which,0,0][-1]
                vhead = (rhead-Rin) / (t+1e-5)
            else:
                rhead = Rout
                vhead = 1
            if vhead > 1 or np.isnan(vhead):
                vhead = 1
            if vhead < vheadmin:
                vhead = vheadmin
            ymax = vhead*t*1.5
            if ymax < 8:
                ymax= 8
            if ymax > 5000 or np.isnan(ymax):
                ymax = 5000
            xmax = ymax**0.5
            if xmax < 8:
                xmax = 8
            vhead.append(vhead)
            thead.append(t)
            rhead.append(rhead)
            ymaxes.append(ymax)
            xmaxes.append(xmax)
            print( "t=%g, rhead=%g, vhead=%g" % (t, rhead, vhead) )
        print( "Saving to file..." )
        np.savez( headfile, vhead = vhead, thead = thead, xmaxes = xmaxes, ymaxes = ymaxes)
    else:
        print( "Loading precomputed head motion data..." )
        npzfile = np.load(headfile)
        vhead=npzfile['vhead']
        thead=npzfile['thead']
    myvheadfunc=interp1d(thead,vhead,kind="cubic",bounds_error=False,fill_value = vhead[-1])
    return myvheadfunc

def ymf(t): 
   return( compute_jet_head_velocity()(t)*t*1.5 )

def get_jet_zmax(t):
    # ymax_filter = interp1d([0,          70,      73,     160,     163,     500, 503 ],
    #                        [ymf(70),ymf(70),ymf(160),ymf(160),ymf(500),ymf(500)],order=1,bounds_error=False)
    # t_convert = interp1d([0,  400, 425, 600, 625,  800,  825, 1000, 1025, 1200, 1205, 2400, 2405 ],
    #                      [400,400, 600, 600, 800,  800, 1000, 1000, 1200, 1200, 2400, 2400, 4800 ],
    #                      kind="linear",bounds_error=False,fill_value = 4800)
    if 0:
        t_convert = interp1d([0,  400, 425, 600, 625,  800,  825, 1000, 1025, 1200, 1205, 2400, 2405 ],
                             [400,400, 600, 600, 800,  800, 1000, 1000, 1200, 1200, 2400, 2400, 4800 ],
                             kind="linear",bounds_error=False,fill_value = 4800)
        ymax = ymf(t_convert(t))
    elif 1:
        t_convert = interp1d([0,  400, 450, 5000 ],
                             [400,400, 800,  800 ],
                             kind="linear",bounds_error=False,fill_value = 800)
        ymax = ymf(t_convert(t))
    else:
        ymax = ymf(800)
    # if ymax < 70:
    #     ymax= 70
    # if ymax < 170:
    #     ymax = 170
    # if ymax < 500:
    #     ymax = 500
    # if ymax < 1000:
    #     ymax = 1000
    # if ymax > 5000 or np.isnan(ymax):
    #     ymax = 5000
    xmax = ymax**0.5
    # if xmax < 8:
    #     xmax = 8
    return(xmax,ymax)

def mk_vis_grb_movie(n1=0,n2=-1):
    flist1 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9].bin") ) )
    flist2 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9][0-9].bin") ) )
    flist1.sort()
    flist2.sort()
    flist = np.concatenate((flist1,flist2))
    if "vol_jet" not in globals():
        updateframe = 0
    else:
        updateframe = 1
    #mlab.options.offscreen = True
    for fldname in flist:
        fldindex = np.int(fldname.split(".")[0].split("e")[-1])
        if fldindex < n1: continue
        if n2 >= 0 and fldindex > n2: continue
        vis_grb(no=fldindex,updateframe=updateframe,dosavefig=1)
        updateframe = 1
    #mlab.options.offscreen = False
    
def vis_grb(doreload=1,no=555,xmax=50,ymax=50,zmax=None,ncellx=101,ncelly=101,ncellz=101,dosavefig=0,order=3,dofieldlines=0,updateframe=0):
    global mlab_lrho_jet, vol_jet, lrhoi_jet, i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet
    if doreload:
        if 'gv3' not in globals():
            grid3d("gdump.bin",use2d=1)
        rfd("fieldline%04d.bin"%no)
        cvel()
    if zmax is None:
        autozoom = 1
        xmax, zmax = get_jet_zmax(t)
        ymax = 3*xmax
        if ymax < 50:
            ymax = 50
        ncelly = np.int32(0.5+1.*ymax/xmax*ncellx)
        #ymax = 10*xmax
        # ncelly = 3*ncellx
        #to ensure cubical cells
        ncellz = np.int32(0.5+1.*zmax/xmax*ncellx)
        #if ncellz > 1000: ncellz = 1000
        print( "xmax = %g, ymax = %g, zmax = %g; ncellx = %d, ncelly = %d, ncellz = %d" 
               % (xmax, ymax, zmax, ncellx, ncelly, ncellz) )
    else:
        autozoom = 0
    if updateframe:
        scene = mlab.figure(1)
    else:
        scene = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(960,540))#size=(210*2, 297*2))
        #scene.scene.disable_render = True
    print( "Running interp3d..." ); sys.stdout.flush()
    i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
        interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    # i3d_jet1,j3d_jet1,k3d_jet1,xi_jet1,yi_jet1,zi_jet1 =\
    #     interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    print( "Done with inter3d for jet..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for jet..." ); sys.stdout.flush()
    lrhoi_jet = np.float32(trilin(lrho,i3d_jet,j3d_jet,k3d_jet,order=order))
    # lrhoi_jet[xi_jet**2+yi_jet**2+zi_jet**2<1.1]=lrhoi_jet[xi_jet**2+yi_jet**2+zi_jet**2<1.1]*0
    # lrhoi_jet[xi_jet**2+yi_jet**2+zi_jet**2<0.5]=lrhoi_jet[xi_jet**2+yi_jet**2+zi_jet**2<0.5]*0+8
    lrhoi_jet[ncellx/2,ncelly/2,ncellz/2]=8
    lrhoi_jet[ncellx/2,ncelly/2,ncellz/2+1]=0
    #lrhoi_jet = lrho[np.int32(0.5+i3d_jet),np.int32(0.5+j3d_jet),np.int32(0.5+k3d_jet)]
    if 1:
        vmin = 0
        vmax = 10
    else:
        vmin = lrhoi_jet.min()
        vmax = lrhoi_jet.max()
    print( "Done with trilinear interpolation for jet..." ); sys.stdout.flush()
    if updateframe:
        mlab_lrho_jet.mlab_source.reset(scalars = lrhoi_jet, x = xi_jet, y = yi_jet, z = zi_jet)
        vol_jet.mlab_source.set(vmin=vmin,vmax=vmax)
    else:
        mlab_lrho_jet = mlab.pipeline.scalar_field(xi_jet,yi_jet,zi_jet,lrhoi_jet)
        # bsqorhoi_jet = np.float32((bsq/rho)[i3d_jet,j3d_jet,k3d_jet])
        # mlab_bsqorho = mlab.pipeline.scalar_field(xi_jet,yi_jet,zi_jet,bsqorhoi_jet)
        # pdb.set_trace()
        mlab.clf()
        # Change the otf (opacity transfer function) of disk and jet:
        from tvtk.util.ctf import PiecewiseFunction
        print( "Running volume rendering for jet..." ); sys.stdout.flush()
        vol_jet = mlab.pipeline.volume(mlab_lrho_jet,vmin=vmin,vmax=vmax) #,vmin=-6,vmax=1)
        print( "Done with volume rendering of jet..." ); sys.stdout.flush()
        if 1:
            otf_jet = PiecewiseFunction()
            otf_jet.add_point(vmin, 0.)
            otf_jet.add_point(2.391, 0.265)
            otf_jet.add_point(3.525, 0.878)
            otf_jet.add_point(vmax, 1.)
            vol_jet._otf = otf_jet
            vol_jet._volume_property.set_scalar_opacity(otf_jet)
        vol_jet.volume_mapper.blend_mode = 'minimum_intensity'
        #
        # Streamlines
        #
        if dofieldlines:
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
            print( "Running trilinear interpolation for Bx..." ); sys.stdout.flush()
            Bxi = np.float32(trilin(Bx,i3d_jet,j3d_jet,k3d_jet,order=order))
            print( "Done with trilinear interpolation for Bx..." ); sys.stdout.flush()
            print( "Running trilinear interpolation for By..." ); sys.stdout.flush()
            Byi = np.float32(trilin(By,i3d_jet,j3d_jet,k3d_jet,order=order))
            print( "Done with trilinear interpolation for By..." ); sys.stdout.flush()
            print( "Running trilinear interpolation for Bz..." ); sys.stdout.flush()
            Bzi = np.float32(trilin(Bz,i3d_jet,j3d_jet,k3d_jet,order=order))
            print( "Done with trilinear interpolation for Bz..." ); sys.stdout.flush()
            streamlines = []
            xpos = [3.0137298, -0.39477735484, 2.5, 2.5]
            ypos = [3.20705557,  -0.477990151219, 0,  0.]
            zpos = [4.40640898,  -29.3698147762, 90.5538316429,  -90.5538316429]
            intdir = ['backward', 'backward', 'forward', 'forward']
            for sn in xrange(4):
                print( "Running rendering of streamline #%d..." % (sn+1) ); sys.stdout.flush()
                streamline = mlab.flow(xi_jet, yi_jet, zi_jet, Bxi, Byi, Bzi, seed_scale=0.01,
                                 seed_resolution=5,
                                 integration_direction=intdir[sn],
                                 seedtype='point')
                streamlines.append(streamline)
                streamline.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
                streamline.streamline_type = 'tube'
                streamline.tube_filter.radius = 0.2
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
                print( "Done with streamline #%d..." % (sn+1)); sys.stdout.flush()
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
            rbh = 1.0
            thbh = np.linspace(0,np.pi,128,endpoint=1)[:,None]
            phbh = np.linspace(0,2*np.pi,128,endpoint=1)[None,:]
            thbh = thbh + 0*phbh + 0*thbh
            phbh = phbh + 0*phbh + 0*thbh
            xbh = rbh*sin(thbh)*cos(phbh)
            ybh = rbh*sin(thbh)*sin(phbh)
            zbh = rbh*cos(thbh)
            mlab.mesh(xbh, ybh, zbh, scalars=1+0*thbh, colormap='hsv',vmin=0, vmax = 1)
    #end if updateframe
    #move camera:
    xyzcam = np.array([-1, 0, 0])
    #normalize
    xyzcam /= np.sum(xyzcam**2)**0.5
    #set the distance to desired one
    xyzcam *= zmax*1.2
    #scene.scene.disable_render = False
    #this is needed to draw the scene somehow...
    mlab.view(focalpoint=[0,0,0],distance=zmax)
    scene.scene.camera.position = xyzcam
    scene.scene.camera.focal_point = [0, 0, 0]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.060975207259923878, 0.88384564068235227, 0.46378756726157683]
    scene.scene.camera.clipping_range = [0.5*1.2*zmax, 2*1.2*zmax]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    #mlab.view(focalpoint=[0,0,0],distance=500)
    #mlab.show()
    print( "Done rendering!" ); sys.stdout.flush()
    if dosavefig:
        print( "Saving snapshot..." ); sys.stdout.flush()
        mlab.savefig("disk_jet_nozoomchange_%04d.png" % no, figure=scene, magnification=2.0)
        print( "Done!" ); sys.stdout.flush()


def visualize_grid(no = 0, doreload=1):
    if doreload:
        grid3d("gdump.bin",use2d=1)
        rfd("fieldline%04d.bin"%no)
        cvel()
    scene = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(210*2, 297*2))
    mlab.clf()
    sg = create_structured_grid(s=lrho,sname="density",v=None,vname=None)
    # Now visualize the data.
    d = mlab.pipeline.add_dataset(sg)
    gx = mlab.pipeline.grid_plane(d)
    # gy = mlab.pipeline.grid_plane(d)
    # gy.grid_plane.axis = 'y'
    gz = mlab.pipeline.grid_plane(d)
    gz.grid_plane.axis = 'z'
    iso = mlab.pipeline.iso_surface(d)

        

#@mayavi2.standalone    
def visualize_data(doreload=1,no=5468,xmax=200,ymax=200,zmax=1000,ncellx=200,ncelly=200,ncellz=1000,xmax_disk=200,ymax_disk=200,zmax_disk=200,ncellx_disk=200,ncelly_disk=200,ncellz_disk=200,dosavefig=0):
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
    print( "Running interp3d for disk..." ); sys.stdout.flush()
    i3d_disk,j3d_disk,k3d_disk,xi_disk,yi_disk,zi_disk =\
        interp3d(xmax=xmax_disk,ymax=ymax_disk,zmax=zmax_disk,ncellx=ncellx_disk,ncelly=ncelly_disk,ncellz=ncellz_disk)
    print( "Done with interp3d for disk..." ); sys.stdout.flush()
    print( "Running interp3d for jet..." ); sys.stdout.flush()
    i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
        interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    print( "Done with inter3d for jet..." ); sys.stdout.flush()
    #lrhoxpos = lrho*(r*sin(h)*cos(ph)>0)
    print( "Running trilinear interpolation for disk..." ); sys.stdout.flush()
    lrhoi_disk = np.float32(trilin(lrho,i3d_disk,j3d_disk,k3d_disk))
    print( "Done with trilinear interpolation for disk..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for jet..." ); sys.stdout.flush()
    lrhoi_jet = np.float32(trilin(lrho,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for jet..." ); sys.stdout.flush()
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
    print( "Running trilinear interpolation for Bx..." ); sys.stdout.flush()
    Bxi = np.float32(trilin(Bx,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for Bx..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for By..." ); sys.stdout.flush()
    Byi = np.float32(trilin(By,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for By..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for Bz..." ); sys.stdout.flush()
    Bzi = np.float32(trilin(Bz,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for Bz..." ); sys.stdout.flush()
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
        print( "Running volume rendering for disk..." ); sys.stdout.flush()
        vol_disk = mlab.pipeline.volume(mlab_lrho_disk) #,vmin=-6,vmax=1)
        print( "Done with volume rendering of disk..." ); sys.stdout.flush()
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
        vol_disk.update_ctf = 1
        vol_disk.update_ctf = 0
        print( "Running volume rendering for jet..." ); sys.stdout.flush()
        vol_jet = mlab.pipeline.volume(mlab_lrho_jet) #,vmin=-6,vmax=1)
        print( "Done with volume rendering of jet..." ); sys.stdout.flush()
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
        xpos = [0.8, -0.5, 1, -0.7]
        ypos = [0.0,  0.0, 0,  0.3]
        zpos = [0,  0, 0,  0]
        intdir = ['backward', 'backward', 'forward', 'forward']
        for sn in xrange(4):
            print( "Running rendering of streamline #%d..." % (sn+1) ); sys.stdout.flush()
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
            print( "Done with streamline #%d..." % (sn+1)); sys.stdout.flush()
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
    print( "Done rendering!" ); sys.stdout.flush()
    if dosavefig:
        print( "Saving snapshot..." ); sys.stdout.flush()
        mlab.savefig("disk_jet_with_field_lines.png", figure=scene, magnification=6.0)
        print( "Done!" ); sys.stdout.flush()


def visualize_rad(doreload=1,no=5468,xmax=100,ymax=100,zmax=500,ncellx=100,ncelly=100,ncellz=500,xmax_disk=100,ymax_disk=100,zmax_disk=100,ncellx_disk=100,ncelly_disk=100,ncellz_disk=100,dosavefig=0):
    if doreload:
        grid3d("gdump.bin",use2d=1)
        #rfd("fieldline9000.bin")
        rfd("fieldline%04d.bin"%no)
        cvel()
    dictau = compute_taurad()
    scene = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(210*2, 297*2))
    scene.scene.disable_render = True
    mlab.clf()
    #pdb.set_trace()
    #choose 1.8 > sqrt(3):
    # grid the data.
    # var = lrho[...,0][r[...,0]<1.8*rmax]
    # vi = griddata((x, z), var, (xi, zi), method="linear")
    # pdb.set_trace()
    print( "Running interp3d for disk..." ); sys.stdout.flush()
    i3d_disk,j3d_disk,k3d_disk,xi_disk,yi_disk,zi_disk =\
        interp3d(xmax=xmax_disk,ymax=ymax_disk,zmax=zmax_disk,ncellx=ncellx_disk,ncelly=ncelly_disk,ncellz=ncellz_disk)
    print( "Done with interp3d for disk..." ); sys.stdout.flush()
    print( "Running interp3d for jet..." ); sys.stdout.flush()
    i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
        interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
    print( "Done with inter3d for jet..." ); sys.stdout.flush()
    #lrhoxpos = lrho*(r*sin(h)*cos(ph)>0)
    print( "Running trilinear interpolation for disk..." ); sys.stdout.flush()
    lrhoi_disk = np.float32(trilin(lrho,i3d_disk,j3d_disk,k3d_disk))
    print( "Done with trilinear interpolation for disk..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for jet..." ); sys.stdout.flush()
    #only optically-thin radiation density
    #maxrad = urad[(taudic["tau2"]<1)*(urad>0)].max()
    #minrad = urad[(taudic["tau2"]<1)*(urad>0)*(r<100)].min()
    maxrad = 1e-5
    minrad = 1e-10
    vminrad = np.log10(minrad)
    vmaxrad = np.log10(maxrad)
    mindisk = 1e-15
    maxdisk = 1e-2
    vmindisk = np.log10(mindisk)
    vmaxdisk = np.log10(maxdisk)
    transition_f = np.maximum(np.minimum(1,3*(1-dictau["tau2"])),0)
    qty = np.log10( np.minimum(np.maximum(urad*transition_f,minrad),maxrad) )
    lrhoi_jet = np.float32(trilin(qty,i3d_jet,j3d_jet,k3d_jet))
    lrhoi_jet = ndimage.filters.gaussian_filter(lrhoi_jet,2,order=3,mode="nearest")
    lrhoi_jet[ncellx/2,ncelly/2,ncellz/2] = np.log10(minrad)
    lrhoi_jet[ncellx/2,ncelly/2,ncellz/2+1] = np.log10(maxrad)
    print( "Done with trilinear interpolation for jet..." ); sys.stdout.flush()
    lrhoi_disk[ncellx_disk/2,ncelly_disk/2,ncellz_disk/2] = vmindisk
    lrhoi_disk[ncellx_disk/2,ncelly_disk/2,ncellz_disk/2+1] = vmaxdisk
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
    print( "Running trilinear interpolation for Bx..." ); sys.stdout.flush()
    Bxi = np.float32(trilin(Bx,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for Bx..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for By..." ); sys.stdout.flush()
    Byi = np.float32(trilin(By,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for By..." ); sys.stdout.flush()
    print( "Running trilinear interpolation for Bz..." ); sys.stdout.flush()
    Bzi = np.float32(trilin(Bz,i3d_jet,j3d_jet,k3d_jet))
    print( "Done with trilinear interpolation for Bz..." ); sys.stdout.flush()
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
        from tvtk.util.ctf import PiecewiseFunction
        print( "Running volume rendering for jet..." ); sys.stdout.flush()
        vol_jet = mlab.pipeline.volume(mlab_lrho_jet) #,vmin=-6,vmax=1)
        print( "Done with volume rendering of jet..." ); sys.stdout.flush()
        vol_jet.volume_mapper.blend_mode = 'composite'
        #this is not used, have to do something else to change it
        #vol_jet.module_manager.scalar_lut_manager.lut_mode = 'Reds'
        otf_jet = PiecewiseFunction()
        if 0:
            otf_jet.add_point(-6., 0.429)
            otf_jet.add_point(-4.547, 0.429)
            otf_jet.add_point(-2.92, 0)
            otf_jet.add_point(1, 0.)
        elif 0:
            #less diffuse jet, so it does not touch box boundaries
            otf_jet.add_point(-6.0, 0.408)
            otf_jet.add_point(-4.8, 0.408)
            otf_jet.add_point(-3.8, 0.)
            otf_jet.add_point(1., 0.)
        else:
            vmin = vminrad #lrhoi_jet.min()
            vmax = vmaxrad
            print("Jet urad vmin/vmax: %g, %g" % (vmin, vmax) )
            otf_jet.add_point(vmin, 0)
            otf_jet.add_point(vmin+0.05*(vmax-vmin), 0)
            otf_jet.add_point(vmin+0.5*(vmax-vmin), 0.1)
            otf_jet.add_point(vmax, 1)
        vol_jet._otf = otf_jet
        vol_jet._volume_property.set_scalar_opacity(otf_jet)
        print( "Running volume rendering for disk..." ); sys.stdout.flush()
        vol_disk = mlab.pipeline.volume(mlab_lrho_disk) #,vmin=-6,vmax=1)
        print( "Done with volume rendering of disk..." ); sys.stdout.flush()
        vol_disk.volume_mapper.blend_mode = 'composite'
        # Change the otf (opacity transfer function) of disk and jet:
        otf_disk = PiecewiseFunction()
        if 0:
            otf_disk.add_point(-6, 0)
            otf_disk.add_point(-1.7, 0)
            otf_disk.add_point(-0.25, 0.8)
            otf_disk.add_point(1, 0.8)
        elif 0:
            #brighter disk
            otf_disk.add_point(-6., 0)
            otf_disk.add_point(-1.733, 0)
            otf_disk.add_point(-1.133, 0.51)
            otf_disk.add_point(-0.533, 0.796)
            otf_disk.add_point(1., 1.)
        else:
            #vmin = lrhoi_disk.min()
            #vmax = lrhoi_disk.max()
            vmin = np.log10(mindisk)
            vmax = np.log10(maxdisk)
            print("Disk vmin/vmax: %g, %g" % (vmin, vmax) )
            otf_disk.add_point(vmin, 0)
            otf_disk.add_point(vmin+0.8*(vmax-vmin), 0.)
            otf_disk.add_point(vmin+0.93*(vmax-vmin), 0.08)
            otf_disk.add_point(vmax, 0.347)
        vol_disk._otf = otf_disk
        vol_disk._volume_property.set_scalar_opacity(otf_disk)
        vol_disk.update_ctf = 1
        vol_disk.update_ctf = 0
    if 1:
        streamlines = []
        OmegaH = a/(2*rhor)
        phase = 0.25*OmegaH
        length = r[bsq/rho>1].max()
        xpos = [np.sin(phase), np.sin(phase)]
        ypos = [np.cos(phase), np.cos(phase)]
        zpos = [1,-1]
        intdir = ['forward', 'backward']
        for sn in xrange(len(xpos)):
            print( "Running rendering of streamline #%d..." % (sn+1) ); sys.stdout.flush()
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
            streamline.stream_tracer.maximum_propagation = 3*length
            streamline.seed.widget.position = np.array([ xpos[sn],  ypos[sn],  zpos[sn]])
            streamline.seed.widget.enabled = False
            streamline.update_streamlines = 1
            print( "Done with streamline #%d..." % (sn+1)); sys.stdout.flush()
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
    scene.scene.camera.position = [-859.35106020552962, 428.88279116475701, -543.94833829177082]
    scene.scene.camera.focal_point = [0.0, 0.0, 0.0]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.20772523499523163, -0.58144536411289605, -0.78662031202976024]
    scene.scene.camera.clipping_range = [368.3714513741902, 2032.1100336754646]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    #mlab.view(focalpoint=[0,0,0],distance=500)
    #mlab.show()
    scene.scene.disable_render = False
    print( "Done rendering!" ); sys.stdout.flush()
    if dosavefig:
        print( "Saving snapshot..." ); sys.stdout.flush()
        mlab.savefig("disk_jet_with_field_lines.png", figure=scene, magnification=6.0)
        print( "Done!" ); sys.stdout.flush()

#vis_grb(dofieldlines=0,dosavefig=1)
#mk_vis_grb_movie(n1=259)

def tiltedomegaf_bad(alpha=90):
    from numpy import sin, cos, tan, sqrt
    cot = lambda x: 1./tan(x)
    csc = lambda x: 1./sin(x)
    sec = lambda x: 1./cos(x)
    
    alpha *= np.pi/180.;
    vu = uu/uu[0]
    v1 = vu[1] - (vu[1]/B[1])*B[1] #zero identically so no need to transform
    v2 = vu[2] - (vu[1]/B[1])*B[2]
    #needs modification for grid collimated toward x-axis
    v2 *= dxdxp[2,2] #to get physical units (no need for r-contribution since v1 = 0)
    v3 = vu[3] - (vu[1]/B[1])*B[3]
    v3 *= dxdxp[3,3] #to get physical units

    omegatilt = v3*cos(alpha) + sin(alpha)*(-(v3*cos(ph)*cot(h)) + 
     -      (v2*csc(h)**2*sin(ph)*sqrt(cos(alpha)**2*cos(ph)**2 + cot(h)**2*sin(alpha)**2 - cos(ph)*cot(h)*sin(2*alpha) + sin(ph)**2))/
     -       ((cos(alpha) - cos(ph)*cot(h)*sin(alpha))**2 + csc(h)**2*sin(alpha)**2*sin(ph)**2));

    return omegatilt

def prime2cart(V):
    global dxdxp
    Vr = dxdxp[1,1]*V[1]+dxdxp[1,2]*V[2]
    Vh = dxdxp[2,1]*V[1]+dxdxp[2,2]*V[2]
    Vp = V[3]*dxdxp[3,3]
    #
    Vrnorm=Vr
    Vhnorm=Vh*np.abs(r)
    Vpnorm=Vp*np.abs(r*np.sin(h))
    #
    Vznorm=Vrnorm*np.cos(h)-Vhnorm*np.sin(h)
    VRnorm=Vrnorm*np.sin(h)+Vhnorm*np.cos(h)
    Vxnorm=VRnorm*np.cos(ph)-Vpnorm*np.sin(ph)
    Vynorm=VRnorm*np.sin(ph)+Vpnorm*np.cos(ph)
    return(np.array([V[0],Vxnorm,Vynorm,Vznorm]))

def getxyz(r,h,ph):
    x = r*np.sin(h)*cos(ph)
    y = r*np.sin(h)*sin(ph)
    z = r*np.cos(h)
    return x,y,z

def tiltedomegaf():
    vu = uu/uu[0]
    vprime = vu - (vu[1]/B[1])*B #zero for vprime[1] identically so no need to transform
    vcart = prime2cart(vprime)
    x,y,z = getxyz(r,h,ph)
    #Rp = Rprime, in the rotated system
    Rp = (y**2+z**2)**0.5 
    ephirot = [0,0,-z/Rp,y/Rp]
    omegatilt = vcart[1]*ephirot[1] + vcart[2]*ephirot[2] + vcart[3]*ephirot[3]
    omegatilt /= Rp
    return omegatilt

def mk_movie(n1=0,n2=-1):
    flist1 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9].bin") ) )
    flist2 = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9][0-9].bin") ) )
    flist1.sort()
    flist2.sort()
    flist = np.concatenate((flist1,flist2))
    for fldname in flist:
        fldindex = np.int(fldname.split(".")[0].split("e")[-1])
        if fldindex < n1: continue
        if n2 >= 0 and fldindex > n2: continue
        visualize_fieldlines(no=fldindex,dosavefig=1,domov=1)


# fn - figure number
# fast = 1 avoids interpolations that make images nicer but take extra time
# no = the fieldline dump number to load
# isaligned = 1 means use the box elongated along x-direction (=0 means elongated along z-dir)
# xmax, ymax, zmax = box sizes in x-, y-, and z-directions
# ncellx, ncelly, ncellz = resolutions in x-, y-, and z-directions
# dosavefig = 1 means to save the figure (=0 by default to speed up)
def visualize_fieldlines(fn=1,fast=0,isaligned=0,doreload=1,no=0,xmax=100,ymax=30,zmax=30,ncellx=200,ncelly=60,ncellz=60,dosavefig=0,vmin=-3.6692+1,vmax=2.27925+1,domov=0):
    global ph,lrhoi_jet,i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet,scene,vol_jet,otf_jet
    if isaligned:
        ncellxold = ncellx; ncellx = ncellz; ncellz = ncellxold
        xmaxold = xmax; xmax = zmax; zmax = xmaxold
    if doreload:
        grid3d("gdump.bin",use2d=1)
        #rfd("fieldline9000.bin")
        rfd("fieldline%04d.bin"%no)
        #only needed of FULLOUTPUT is on (to correct the bug in how code outputs data in phi)
        #if it is on, uncomment the below two lines and set FULLOUTPUT to value in the code
        #FULLOUTPUT = 2
        #ph -= FULLOUTPUT*2*np.pi/(nz-2*FULLOUTPUT)
        cvel()
    bsqorhofid = 0.01*np.max(bsq/rho)
    rhead = np.max(r[bsq/rho>bsqorhofid])
    print "rhead = %g" % rhead
    scene = mlab.figure(fn, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(210*2, 297*2))
    mlab.clf()
    if 1:
        print( "Running interp3d for jet..." ); sys.stdout.flush()
        i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
            interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
        print( "Done with inter3d for jet..." ); sys.stdout.flush()
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
        print( "Running trilinear interpolation for Bx..." ); sys.stdout.flush()
        Bxi = np.float32(trilin(Bx,i3d_jet,j3d_jet,k3d_jet,fast=fast))
        print( "Done with trilinear interpolation for Bx..." ); sys.stdout.flush()
        print( "Running trilinear interpolation for By..." ); sys.stdout.flush()
        Byi = np.float32(trilin(By,i3d_jet,j3d_jet,k3d_jet,fast=fast))
        print( "Done with trilinear interpolation for By..." ); sys.stdout.flush()
        print( "Running trilinear interpolation for Bz..." ); sys.stdout.flush()
        Bzi = np.float32(trilin(Bz,i3d_jet,j3d_jet,k3d_jet,fast=fast))
        print( "Done with trilinear interpolation for Bz..." ); sys.stdout.flush()
        # pdb.set_trace()
    if 1:
        # print( "Running interp3d for jet..." ); sys.stdout.flush()
        # i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
        #   interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
        print( "Done with inter3d for jet..." ); sys.stdout.flush()
        print( "Running trilinear interpolation for jet..." ); sys.stdout.flush()
        lrhoi_jet = np.float32(trilin(lrho,i3d_jet,j3d_jet,k3d_jet,fast=fast))
        print( "Done with trilinear interpolation for jet..." ); sys.stdout.flush()
        mlab_lrho_jet = mlab.pipeline.scalar_field(xi_jet,yi_jet,zi_jet,lrhoi_jet,fast=fast)
        from tvtk.util.ctf import PiecewiseFunction
        print( "Running volume rendering for jet..." ); sys.stdout.flush()
        #vmin = lrhoi_jet.min();
        #vmax = lrhoi_jet.max();
        print vmin, vmax
        vol_jet = mlab.pipeline.volume(mlab_lrho_jet,vmin=vmin,vmax=vmax)
        print( "Done with volume rendering of jet..." ); sys.stdout.flush()
        vol_jet.volume_mapper.blend_mode = 'composite'
        otf_jet = PiecewiseFunction()
        if 0:
            otf_jet.add_point(-6., 0.429)
            otf_jet.add_point(-4.547, 0.429)
            otf_jet.add_point(-2.92, 0.05)
            otf_jet.add_point(-1, 0)
            otf_jet.add_point(1, 0.)
        elif 0:
            #less diffuse jet, so it does not touch box boundaries
            otf_jet.add_point(-2, 1.)
            otf_jet.add_point(1, 0.1)
            otf_jet.add_point(2, 0.)
            otf_jet.add_point(4., 0.)
        elif 0:
            #less diffuse jet, so it does not touch box boundaries
            otf_jet.add_point(lrhoi_jet.min(), .5)
            otf_jet.add_point(lrhoi_jet.max(), 0.)
        elif 0:
            otf_jet.add_point(vmin, 0.5)
            otf_jet.add_point(vmin-(-3.6692)-2.5, 0.5)
            otf_jet.add_point(vmin-(-3.6692)-2, 0.02)
            otf_jet.add_point(-0.5, 0.)
            otf_jet.add_point(vmax, 0.)
        elif 1: #good for lowd
            otf_jet.add_point(vmin, 0.5)
            otf_jet.add_point(-2.5, 0.5)
            otf_jet.add_point(-2, 0.02)
            otf_jet.add_point(-0.5, 0.02)
            otf_jet.add_point(vmax, 0.02)
        else: #good for hid
            otf_jet.add_point(vmin, 0.5)
            otf_jet.add_point(0.5*(vmin+vmax), 0.04)
            otf_jet.add_point(vmax, 0.0)
        vol_jet._otf = otf_jet
        vol_jet._volume_property.set_scalar_opacity(otf_jet)        
    if 1:
        streamlines = []
        if domov:
            xpos = [0.5,                     -0.5]
            ypos = [0.5*np.sin(OmegaNS*t),   0.5*np.sin(OmegaNS*t)]
            zpos = [0.5*np.cos(OmegaNS*t),   0.5*np.cos(OmegaNS*t)]
        else: 
            xpos = [1, -1,  0.5,  0.5, -0.5,  -0.5]
            ypos = [0,  0,  0.0,  0.0,  0.0,   0.0]
            zpos = [0,  0,  0.5, -0.5,  0.5,  -0.5]
        intdir = ['forward', 'forward', 'forward', 'forward', 'forward', 'forward']
        for sn in xrange(len(xpos)):
            print( "Running rendering of streamline #%d..." % (sn+1) ); sys.stdout.flush()
            streamline = mlab.flow(xi_jet, yi_jet, zi_jet, Bxi, Byi, Bzi, seed_scale=0.01,
                             seed_resolution=5,
                             integration_direction=intdir[sn],
                             seedtype='point')
            streamlines.append(streamline)
            streamline.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
            streamline.streamline_type = 'tube'
            streamline.tube_filter.radius = 0.5
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
            streamline.stream_tracer.maximum_propagation = rhead if np.abs(xpos[sn])==1 else 3*rhead
            streamline.seed.widget.position = np.array([ xpos[sn],  ypos[sn],  zpos[sn]])
            streamline.seed.widget.enabled = False
            streamline.update_streamlines = 1
            print( "Done with streamline #%d..." % (sn+1)); sys.stdout.flush()
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
    if 0:
        #show the black hole
        rbh = rhor
        thbh = np.linspace(0,np.pi,128,endpoint=1)[:,None]
        phbh = np.linspace(0,2*np.pi,128,endpoint=1)[None,:]
        thbh = thbh + 0*phbh + 0*thbh
        phbh = phbh + 0*phbh + 0*thbh
        xbh = rbh*sin(thbh)*cos(phbh)
        ybh = rbh*sin(thbh)*sin(phbh)
        zbh = rbh*cos(thbh)
        mlab.mesh(xbh, ybh, zbh, scalars=1+0*thbh, colormap='jet',vmin=0, vmax = 1)
    if 1:
        #myr = 1.05
        myi = 0 #iofr(myr)
        #mesh
        #var = (1.-1./uu[0]**2)**0.5
        if isaligned:
            cvel()
            faraday()
            var = omegaf2*dxdxp[3,3]
        else:
            var = tiltedomegaf()
        #faraday()
        #var = omegaf2*dxdxp[3,3]
        s = wraparound(var)[myi,:,:]
        x = wraparound(r*sin(h)*cos(ph))[myi,:,:]
        y = wraparound(r*sin(h)*sin(ph))[myi,:,:]
        z = wraparound(r*cos(h))[myi,:,:]
        surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=1,vmin=0,vmax=1)
        surf.actor.property.backface_culling = True
        #mlab.colorbar(object=surf,title="Surface angular velocity")
    #move camera:
    #scene.scene.show_axes = False
    scene.scene.camera.position = [0, -200, 0]
    scene.scene.camera.focal_point = [0, 0, 0]
    scene.scene.camera.view_angle = 30.0
    if isaligned:
        scene.scene.camera.view_up = [0.0, 0.0, 1.0]
    else:
        scene.scene.camera.view_up = [1.0, 0.0, 0.0]
    scene.scene.camera.clipping_range = [100, 800]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    #mlab.view(focalpoint=[0,0,0],distance=500)
    #mlab.show()
    print( "Done rendering!" ); sys.stdout.flush()
    if dosavefig:
        print( "Saving snapshot..." ); sys.stdout.flush()
        mlab.savefig("frame%04d.png" % no, figure=scene, magnification=1.0)
        print( "Done!" ); sys.stdout.flush()


def faraday():
    global fdd, fuu, omegaf1, omegaf2, omegaf1b, omegaf2b, rhoc
    if 'fdd' in globals():
        del fdd
    if 'fuu' in globals():
        del fuu
    if 'omegaf1' in globals():
        del omegaf1
    if 'omemaf2' in globals():
        del omegaf2
    # these are native values according to HARM
    fdd = np.zeros((4,4,nx,ny,nz),dtype=rho.dtype)
    #fdd[0,0]=0*gdet
    #fdd[1,1]=0*gdet
    #fdd[2,2]=0*gdet
    #fdd[3,3]=0*gdet
    fdd[0,1]=gdet*(uu[2]*bu[3]-uu[3]*bu[2]) # f_tr
    fdd[1,0]=-fdd[0,1]
    fdd[0,2]=gdet*(uu[3]*bu[1]-uu[1]*bu[3]) # f_th
    fdd[2,0]=-fdd[0,2]
    fdd[0,3]=gdet*(uu[1]*bu[2]-uu[2]*bu[1]) # f_tp
    fdd[3,0]=-fdd[0,3]
    fdd[1,3]=gdet*(uu[2]*bu[0]-uu[0]*bu[2]) # f_rp = gdet*B2
    fdd[3,1]=-fdd[1,3]
    fdd[2,3]=gdet*(uu[0]*bu[1]-uu[1]*bu[0]) # f_hp = gdet*B1
    fdd[3,2]=-fdd[2,3]
    fdd[1,2]=gdet*(uu[0]*bu[3]-uu[3]*bu[0]) # f_rh = gdet*B3
    fdd[2,1]=-fdd[1,2]
    #
    fuu = np.zeros((4,4,nx,ny,nz),dtype=rho.dtype)
    #fuu[0,0]=0*gdet
    #fuu[1,1]=0*gdet
    #fuu[2,2]=0*gdet
    #fuu[3,3]=0*gdet
    fuu[0,1]=-1/gdet*(ud[2]*bd[3]-ud[3]*bd[2]) # f^tr
    fuu[1,0]=-fuu[0,1]
    fuu[0,2]=-1/gdet*(ud[3]*bd[1]-ud[1]*bd[3]) # f^th
    fuu[2,0]=-fuu[0,2]
    fuu[0,3]=-1/gdet*(ud[1]*bd[2]-ud[2]*bd[1]) # f^tp
    fuu[3,0]=-fuu[0,3]
    fuu[1,3]=-1/gdet*(ud[2]*bd[0]-ud[0]*bd[2]) # f^rp
    fuu[3,1]=-fuu[1,3]
    fuu[2,3]=-1/gdet*(ud[0]*bd[1]-ud[1]*bd[0]) # f^hp
    fuu[3,2]=-fuu[2,3]
    fuu[1,2]=-1/gdet*(ud[0]*bd[3]-ud[3]*bd[0]) # f^rh
    fuu[2,1]=-fuu[1,2]
    #
    # these 2 are equal in degen electrodynamics when d/dt=d/dphi->0
    omegaf1=fdd[0,1]/fdd[1,3] # = ftr/frp
    omegaf2=fdd[0,2]/fdd[2,3] # = fth/fhp
    #
    # from jon branch, 04/10/2012
    #
    B1hat=B[1]*np.sqrt(gv3[1,1])
    B2hat=B[2]*np.sqrt(gv3[2,2])
    B3nonhat=B[3]
    v1hat=uu[1]*np.sqrt(gv3[1,1])/uu[0]
    v2hat=uu[2]*np.sqrt(gv3[2,2])/uu[0]
    v3nonhat=uu[3]/uu[0]
    #
    aB1hat=np.fabs(B1hat)
    aB2hat=np.fabs(B2hat)
    av1hat=np.fabs(v1hat)
    av2hat=np.fabs(v2hat)
    #
    vpol=np.sqrt(av1hat**2 + av2hat**2)
    Bpol=np.sqrt(aB1hat**2 + aB2hat**2)
    #
    #omegaf1b=(omegaf1*aB1hat+omegaf2*aB2hat)/(aB1hat+aB2hat)
    #E1hat=fdd[0,1]*np.sqrt(gn3[1,1])
    #E2hat=fdd[0,2]*np.sqrt(gn3[2,2])
    #Epabs=np.sqrt(E1hat**2+E2hat**2)
    #Bpabs=np.sqrt(aB1hat**2+aB2hat**2)+1E-15
    #omegaf2b=Epabs/Bpabs
    #
    # assume field swept back so omegaf is always larger than vphi (only true for outflow, so put in sign switch for inflow as relevant for disk near BH or even jet near BH)
    # GODMARK: These assume rotation about z-axis
    omegaf2b=np.fabs(v3nonhat) + np.sign(uu[1])*(vpol/Bpol)*np.fabs(B3nonhat)
    #
    omegaf1b=v3nonhat - B3nonhat*(v1hat*B1hat+v2hat*B2hat)/(B1hat**2+B2hat**2)
    #
    # charge
    #
    if 0:
        rhoc = np.zeros_like(rho)
        if nx>=2:
            rhoc[1:-1] += ((gdet*fuu[0,1])[2:]-(gdet*fuu[0,1])[:-2])/(2*_dx1)
        if ny>2:
            rhoc[:,1:-1] += ((gdet*fuu[0,2])[:,2:]-(gdet*fuu[0,2])[:,:-2])/(2*_dx2)
        if ny>=2:
            rhoc[:,0,:nz/2] += ((gdet*fuu[0,2])[:,1,:nz/2]+(gdet*fuu[0,2])[:,0,nz/2:])/(2*_dx2)
            rhoc[:,0,nz/2:] += ((gdet*fuu[0,2])[:,1,nz/2:]+(gdet*fuu[0,2])[:,0,:nz/2])/(2*_dx2)
        if nz>2:
            rhoc[:,:,1:-1] += ((gdet*fuu[0,3])[:,:,2:]-(gdet*fuu[0,3])[:,:,:-2])/(2*_dx3)
        if nz>=2:
            rhoc[:,:,0] += ((gdet*fuu[0,3])[:,:,1]-(gdet*fuu[0,3])[:,:,-1])/(2*_dx3)
            rhoc[:,:,-1] += ((gdet*fuu[0,3])[:,:,0]-(gdet*fuu[0,3])[:,:,-2])/(2*_dx3)
        rhoc /= gdet


def vis_pulsar(doreload=1,no=96,xmax=21,ymax=21,zmax=21,ncellx=201,ncelly=201,ncellz=201,dosavefig=0,order=3,dofieldlines=1,updateframe=0,doreinterp=1):
    global mlab_lrho_jet, vol_jet, lrhoi_jet, i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet, Bxi, Byi, Bzi
    if doreload:
        if 'gv3' not in globals():
            grid3d("gdump.bin",use2d=1)
        rfd("fieldline%04d.bin"%no)
        cvel()
    if updateframe:
        scene = mlab.figure(1)
    else:
        scene = mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(960,540))#size=(210*2, 297*2))
        #scene.scene.disable_render = True
    if doreload:
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
    if doreinterp:
        i3d_jet,j3d_jet,k3d_jet,xi_jet,yi_jet,zi_jet =\
            interp3d(xmax=xmax,ymax=ymax,zmax=zmax,ncellx=ncellx,ncelly=ncelly,ncellz=ncellz)
        print( "Running trilinear interpolation for Bx..." ); sys.stdout.flush()
        Bxi = np.float32(trilin(Bx,i3d_jet,j3d_jet,k3d_jet,order=order))
        print( "Done with trilinear interpolation for Bx..." ); sys.stdout.flush()
        print( "Running trilinear interpolation for By..." ); sys.stdout.flush()
        Byi = np.float32(trilin(By,i3d_jet,j3d_jet,k3d_jet,order=order))
        print( "Done with trilinear interpolation for By..." ); sys.stdout.flush()
        print( "Running trilinear interpolation for Bz..." ); sys.stdout.flush()
        Bzi = np.float32(trilin(Bz,i3d_jet,j3d_jet,k3d_jet,order=order))
        print( "Done with trilinear interpolation for Bz..." ); sys.stdout.flush()
    if dofieldlines:
        streamlines = []
        xpos = [Rin*np.sin(AlphaNS), -Rin*np.sin(AlphaNS)]
        ypos = [0, 0]
        zpos = [Rin*np.cos(AlphaNS), -Rin*np.cos(AlphaNS)]
        intdir = ['forward', 'backward']
        if 1:
            for sn in xrange(2):
                print( "Running rendering of streamline #%d..." % (sn+1) ); sys.stdout.flush()
                streamline = mlab.flow(xi_jet, yi_jet, zi_jet, Bxi, Byi, Bzi, seed_scale=0.01,
                                 seed_resolution=5,
                                 integration_direction=intdir[sn],
                                 seedtype='point')
                streamlines.append(streamline)
                streamline.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
                streamline.streamline_type = 'tube'
                streamline.tube_filter.radius = 0.2
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
                print( "Done with streamline #%d..." % (sn+1)); sys.stdout.flush()
                #pdb.set_trace()
        if 1:
            myr = 1.0
            myi = iofr(myr)
            #mesh
            s = wraparound((np.abs(B[1])*dxdxp[1,1]))[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
        if 1:
            myr = 2.5
            myi = iofr(myr)
            #mesh
            s = wraparound((np.abs(B[1])*dxdxp[1,1]))[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
        if 1:
            myr = 1.0
            myi = iofr(myr)
            #mesh
            s = wraparound(bsq)[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
            surf.actor.property.backface_culling = True
        if 1:
            myr = 2.5
            myi = iofr(myr)
            #mesh
            s = wraparound(bsq)[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
            surf.actor.property.backface_culling = True
        if 1:
            myr = 10
            myi = iofr(myr)
            #mesh
            s = wraparound(Brnorm**2+Bhnorm**2+Bpnorm**2)[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
            surf.actor.property.backface_culling = True
        if 1:
            myr = 10
            myi = iofr(myr)
            #mesh
            s = wraparound(bsq)[myi,:,:]
            x = wraparound(r*sin(h)*cos(ph-OmegaNS*t))[myi,:,:]
            y = wraparound(r*sin(h)*sin(ph-OmegaNS*t))[myi,:,:]
            z = wraparound(r*cos(h))[myi,:,:]
            print( "Rotating Br distribution by dphi = %g" % np.arcsin(sin(OmegaNS*t)) )
            surf = mlab.mesh(x, y, z, scalars=s, colormap='jet',opacity=0.85)
            surf.actor.property.backface_culling = True
        if 1:
            #show the NS
            rbh = 1.0
            thbh = np.linspace(0,np.pi,128,endpoint=1)[:,None]
            phbh = np.linspace(0,2*np.pi,128,endpoint=1)[None,:]
            thbh = thbh + 0*phbh + 0*thbh
            phbh = phbh + 0*phbh + 0*thbh
            xbh = rbh*sin(thbh)*cos(phbh)
            ybh = rbh*sin(thbh)*sin(phbh)
            zbh = rbh*cos(thbh)
            mlab.mesh(xbh, ybh, zbh, scalars=1+0*thbh, colormap='hsv',vmin=0, vmax = 1)
    #end if updateframe
    #move camera:
    xyzcam = np.array([0, -1, 0])
    #normalize
    xyzcam /= np.sum(xyzcam**2)**0.5
    #set the distance to desired one
    xyzcam *= zmax*1.2
    #scene.scene.disable_render = False
    #this is needed to draw the scene somehow...
    mlab.view(focalpoint=[0,0,0],distance=zmax)
    scene.scene.camera.position = xyzcam
    scene.scene.camera.focal_point = [0, 0, 0]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.scene.camera.clipping_range = [0.1*zmax, 100*1.2*zmax]
    scene.scene.x_minus_view()
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()
    # iso.contour.maximum_contour = 75.0
    #vec = mlab.pipeline.vectors(d)
    #vec.glyph.mask_input_points = True
    #vec.glyph.glyph.scale_factor = 1.5
    #move the camera so it is centered on (0,0,0)
    #mlab.view(focalpoint=[0,0,0],distance=500)
    #mlab.show()
    print( "Done rendering!" ); sys.stdout.flush()
    if dosavefig:
        print( "Saving snapshot..." ); sys.stdout.flush()
        mlab.savefig("disk_jet_nozoomchange_%04d.png" % no, figure=scene, magnification=2.0)
        print( "Done!" ); sys.stdout.flush()
