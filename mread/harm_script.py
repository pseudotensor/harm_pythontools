import matplotlib
from matplotlib import rc
#Uncomment the following if you want to use LaTeX in figures
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('mathtext',fontset='cm')
# rc('mathtext',rm='stix')
# rc('text', usetex=True)
# #add amsmath to the preamble
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

import numpy as np
import matplotlib.pyplot as plt

#read in a grid file
def rg(dump):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp
    #read image
    fin = open( "dumps/" + dump, "rb" )
    header = fin.readline().split()
    t = myfloat(np.float64(header[0]))
    nx = int(header[1])
    ny = int(header[2])
    nz = 1
    _dx1=myfloat(float(header[5]))
    _dx2=myfloat(float(header[6]))
    _dx3=2*np.pi
    a=myfloat(float(header[9]))
    gam=myfloat(float(header[10]))
    Rin=myfloat(float(header[22]))
    Rout=myfloat(float(header[23]))
    hslope=myfloat(float(header[24]))
    R0=myfloat(float(header[25]))
    fin.close()
    gd = np.loadtxt( "dumps/"+dump, 
                  dtype=np.float64, 
                  skiprows=1, 
                  unpack = True ).view().reshape((-1,ny,nx), order='F')
    #transpose: (quantity,theta,r) to (quantity,r,theta)
    gd=myfloat(gd.transpose(0,2,1))
    #make into 3D array: from (quantity,r,theta) to (quantity,r,theta,phi) with 1 cell in phi
    gd=gd[:,:,:,None]
    #clean up memory
    ti,tj,x1,x2,r,h = gd[0:6,:,:].view()
    tk = 0*ti
    ph = 0*h
    gv3 = gd[6:22].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4)
    gn3 = gd[22:38].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4)
    gcov = gv3
    gcon = gn3
    guu = gn3
    gdd = gv3
    gdet = gd[38]
    drdx = gd[39:55].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4)
    dxdxp = drdx

#read in a dump file
def rd(dump):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor
    #read image
    fin = open( "dumps/" + dump, "rb" )
    header = fin.readline().split()
    t = myfloat(np.float64(header[0]))
    nx = int(header[1])
    ny = int(header[2])
    nz = 1
    _dx1=myfloat(float(header[5]))
    _dx2=myfloat(float(header[6]))
    _dx3=2*np.pi
    a=myfloat(float(header[9]))
    gam=myfloat(float(header[10]))
    Rin=myfloat(float(header[22]))
    Rout=myfloat(float(header[23]))
    hslope=myfloat(float(header[24]))
    R0=myfloat(float(header[25]))
    fin.close()
    gd = np.loadtxt( "dumps/"+dump, 
                  dtype=np.float64, 
                  skiprows=1, 
                  unpack = True ).view().reshape((-1,ny,nx), order='F')
    gd=myfloat(gd.transpose(0,2,1))
    gd=gd[:,:,:,None]
    ti,tj,x1,x2,r,h,rho,ug = gd[0:8,:,:].view() 
    tk = 0*ti
    ph = 0*h
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[8:11]
    B[1:4] = gd[11:14]
    divb = gd[14]
    uu = gd[15:19]
    ud = gd[19:23]
    bu = gd[23:27]
    bd = gd[27:31]
    bsq = mdot(bu,bd)
    v1m,v1p,v2m,v2p=gd[31:35]
    gdet=gd[35]
    rhor = 1+(1-a**2)**0.5
    if "guu" in globals():
        #lapse
        alpha = (-guu[0,0])**(-0.5)

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

def psicalc():
    """
    Computes the field vector potential
    """
    daphi = -(gdet*B[1]).mean(-1)*_dx2
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    return(aphi)

def myfloat(f,acc=1):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1:
        return( np.float32(f) )
    else:
        return( np.float64(f) )

def plco(myvar,xcoord=None,ycoord=None,ax=None,**kwargs):
    global r,h,ph
    plt.clf()
    return plc(myvar,xcoord,ycoord,ax,**kwargs)

def plc(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    global r,h,ph
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k',0)
    mirrory = kwargs.pop('mirrory',0)
    #cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled',False)
    xy = kwargs.pop('xy',0)
    xmax = kwargs.pop('xmax',10)
    ymax = kwargs.pop('ymax',5)
    if xy:
        xcoord = r * np.sin(h)
        ycoord = r * np.cos(h)
        if mirrory: ycoord *= -1
    if (None != xcoord and None != ycoord):
        xcoord = xcoord[:,:,None] if xcoord.ndim == 2 else xcoord[:,:,k:k+1]
        ycoord = ycoord[:,:,None] if ycoord.ndim == 2 else ycoord[:,:,k:k+1]
    myvar = myvar[:,:,None] if myvar.ndim == 2 else myvar[:,:,k:k+1]
    if ax is None:
        ax = plt.gca()
    if( xcoord == None or ycoord == None ):
        if isfilled:
            res = ax.contourf(myvar[:,:,0].transpose(),nc,**kwargs)
        else:
            res = ax.contour(myvar[:,:,0].transpose(),nc,**kwargs)
    else:
        if isfilled:
            res = ax.contourf(xcoord[:,:,0],ycoord[:,:,0],myvar[:,:,0],nc,**kwargs)
        else:
            res = ax.contour(xcoord[:,:,0],ycoord[:,:,0],myvar[:,:,0],nc,**kwargs)
    if( cb == True): #use color bar
        plt.colorbar(res,ax=ax)
    if xy:
        plt.xlim(0,xmax)
        plt.ylim(-ymax,ymax)
    return res

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
    if 0:
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
        if ny>=2 and nz > 1: #not sure if properly works for 2D XXX
            rhoc[:,0,:nz/2] += ((gdet*fuu[0,2])[:,1,:nz/2]+(gdet*fuu[0,2])[:,0,nz/2:])/(2*_dx2)
            rhoc[:,0,nz/2:] += ((gdet*fuu[0,2])[:,1,nz/2:]+(gdet*fuu[0,2])[:,0,:nz/2])/(2*_dx2)
        if nz>2:
            rhoc[:,:,1:-1] += ((gdet*fuu[0,3])[:,:,2:]-(gdet*fuu[0,3])[:,:,:-2])/(2*_dx3)
        if nz>=2:
            rhoc[:,:,0] += ((gdet*fuu[0,3])[:,:,1]-(gdet*fuu[0,3])[:,:,-1])/(2*_dx3)
            rhoc[:,:,-1] += ((gdet*fuu[0,3])[:,:,0]-(gdet*fuu[0,3])[:,:,-2])/(2*_dx3)
        rhoc /= gdet

def Tcalcud():
    global Tud, TudEM, TudMA
    global mu, sigma
    global enth
    global unb, isunbound
    pg = (gam-1)*ug
    w=rho+ug+pg
    eta=w+bsq
    if 'Tud' in globals():
        del Tud
    if 'TudMA' in globals():
        del TudMA
    if 'TudEM' in globals():
        del TudEM
    if 'mu' in globals():
        del mu
    if 'sigma' in globals():
        del sigma
    if 'unb' in globals():
        del unb
    if 'isunbound' in globals():
        del isunbound
    Tud = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudMA = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    TudEM = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
    for kapa in np.arange(4):
        for nu in np.arange(4):
            if(kapa==nu): delta = 1
            else: delta = 0
            TudEM[kapa,nu] = bsq*uu[kapa]*ud[nu] + 0.5*bsq*delta - bu[kapa]*bd[nu]
            TudMA[kapa,nu] = w*uu[kapa]*ud[nu]+pg*delta
            #Tud[kapa,nu] = eta*uu[kapa]*ud[nu]+(pg+0.5*bsq)*delta-bu[kapa]*bd[nu]
            Tud[kapa,nu] = TudEM[kapa,nu] + TudMA[kapa,nu]
    mu = -Tud[1,0]/(rho*uu[1])
    sigma = TudEM[1,0]/TudMA[1,0]
    enth=1+ug*gam/rho
    unb=enth*ud[0]
    isunbound=(-unb>1.0)

def aux():
    faraday()
    Tcalcud()

if __name__ == "__main__":
    if False:
        #1D plot example
        plt.clf()
        rg("gdump")
        rd("dump000")
        plt.plot(r[:,ny/2,0],rho[:,ny/2,0])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("r")
        plt.ylabel("rho")
    if False:
        #2D plot example
        plt.clf()
        rg("gdump")
        rd("dump000")
        #R-z plot of the logarithm of density distribution
        plc(r,np.log10(rho),cb=True,xy=1,xmax=100,ymax=50)
