import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from streamlines import streamplot
from streamlines import fstreamplot
#rc('verbose', level='debug')
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('mathtext',fontset='cm')
#rc('mathtext',rm='stix')
#rc('text', usetex=True)

#from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show

import gc
import numpy as np
import array
#import scipy as sc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.optimize import brentq
#from scipy.interpolate import Rbf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib import mpl
from matplotlib import cm,ticker
from numpy import ma
import matplotlib.colors as colors
import os,glob
import pylab
import sys
import streamlines
from matplotlib.patches import Ellipse
import pdb

#global rho, ug, vu, uu, B, CS
#global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,gdet,conn,gn3,gv3,ck,dxdxp

def get2davg(fname=None,usedefault=0,whichgroup=-1,whichgroups=-1,whichgroupe=-1,itemspergroup=20):
    """Choose usedefault=1 to use average file with raw data.  
       Choose usedefault=2 to use file with floor effects removed
              by applying a floor cutoff (b^2/rho>20 is thrown out) 
              prior to averaging."""
    if whichgroup >= 0:
        whichgroups = whichgroup
        whichgroupe = whichgroup + 1
    elif whichgroupe < 0:
        whichgroupe = whichgroups + 1
    #check values for sanity
    if fname is None and usedefault == 0 and (whichgroups < 0 or whichgroupe < 0 or whichgroups >= whichgroupe or itemspergroup <= 0):
        print( "whichgroups = %d, whichgroupe = %d, itemspergroup = %d not allowed" 
               % (whichgroups, whichgroupe, itemspergroup) )
        return None
    #
    if fname is None:
        if usedefault==1:
            fname = "avg2d.npy"
        elif usedefault==2:
            fname = "avg2dnf.npy"
        else:
            fname = "avg2d%02d_%04d_%04d.npy" % (itemspergroup, whichgroups, whichgroupe)
    if os.path.isfile( fname ):
        print( "File %s exists, loading from file..." % fname )
        sys.stdout.flush()
        avgtot=np.load( fname )
        return( avgtot )
    else:
        print( "File %s does not exist, computing average from fieldline files" % fname )
        sys.stdout.flush()
        #return
    n2avg = 0
    nitems = 0
    myrange = np.arange(whichgroups,whichgroupe)
    numrange = myrange.shape[0]
    for (i,g) in enumerate(myrange):
        avgone=get2davgone( whichgroup = g, itemspergroup = itemspergroup )
        if avgone == None:
            continue
        if 0==i:
            avgtot = np.zeros_like(avgone)
            ts=avgone[0,0,0]
        tf=avgone[0,1,0]
        avgtot += avgone[:avgtot.shape[0]]
        nitems += avgone[0,2,0]
        n2avg += 1
    avgtot[0,0,0] = ts
    avgtot[0,1,0] = tf
    avgtot[0,2,0] = nitems
    #get the average
    if n2avg == 0:
        print( "0 total files, so no data generated." )
        return( None )
    #avoid renormalizing the header
    avgtot[1:] /= n2avg
    #only save if more than 1 dump
    if n2avg > 1:
        print( "Saving data to file..." )
        np.save( fname, avgtot )
    return( avgtot )
    
def rdavg2d(fname=None,usedefault=1):
    if fname is None:
        avgmem = get2davg(usedefault=usedefault)
    else:
        avgmem = get2davg(fname=fname)
    assignavg2dvars(avgmem)
    return(avgmem)

def assignavg2dvars(avgmem,DTf=5):
    global avg_ts,avg_te,avg_te1,avg_nitems,avg_rho,avg_ug,avg_bsq,avg_unb,avg_uu,avg_bu,avg_ud,avg_bd,avg_B,avg_gdetB,avg_omegaf2,avg_rhouu,avg_rhobu,avg_rhoud,avg_rhobd,avg_uguu,avg_ugud,avg_Tud,avg_fdd,avg_rhouuud,avg_uguuud,avg_bsquuud,avg_bubd,avg_uuud
    global avg_TudEM, avg_TudMA, avg_mu, avg_sigma, avg_bsqorho, avg_absB, avg_absgdetB, avg_psisq
    global avg_gamma
    global avg_gdetF
    global avg_bsquu
    global avg_absbu, avg_absbd, avg_absuu, avg_absud, avg_absomegaf2
    #avg defs
    i=0
    avg_ts=avgmem[i,0,:];
    avg_te=avgmem[i,1,:]; 
    avg_nitems=avgmem[i,2,:];i+=1
    avg_te1=np.zeros_like(avg_te)
    if avg_nitems[0]>0:
        if avg_ts[0] > 0 and avg_te[0] == 0: #fix the end time if not saved
            avg_te[0] = avg_ts[0] + (avg_nitems[0]-1) * DTf
            avg_te1[0] = avg_te[0]+ DTf
            print( "Final time not set; guessing DTf = %g and obtaining tf = %g, tf1 = %g" % (DTf, avg_te[0], avg_te1[0]) )
        else:
            avg_te1[0]=avg_te[0]+(avg_te[0]-avg_ts[0])/(avg_nitems[0]-1)
            print( "Extrapolating final time: %g from %g" % (avg_te1[0], avg_te[0]) )
    else:
        print( "Number of elements = 1, so using default final time: %g" % (avg_te[0]) )
        avg_te1[0]=avg_te[0]
    #quantities
    avg_rho=avgmem[i,:,:,None];i+=1
    avg_ug=avgmem[i,:,:,None];i+=1
    avg_bsq=avgmem[i,:,:,None];i+=1
    avg_unb=avgmem[i,:,:,None];i+=1
    n=4
    avg_uu=avgmem[i:i+n,:,:,None];i+=n
    avg_bu=avgmem[i:i+n,:,:,None];i+=n
    avg_ud=avgmem[i:i+n,:,:,None];i+=n
    avg_bd=avgmem[i:i+n,:,:,None];i+=n
    #cell-centered magnetic field components
    n=3;
    avg_B=avgmem[i:i+n,:,:,None];i+=n
    avg_gdetB=avgmem[i:i+n,:,:,None];i+=n
    avg_omegaf2=avgmem[i,:,:,None];i+=1
    #
    n=4
    avg_rhouu=avgmem[i:i+n,:,:,None];i+=n
    avg_rhobu=avgmem[i:i+n,:,:,None];i+=n
    avg_rhoud=avgmem[i:i+n,:,:,None];i+=n
    avg_rhobd=avgmem[i:i+n,:,:,None];i+=n
    avg_uguu=avgmem[i:i+n,:,:,None];i+=n
    avg_ugud=avgmem[i:i+n,:,:,None];i+=n
    #
    n=16
    #energy fluxes and faraday
    avg_Tud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    avg_fdd=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part1: rho u^m u_l
    avg_rhouuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part2: u u^m u_l
    avg_uguuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part3: b^2 u^m u_l
    avg_bsquuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # part6: b^m b_l
    avg_bubd=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    # u^m u_l
    #print( "i = %d, avgmem.shape[0] = %d " % (i, avgmem.shape[0]) )
    #sys.stdout.flush()
    avg_uuud=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
    #new format, extra columns
    if( avgmem.shape[0] > 164 ):
        n=16
        #EM/MA
        avg_TudEM=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
        avg_TudMA=avgmem[i:i+n,:,:,None].reshape((4,4,nx,ny,1));i+=n
        #mu,sigma
        n=1
        avg_mu=avgmem[i,:,:,None];i+=n
        avg_sigma=avgmem[i,:,:,None];i+=n
        avg_bsqorho=avgmem[i,:,:,None];i+=n
        n=3
        avg_absB=avgmem[i:i+n,:,:,None];i+=n
        avg_absgdetB=avgmem[i:i+n,:,:,None];i+=n
        if avgmem.shape[0] >= 206:
            n=1
            avg_psisq=avgmem[i,:,:,None];i+=n
        else:
            n=1
            print( "Old-ish format: missing avg_psisq, filling it in with zeros." )
            avg_psisq=np.zeros_like(avg_mu);i+=n
    else:
        print( "Old format: missing avg_TudEM, avg_TudMA, avg_mu, avg_sigma, avg_bsqorho, etc." )
    if avgmem.shape[0] >= 206+9:
        n=9
        #gdetF
        avg_gdetF=avgmem[i:i+n,:,:,None].reshape((3,3,nx,ny,1));i+=n
    else:
        print( "Old-ish format: missing avg_gdetF etc." )
    if avgmem.shape[0] >= 206+9+4:
        n=4
        avg_bsquu=avgmem[i:i+n,:,:,None];i+=n
    else:
        n=4
        print( "Old-ish format: missing avg_bsquu, filling it in with zeros." )
        avg_bsquu=np.zeros_like(avg_rhouu);i+=n
    if avgmem.shape[0] >= 206+9+4+17:
        n=4
        avg_absbu=avgmem[i:i+n,:,:,None];i+=n
        avg_absbd=avgmem[i:i+n,:,:,None];i+=n
        avg_absuu=avgmem[i:i+n,:,:,None];i+=n
        avg_absud=avgmem[i:i+n,:,:,None];i+=n
        n=1
        avg_absomegaf2=avgmem[i,:,:,None];i+=n
    else:
        print( "Old-ish format: missing avg_absbu, avg_absbd, avg_absuu, avg_absud, avg_absomegaf2" )
    #derived quantities
    avg_gamma=avg_uu[0]/(-gn3[0,0])**0.5


def get2davgone(whichgroup=-1,itemspergroup=20,removefloors=False):
    """
    """
    global avg_ts,avg_te,avg_nitems,avg_rho,avg_ug,avg_bsq,avg_unb,avg_uu,avg_bu,avg_ud,avg_bd,avg_B,avg_gdetB,avg_omegaf2,avg_rhouu,avg_rhobu,avg_rhoud,avg_rhobd,avg_uguu,avg_ugud,avg_Tud,avg_fdd,avg_rhouuud,avg_uguuud,avg_bsquuud,avg_bubd,avg_uuud
    global avg_TudEM, avg_TudMA, avg_mu, avg_sigma, avg_bsqorho, avg_absB, avg_absgdetB, avg_psisq
    global avg_gdetF
    global avg_bsquu
    global avg_absbu, avg_absbd, avg_absuu, avg_absud, avg_absomegaf2
    global rho
    global ug
    if whichgroup < 0 or itemspergroup <= 0:
        print( "whichgroup = %d, itemspergroup = %d not allowed" % (whichgroup, itemspergroup) )
        return None
    fname = "avg2d%02d_%02d.npy" % (itemspergroup, whichgroup)
    if os.path.isfile( fname ):
        print( "File %s exists, loading from file..." % fname )
        avgmem=np.load( fname )
        return( avgmem )
    else:
        print( "File %s does not exist, generating it..." % fname )
    tiny=np.finfo(rho.dtype).tiny
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    flist.sort()
    #
    #print "Number of time slices: %d" % flist.shape[0]
    #store 2D data
    navg=206+9+4+17 #206+9+4+17
    avgmem=np.zeros((navg,nx,ny),dtype=np.float32)
    assignavg2dvars(avgmem)
    ##
    ######################################
    ##
    ## NEED TO ADD vmin/vmax VELOCITY COMPONENTS
    ##
    ######################################
    ##
    #print "Total number of quantities: %d" % (i)
    print "Doing %d-th group of %d items" % (whichgroup, itemspergroup)
    sys.stdout.flush()
    #end avg defs
    for fldindex, fldname in enumerate(flist):
        gc.collect()
        if( whichgroup >=0 and itemspergroup > 0 ):
            if( fldindex / itemspergroup != whichgroup ):
                continue
        print( "Reading " + fldname + " ..." )
        sys.stdout.flush()
        rfd("../"+fldname)
        sys.stdout.flush()
        cvel()  #does not operate on rho and ug, so fine heree
        if removefloors:
            #from Jon
            rinterp=(r-9.0)*(1.0-0.0)/(0.0-9.0)   # gives 0 for use near 9   gives 1 for use near 0
            resetto1=(rinterp>1.0)
            resetto0=(rinterp<0.0)
            rinterp = rinterp*(1-resetto0)*(1-resetto1) + resetto1
            #rinterp[rinterp>1.0]=1.0
            #rinterp[rinterp<0.0]=0.0
            cond3=(bsq/rho < (rinterp*30.0 + (1.0-rinterp)*10.0))
            isfloor = 1-cond3
            #zero out floor contribution
            if isfloor.any():
                rho *= cond3
                ug *= cond3
            del cond3
            del rinterp
            del resetto1
            del resetto0
            del isfloor
        print( "Computing " + fldname + " ..." )
        Tcalcud()
        faraday()
        #if first item in group
        if fldindex == itemspergroup * whichgroup:
            avg_ts[0]=t
        #if last item in group
        if fldindex == itemspergroup * whichgroup + (itemspergroup - 1):
            avg_te[0]=t
        avg_nitems[0]+=1
        #quantities
        avg_rho+=rho.sum(-1)[:,:,None]
        avg_ug+=ug.sum(-1)[:,:,None]
        avg_bsq+=bsq.sum(-1)[:,:,None]
        enth=1+ug*gam/rho
        avg_unb+=(enth*ud[0]).sum(-1)[:,:,None]
        avg_uu+=uu.sum(-1)[:,:,:,None]
        avg_bu+=bu.sum(-1)[:,:,:,None]
        avg_ud+=ud.sum(-1)[:,:,:,None]
        avg_bd+=bd.sum(-1)[:,:,:,None]
        #cell-centered magnetic field components
        n=3;
        avg_B+=B[1:4].sum(-1)[:,:,:,None]
        avg_gdetB+=gdetB[1:4].sum(-1)[:,:,:,None]
        #
        avg_omegaf2+=omegaf2.sum(-1)[:,:,None]
        #
        n=4
        avg_rhouu+=(rho*uu).sum(-1)[:,:,:,None]
        avg_rhobu+=(rho*bu).sum(-1)[:,:,:,None]
        avg_rhoud+=(rho*ud).sum(-1)[:,:,:,None]
        avg_rhobd+=(rho*bd).sum(-1)[:,:,:,None]
        avg_uguu+=(ug*uu).sum(-1)[:,:,:,None]
        avg_ugud+=(ug*ud).sum(-1)[:,:,:,None]
        #
        n=16
        #energy fluxes and faraday
        avg_Tud+=Tud.sum(-1)[:,:,:,:,None]
        avg_fdd+=fdd.sum(-1)[:,:,:,:,None]
        #
        if False:
            #incorrect since sum(-1) is noncommutative
            uuud=odot(uu,ud).sum(-1)[:,:,:,:,None]
            # part1: rho u^m u_l
            avg_rhouuud+=rho.sum(-1)[:,:,None]*uuud
            # part2: u u^m u_l
            avg_uguuud+=ug.sum(-1)[:,:,None]*uuud
            # part3: b^2 u^m u_l
            avg_bsquuud+=bsq.sum(-1)[:,:,None]*uuud
            # part6: b^m b_l
            avg_bubd+=odot(bu,bd)[:,:,:,:,None].sum(-1)
            # u^m u_l
            avg_uuud+=uuud
        else:
            #properly compute average
            uuud=odot(uu,ud)[:,:,:,:,None]
            # part1: rho u^m u_l
            avg_rhouuud+=(rho[:,:,None]*uuud).sum(-1)
            # part2: u u^m u_l
            avg_uguuud+=(ug[:,:,None]*uuud).sum(-1)
            # part3: b^2 u^m u_l
            avg_bsquuud+=(bsq[:,:,None]*uuud).sum(-1)
            # part6: b^m b_l
            avg_bubd+=odot(bu,bd)[:,:,:,:,None].sum(-1)
            # u^m u_l
            avg_uuud+=uuud.sum(-1)

        #EM/MA
        avg_TudEM+=TudEM.sum(-1)[:,:,:,:,None]
        avg_TudMA+=TudMA.sum(-1)[:,:,:,:,None]
        #mu,sigma
        avg_mu += (-Tud[1,0]/(rho*uu[1])).sum(-1)[:,:,None]
        avg_sigma += (-TudEM[1,0]/TudMA[1,0]).sum(-1)[:,:,None]
        avg_bsqorho += (bsq/rho).sum(-1)[:,:,None]
        n=3
        avg_absB += np.abs(B[1:4]).sum(-1)[:,:,:,None]
        avg_absgdetB += np.abs(gdetB[1:4]).sum(-1)[:,:,:,None]
        n=1
        aphi = fieldcalcface()
        avg_psisq += ((_dx3*aphi.sum(-1))**2)[:,:,None]
        n=9
        if gdetF is not None:
            avg_gdetF[:,:] += (gdetF[1:,:].sum(-1))[:,:,:,:,None]
        n=4
        if avg_bsquu is not None:
            avg_bsquu += (bsq*uu).sum(-1)[:,:,:,None]
        #absolute values
        n=17
        if avg_absbu is not None:
            avg_absbu+=np.abs(bu).sum(-1)[:,:,:,None]
            avg_absbd+=np.abs(bd).sum(-1)[:,:,:,None]
            avg_absuu+=np.abs(uu).sum(-1)[:,:,:,None]
            avg_absud+=np.abs(ud).sum(-1)[:,:,:,None]
            avg_absomegaf2+=np.abs(omegaf2).sum(-1)[:,:,None]
    if avg_nitems[0] == 0:
        print( "No files found" )
        return None
    #divide all lines but the header line [which holds (ts,te,nitems)]
    #by the number of elements to get time averages
    avgmem[1:]/=(np.float32(avg_nitems[0])*np.float32(nz))
    print( "Saving to file..." )
    np.save( fname, avgmem )
    print( "Done!" )
    return(avgmem)

def extractlena():
    #along field lines 0.5,0.8,0.99
    cvel()
    Tcalcud()
    faraday()
    psi=fieldcalctoth()/dxdxp[3][3][:,:,0]
    #
    Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
    Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
    Bp = B[3]*dxdxp[3,3]
    #
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #
    Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
    BRnorm=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
    #
    eta=rho*uu[1]/B[1]
    #
    z = -r*np.cos(h)
    R = r*np.sin(h)
    #
    uur = dxdxp[1,1]*uu[1]+dxdxp[1,2]*uu[2]
    uuh = dxdxp[2,1]*uu[1]+dxdxp[2,2]*uu[2]
    uup = uu[3] * dxdxp[3,3]
    #
    uurnorm = uur
    uuhnorm=uuh*np.abs(r)
    uupnorm=uup*np.abs(r*np.sin(h))
    #
    #(cpsi, cr, ch, cR, cz, cBr, cBtheta, cBphi, ceta, crho, cuur, cuutheta, cuuphi, cgamma, comegaf, cmu, csigma, cbsq) = cvals

    vals = (psi, r, h, R, z, -Brnorm, -Bhnorm, -Bpnorm, -eta, rho, uurnorm, uuhnorm, uupnorm, gamma, omegaf2*dxdxp[3][3], mu, sigma, bsq)
    nu = 1.0
    om0 = 0.75

    for psi0 in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0):
        #FIELDLINE: psi = psi0
        fname = "fieldline_nu%0.3g_om%0.2g_psi%0.3g.dat" % (nu, om0, psi0)
        print( "Doing psi = %0.3g, fname = %s" % (psi0, fname) )
        sys.stdout.flush()
        cvals = findroot2d(psi-psi0, vals, axis = 0 )
        fp = open( fname, "wt" )
        fp.write( "#psi, r, \\theta, R, z, B^r, B^\\theta, B^\\phi, \\eta, rho, u^r, u^\\theta, u^\\phi, \\gamma, \\Omega_F, \\mu, \\sigma, b^2 = B^2-E^2\n" )
        fp.flush()
        os.fsync(fp.fileno())
        np.savetxt( fp, np.array(cvals,dtype=np.float64).transpose(), fmt='%21.15g' )
        fp.close()

    for z0 in (10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 5e10):
        #FIELDLINE: psi = psi0
        fname = "fieldline_nu%0.3g_om%0.3g_z%0.3g.dat" % (nu, om0, z0)
        print( "Doing z0 = %0.3g, fname = %s" % (z0, fname) )
        sys.stdout.flush()
        cvals = findroot2d(z-z0, vals, axis = 1 )
        fp = open( fname, "wt" )
        fp.write( "#psi, r, \\theta, R, z, B^r, B^\\theta, B^\\phi, \\eta, rho, u^r, u^\\theta, u^\\phi, \\gamma, \\Omega_F, \\mu, \\sigma, b^2 = B^2-E^2\n" )
        fp.flush()
        os.fsync(fp.fileno())
        np.savetxt( fp, np.array(cvals,dtype=np.float64).transpose(), fmt='%21.15g' )
        fp.close()

def readlena(fname):
    global cpsi, cr, ch, cR, cz, cBr, cBtheta, cBphi, ceta, crho, cuur, cuutheta, cuuphi, cgamma, comegaf, cmu, csigma, cbsq
    cvals = np.loadtxt( fname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True )
    cpsi, cr, ch, cR, cz, cBr, cBtheta, cBphi, ceta, crho, cuur, cuutheta, cuuphi, cgamma, comegaf, cmu, csigma, cbsq = cvals    
    return(cvals)

def findroot2d( fin, xin, isleft=True, nbnd = 1, axis = 0, fallback = 0, fallbackval = 0 ):
    """ returns roots, y(x), so that f(x,y(x)) = 0 (here axis = 0)
        (axis selects independent variable)
    """
    if fin.ndim == 3:
        fin = fin[:,:,0]
    if fin.ndim != 2:
        raise( ValueError( "fin.ndim = %d, should be 2" % fin.ndim ) )
    if axis == 1:
        f = fin.transpose()
    else:
        f = fin
    if not isinstance(xin,tuple):
        xtuple = (xin,)
    else:
        xtuple = xin
    if np.array(fallbackval,ndmin=1).shape[0]==0:
        fallbackvalarr = np.zeros(n)+np.array(fallbackval)[0:1]
    else:
        fallbackvalarr = fallbackval
    xout = ()
    for (j, x) in enumerate(xtuple):
        if x.ndim == 3:
            x = x[:,:,0]
        if x.ndim != 2:
            raise( ValueError( "x[%d].ndim = %d, should be 2" % (j, x.ndim) ) )
        n = f.shape[0]
        if axis == 1:
            x = x.transpose()
        if f.shape != x.shape:
            raise( ValueError( "f and x have different shapes" ) )
        xsol = np.empty((n),dtype=f.dtype)
        for i in np.arange(0,n):
            xsol[i] = findroot1d( f[i], x[i], isleft, nbnd, fallback, fallbackvalarr )
        xout += (xsol,)
    if len(xout) == 1:
        return( xout[0] )
    else:
        return( xout )
        
    
def findroot1d( f, x, isleft=True, nbnd = 1, fallback = 0, fallbackval = 0 ):
    """ find a 1-D root """
    #fallback=1
    if isleft==True:
        ind=0
        dir=1
    else:
        ind=-1
        dir=-1
    if f[ind] > 0:
        coef = 1
    else:
        coef = -1
    #multiplies the final function so f is increasing
    #otherwise scipy gives an error
    if isleft == False:
        interpcoef = coef
    else:
        interpcoef = -coef
    n = x.shape[0]
    ilist = np.arange(0,n)
    indexlist = f*coef<0
    if( not indexlist.any() ):
        if not fallback:
            return( float('nan') )
        else:
            return( fallbackval )
    i0 = ilist[indexlist][ind]
    if f[i0]*f[i0-dir] > 0:
        raise( ValueError("Could not bracket root") )
    ir = i0 + nbnd*dir
    il = i0 - (nbnd+1)*dir
    #limit il, ir to be between 0 and n-1:
    ir = max(0,min(ir,n-1))
    il = max(0,min(il,n-1))
    #order them
    istart = min(il,ir)
    iend = max(il,ir)
    kind = 'linear'
    x2interp = x[istart:iend+1]
    f2interp = interpcoef*f[istart:iend+1]
    #
    if f2interp.shape[0] < 4 or (f2interp[:-1]>=f2interp[1:]).any():
        #too few elements or non-monotonic behavior
        ir = i0 
        il = i0 - dir
        #limit il, ir to be between 0 and n-1:
        ir = max(0,min(ir,n-1))
        il = max(0,min(il,n-1))
        #order them
        istart = min(il,ir)
        iend = max(il,ir)
        kind = 'linear'
        x2interp = x[istart:iend+1]
        f2interp = interpcoef*f[istart:iend+1]
    #
    if f2interp.shape[0]==1:
        #too few elements
        kind='nearest'
        raise( "Too few entries" )
        xinterp = interp1d( f2interp, x2interp, kind=kind, copy = False )
    else:
        xinterp = interp1d( f2interp, x2interp, kind=kind, copy = False )
    ans = xinterp(0.0)
    #if ans < min(x[i0],x[i0-dir]) or ans > max(x[i0],x[i0-dir]):
    #    raise( ValueError("ans = %g out of bounds, (%g,%g)" % (ans,x[i0-dir],x[i0])) )
    return( ans )

def plot2davg(dosq=True,whichplot=-1):
    global eout1, eout2, eout, avg_aphi,avg_aphi2,powjetwind,powjet,jminjet,jmaxjet,jminwind,jmaxwind,mymu,maxaphibh
    #use ratio of averages since more stable definition:
    #
    etad = np.zeros_like(uu)
    etad[0] = -1/(-gn3[0,0])**0.5
    avg_mu = -avg_Tud[1,0] / avg_rhouu[1]
    avg_unb = avg_TudMA[1,0] / avg_rhouu[1]
    #sum away from theta = 0
    muminwind= 1. #1./(1.-0.1**2)**0.5
    muminjet=2.0
    unbcutoff=0 #1./(1.-0.1**2)**0.5-1
    rhor=1+(1-a**2)**0.5
    ihor=iofr(rhor)
    #
    qtymem=getqtyvstime(ihor,0.2)
    if avg_ts[0] != 0:
        fti = avg_ts[0]
    else:
        fti = 8000
    if avg_te1[0] != 0:
        ftf = avg_te1[0]
    else:
        ftf = 1e5
    print( "Using: ti = %g, tf = %g" % (fti,ftf) )
    md, ftot, fsqtot, f30, fsq30, pjemtot  = plotqtyvstime(qtymem,ihor=ihor,whichplot=-1,fti=fti,ftf=ftf)
    #
    avg_aphi = scaletofullwedge(nz*_dx3*fieldcalc(gdetB1=avg_gdetB[0]))
    avg_aphi2 = scaletofullwedge((nz*avg_psisq)**0.5)
    #aphi1 = scaletofullwedge(_dx3*_dx2*(avg_gdetB[0]).cumsum(1))
    #aphi2 = scaletofullwedge(-_dx3*_dx2*(avg_gdetB[0])[:,::-1].cumsum(1)[:,::-1])
    aphi = avg_aphi if dosq==True else avg_aphi
    aphi1 = aphi
    aphi2 = aphi
    maxaphibh = np.max(aphi[ihor])
    mdotden    = scaletofullwedge(nz*(-gdet*avg_rhouu[1]*_dx2*_dx3).sum(axis=2))
    mdottot       = mdotden.sum(-1)
    #
    eout1den   = scaletofullwedge(nz*(-gdet*avg_Tud[1,0]*_dx2*_dx3).sum(axis=2))
    eout1denEM = scaletofullwedge(nz*(-gdet*avg_TudEM[1,0]*_dx2*_dx3).sum(axis=2))
    eout1denMA = scaletofullwedge(nz*(-gdet*avg_TudMA[1,0]*_dx2*_dx3).sum(axis=2))
    #subtract off rest-energy flux
    eout1denKE = scaletofullwedge(nz*(gdet*(-avg_TudMA[1,0]-avg_rhouu[1])*_dx2*_dx3).sum(axis=2))
    #eout1den=eout1denEM+eout1denMA
    #eout1   = eout1den.cumsum(axis=1)
    eoutEM1 = eout1denEM.cumsum(axis=1)
    eoutEMtot = eout1denEM.sum(axis=1)[ihor]
    eoutMA1 = eout1denMA.cumsum(axis=1)
    eoutKE1 = eout1denKE.cumsum(axis=1)
    eout1 = eoutEM1+eoutMA1
    eouttot = eout1den.sum(axis=-1)
    #sum from from theta = pi
    eout2den   = scaletofullwedge(nz*(-gdet*avg_Tud[1,0]*_dx2*_dx3)[:,::-1].sum(axis=2))
    eout2denEM = scaletofullwedge(nz*(-gdet*avg_TudEM[1,0]*_dx2*_dx3)[:,::-1].sum(axis=2))
    eout2denMA = scaletofullwedge(nz*(-gdet*avg_TudMA[1,0]*_dx2*_dx3)[:,::-1].sum(axis=2))
    eout2denKE = scaletofullwedge(nz*(gdet*(-avg_TudMA[1,0]-avg_rhouu[1])*_dx2*_dx3).sum(axis=2))
    #eout2den=eout2denEM+eout2denMA
    #eout2   = eout2den.cumsum(axis=1)[:,::-1]
    eoutEM2 = eout2denEM.cumsum(axis=1)[:,::-1]
    eoutMA2 = eout2denMA.cumsum(axis=1)[:,::-1]
    eoutKE2 = eout2denKE.cumsum(axis=1)[:,::-1]
    eout2 = eoutEM2+eoutMA2
    eout = np.zeros_like(eout1)
    eout[tj[:,:,0]>ny/2] = eout2[tj[:,:,0]>ny/2]
    eout[tj[:,:,0]<=ny/2] = eout1[tj[:,:,0]<=ny/2]
    eoutden = np.zeros_like(eout1den)
    eoutden[tj[:,:,0]>ny/2] = eout2den[tj[:,:,0]>ny/2]
    eoutden[tj[:,:,0]<=ny/2] = eout1den[tj[:,:,0]<=ny/2]
    eoutdenEM = np.zeros_like(eout1denEM)
    eoutdenEM[tj[:,:,0]>ny/2] = eout2denEM[tj[:,:,0]>ny/2]
    eoutdenEM[tj[:,:,0]<=ny/2] = eout1denEM[tj[:,:,0]<=ny/2]
    aphip1 = np.zeros((aphi.shape[0],aphi.shape[1]+1,aphi.shape[2]))    
    aphip1[:,0:ny] = aphi
    daphi = np.zeros_like(ti)
    daphi[:,0:ny]=aphip1[:,1:ny+1]-aphip1[:,0:ny]
    mdot = scaletofullwedge(nz*(gdet*avg_rhouu[1]*_dx2*_dx3)).sum(-1).sum(-1)
    #
    mymu = np.copy(avg_mu)
    mymu[(tj[:,:,0] > ny-10)+(tj[:,:,0] < 10)]=50
    mymu[mymu<1]=50
    #tot
    if False:
        #keep same sign in jet
        hjet1aphia = findroot2d( (aphi1[:,:,0]-maxaphibh)*daphi[:,:,0], h, isleft=True, fallback = 1, fallbackval = np.pi/2 )
        hjet2aphia = findroot2d( (aphi2[:,:,0]-maxaphibh)*daphi[:,:,0], h, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    else:
        #allow different signs in jet
        hjet1aphia = findroot2d( (aphi1[:,:,0]-1*maxaphibh), h, isleft=True, fallback = 1, fallbackval = np.pi/2 )
        hjet2aphia = findroot2d( (aphi2[:,:,0]-1*maxaphibh), h, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    #hjet1aphia*=3
    #hjet2aphia=np.pi-(np.pi-hjet2aphia)*3
    hjet1aphi = np.copy(hjet1aphia)
    hjet2aphi = np.copy(hjet2aphia)
    if False:
        rx=1
        hjet1aphi[r[:,0,0]>rx] = findroot2d( -mymu[:,:,0]+muminjet, h, isleft=True, fallback = 1, fallbackval = np.pi/2  )[r[:,0,0]>rx]
        hjet2aphi[r[:,0,0]>rx] = findroot2d( -mymu[:,:,0]+muminjet, h, isleft=False, fallback = 1, fallbackval = np.pi/2  )[r[:,0,0]>rx]
        hjet1aphi = amin(hjet1aphi,hjet1aphia)
        hjet2aphi = amax(hjet2aphi,hjet2aphia)
        hwind1aphi = findroot2d( -mymu[:,:,0]+muminwind, h, isleft=True, fallback = 1, fallbackval = np.pi/2  )
        hwind2aphi = findroot2d( -mymu[:,:,0]+muminwind, h, isleft=False, fallback = 1, fallbackval = np.pi/2  )
    else:
        hwind1aphi = findroot2d( -avg_uu[1,:,0:ny/2,0], h[:,0:ny/2], isleft=False, fallback = 1, fallbackval = np.pi/2  )
        hwind2aphi = findroot2d( -avg_uu[1,:,ny/2:ny,0], h[:,ny/2:ny], isleft=True, fallback = 1, fallbackval = np.pi/2  )
    ################
    #WIND
    ################
    #
    powwind1 = findroot2d( h[:,:,0] - hwind1aphi[:,None], eout1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powwind2 = findroot2d( h[:,:,0] - hwind2aphi[:,None], eout2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powwind = powwind1+powwind2
    powwindEM1 = findroot2d( h[:,:,0] - hwind1aphi[:,None], eoutEM1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powwindEM2 = findroot2d( h[:,:,0] - hwind2aphi[:,None], eoutEM2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powwindEM = powwindEM1+powwindEM2
    powwindMA1 = findroot2d( h[:,:,0] - hwind1aphi[:,None], eoutMA1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powwindMA2 = findroot2d( h[:,:,0] - hwind2aphi[:,None], eoutMA2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powwindMA = powwindMA1+powwindMA2
    powwindEMKE1 = findroot2d( h[:,:,0] - hwind1aphi[:,None], eoutEM1+eoutKE1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powwindEMKE2 = findroot2d( h[:,:,0] - hwind2aphi[:,None], eoutEM2+eoutKE2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powwindEMKE = powwindEMKE1+powwindEMKE2
    powtot1 = eout1[:,ny/2-1]
    powtot2 = eout2[:,ny/2]
    powtot = powtot1 + powtot2
    powtotEM1 = eoutEM1[:,ny/2-1]
    powtotEM2 = eoutEM2[:,ny/2]
    powtotEM = powtotEM1 + powtotEM2
    powtotMA1 = eoutMA1[:,ny/2-1]
    powtotMA2 = eoutMA2[:,ny/2]
    powtotMA = powtotMA1 + powtotMA2
    #
    ################
    # JET
    ################
    #
    #Tot
    #
    powjet1 = findroot2d( h[:,:,0] - hjet1aphi[:,None], eout1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjet2 = findroot2d( h[:,:,0] - hjet2aphi[:,None], eout2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjet = powjet1+powjet2
    #
    #EM
    #
    powjetEM1 = findroot2d( h[:,:,0] - hjet1aphi[:,None], eoutEM1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetEM2 = findroot2d( h[:,:,0] - hjet2aphi[:,None], eoutEM2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetEM = powjetEM1+powjetEM2
    #
    #KE
    #
    powjetKE1 = findroot2d( h[:,:,0] - hjet1aphi[:,None], eoutKE1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetKE2 = findroot2d( h[:,:,0] - hjet2aphi[:,None], eoutKE2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetKE = powjetKE1+powjetKE2
    powjetEMKE1 = findroot2d( h[:,:,0] - hjet1aphi[:,None], eoutEM1+eoutKE1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetEMKE2 = findroot2d( h[:,:,0] - hjet2aphi[:,None], eoutEM2+eoutKE2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetEMKE = powjetEMKE1+powjetEMKE2
    #
    #MA
    #
    powjetMA1 = findroot2d( h[:,:,0] - hjet1aphi[:,None], eoutMA1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetMA2 = findroot2d( h[:,:,0] - hjet2aphi[:,None], eoutMA2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetMA = powjetMA1+powjetMA2
    #
    #################
    #
    # powjetEM1aphi = findroot2d( aphi1[:,:,0]-maxaphibh, eoutEM1, isleft=True )
    # powjetEM2aphi = findroot2d( aphi2[:,:,0]-maxaphibh, eoutEM2, isleft=False )
    # powjetEM1 = powjetEM1aphi
    # powjetEM2 = powjetEM2aphi
    # powjetEM = powjetEM1+powjetEM2
    #
    #powjetwind1a = findroot2d( (-avg_unb[:,:,0]-(1.0+unbcutoff))*(avg_uu[1,:,:,0]), eout1, isleft=True )
    #powjetwind2a = findroot2d( (-avg_unb[:,:,0]-(1.0+unbcutoff))*(avg_uu[1,:,:,0]), eout2, isleft=False )
    hjetwind1a = findroot2d( (-avg_unb[:,:,0]-(1.0+unbcutoff)), h, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    hjetwind2a = findroot2d( (-avg_unb[:,:,0]-(1.0+unbcutoff)), h, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    hjetwind1b = findroot2d( daphi[:,:,0], h, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    hjetwind2b = findroot2d( daphi[:,:,0], h, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    #limit jet+wind power to be no smaller than jet power
    if False:
        hjetwind1 = amin(hjetwind1b, hjetwind1a) #amax(powjet1,powjetwind1a)
        hjetwind2 = amax(hjetwind2b, hjetwind2a) #amax(powjet2,powjetwind2a)
    else:
        hjetwind1 = hjetwind1a
        hjetwind2 = hjetwind2a
    powjetwind1 = findroot2d( h-hjetwind1[:,None,None], eout1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetwind2 = findroot2d( h-hjetwind2[:,None,None], eout2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetwind = powjetwind1 + powjetwind2
    powjetwindEMKE1 = findroot2d( h-hjetwind1[:,None,None], eoutEM1+eoutKE1, isleft=True, fallback = 1, fallbackval = np.pi/2 )
    powjetwindEMKE2 = findroot2d( h-hjetwind2[:,None,None], eoutEM2+eoutKE2, isleft=False, fallback = 1, fallbackval = np.pi/2 )
    powjetwindEMKE = powjetwindEMKE1 + powjetwindEMKE2
    #
    #Write out power
    #
    rprintout = 100.
    powjetatr = powjetEMKE[iofr(rprintout)]
    powjetwindatr = powjetwindEMKE[iofr(rprintout)]
    powwindatr = powwindEMKE[iofr(rprintout)]
    print( "r = %g: Mdot = %g, etajet = %g, Pjet = %g, etawind = %g, Pwind = %g, Ftot = %g, Fsqtot = %g, pjemtot = %g, eoutEMtot = %g" % ( rprintout, md, powjetatr/md, powjetatr, powjetwindatr/md, powjetwindatr, ftot, fsqtot, pjemtot, eoutEMtot ) )
    foutpower = open( "pjet_2davg_%s.txt" %  os.path.basename(os.getcwd()), "w" )
    printjetwindpower(filehandle = foutpower, r = 100., stage = 0, powjet = powjet, powwind = powwind, muminjet = muminjet, muminwind = muminwind, md=md, powjetEMKE=powjetEMKE, powjetwindEMKE=powjetwindEMKE, 
                      ftot=ftot, fsqtot=fsqtot, f30=f30, fsq30=fsq30, pjemtot=pjemtot, eoutEMtot=eoutEMtot)
    printjetwindpower(filehandle = foutpower, r = 200., stage = 1, powjet = powjet, powwind = powwind, muminjet = muminjet, muminwind = muminwind, md=md, powjetEMKE=powjetEMKE, powjetwindEMKE=powjetwindEMKE)
    printjetwindpower(filehandle = foutpower, r = 400., stage = 2, powjet = powjet, powwind = powwind, muminjet = muminjet, muminwind = muminwind, md=md, powjetEMKE=powjetEMKE, powjetwindEMKE=powjetwindEMKE)
    foutpower.close()
    #HHH
    if whichplot==1:
        #Plot jet, wind, disk powers vs. r
        plt.figure(1)
        plt.clf()
        rhor=1+(1-a**2)**0.5
        r1d=r[:,0,0]
        plt.plot(r1d, powjet, 'b', lw=3,label=r"$P_{\rm j,tot}$")
        plt.plot(r1d, powjetEM, 'b', lw=1,label=r"$P_{\rm j,EM}$")
        plt.plot(r1d, powjetMA, 'b--',lw=1,label=r"$P_{\rm j,MA}$")
        cond=(r1d>5)
        plt.plot(r1d[cond], (powwind-powjet)[cond], 'g', lw=3, label=r"$P_{\rm w,tot}$")
        plt.plot(r1d[cond], (powwindEM-powjetEM)[cond], 'g', lw=1, label=r"$P_{\rm w,EM}$")
        plt.plot(r1d[cond], (powwindMA-powjetMA)[cond], 'g--',lw=1, label=r"$P_{\rm w,MA}$")
        plt.plot(r1d[cond], (powtot-powwind)[cond], 'c', lw=3, label=r"$P_{\rm d,tot}$")
        plt.plot(r1d[cond], (powtotEM-powwindEM)[cond], 'c', lw=1, label=r"$P_{\rm d,EM}$")
        plt.plot(r1d[cond], (powtotMA-powwindMA)[cond], 'c--',lw=1, label=r"$P_{\rm d,MA}$")
        plt.plot(r1d, (powtot-powjet), 'm', lw=3, label=r"$P_{\rm d+w,tot}$")
        plt.plot(r1d, (powtotEM-powjetEM), 'm', lw=1, label=r"$P_{\rm d+w,EM}$")
        plt.plot(r1d, (powtotMA-powjetMA), 'm--',lw=1, label=r"$P_{\rm d+w,MA}$")
        plt.plot(r1d, powtot, 'r', lw=3, label=r"$P_{\rm tot}$")
        plt.plot(r1d, powtotEM, 'r', lw=1, label=r"$P_{\rm tot,EM}$")
        plt.plot(r1d, powtotMA, 'r--',lw=1, label=r"$P_{\rm tot,MA}$")
        plt.plot(r1d, mdot, 'y',lw=3, label=r"$\dot M$")
        plt.legend(ncol=6)
        #plt.plot(r1d, powwind-powjet, label=r"$P_{\rm jet,tot}$")
        plt.xlim(rhor,35)
        plt.ylim(-20,30)
        plt.xlabel(r"$r\ [r_g]$",fontsize=16)
        plt.ylabel("Fluxes",fontsize=16)
        plt.grid()
        #
        plt.figure(2)
        plt.clf()
        #plco(aphi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),colors='k',nc=30)
        #plc(daphi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=(0,),colors='r')
        d=500
        plt.xlim(0,d/2.); plt.ylim(-d,d)
        plc(aphi-maxaphibh,levels=(0,),colors='b',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjet1aphi[:,None,None],levels=(0,),colors='m',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjet2aphi[:,None,None],levels=(0,),colors='m',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hwind1aphi[:,None,None],levels=(0,),colors='c',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hwind2aphi[:,None,None],levels=(0,),colors='c',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(avg_uu[1,:,:,0],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(h-hjetwind1[:,None,None],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(h-hjetwind2[:,None,None],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
     #
    if whichplot==2:
        #plot Mdot vs. r for region v^r < 0
        plt.figure(1)
        ax = plt.gca()
        cond = (avg_uu<0)
        mdot_den = gdet[:,:,0:1]*avg_rhouu
        mdin = intangle( mdot_den, which=cond )
        mdall = intangle( mdot_den )
        plt.plot( r[:,0,0], mdin, ':' )
        plt.plot( r[:,0,0], mdall, '-' )
        ax.set_xscale('log')
        plt.xlim(rhor,20)

    if whichplot==-1:
        #PLOT EVERYTHING
        ##################
        #
        # FIGURE 1
        #
        ##################
        plt.figure(1)
        plt.clf()
        #plco( np.log10(avg_rho), cb=True )
        #plco(choplo(chophi(avg_mu,40),2),cb=True,nc=39)
        plco(avg_mu,levels=np.concatenate(((1.01,),np.arange(2,120,1.))),cb=True)
        #plc(avg_aphi,nc=30,cmap=cm.bone)
        #plc(avg_gamma,levels=np.arange(1.,3.4,0.1),cb=True)
        #
        #plt.clf()
        r1 = 250
        r2 = 1000
        ############
        #
        # FIG 2
        #
        ############
        plt.figure(2)
        plt.clf()
        gs3 = GridSpec(3, 3)
        #gs3.update(left=0.05, right=0.95, top=0.30, bottom=0.03, wspace=0.01, hspace=0.04)
        #mdot
        ax31 = plt.subplot(gs3[-3,:])
        #ymax=ax31.get_ylim()[1]
        #ymax=2*(np.floor(np.floor(ymax+1.5)/2))
        #ax31.set_yticks((ymax/2,ymax))
        i=iofr(r1)
        plt.plot( aphi[i,:,0]/maxaphibh, eout[i,:],'g-' )
        i=iofr(r2)
        plt.plot( aphi[i,:,0]/maxaphibh, eout[i,:],'b-' )
        plt.xlim( 0, 2 )
        plt.ylim( 0, 20 )
        plt.ylabel(r"$P_{\rm j,enc}(\Psi)$")
        ax31.grid(True)
        ax32 = plt.subplot(gs3[-2,:])
        i=iofr(r1)
        print i
        plt.plot( aphi[i,:,0]/maxaphibh, avg_mu[i,:],'g-' )
        plt.plot( aphi[i,:,0]/maxaphibh, avg_gamma[i,:]*(avg_bsqorho[i,:]+1),'g--' )
        i=iofr(r2)
        plt.plot( aphi[i,:,0]/maxaphibh, avg_mu[i,:], 'b-' )
        #somehow don't show up on plot; maybe make use of avg_bsquuud, etc.
        plt.plot( aphi[i,:,0]/maxaphibh, avg_ud[0,i,:]*((gam*avg_ug/avg_rho+avg_bsqorho)[i,:]+1),'b--' )
        plt.plot( aphi[i,:,0]/maxaphibh, avg_ud[0,i,:]*(-avg_sigma[i,:]+1),'b-.' )
        plt.xlim( 0, 2 )
        plt.ylim( 0, 10 )
        plt.ylabel(r"$\mu(\Psi)$")
        ax32.grid(True)
        ax33 = plt.subplot(gs3[-1,:])
        myunb = np.copy(avg_unb)
        myunb[-myunb<=1.+unbcutoff]=myunb[-myunb<=1.0+unbcutoff]*0
        i=iofr(r1)
        #plt.plot( aphi[i,:,0]/maxaphibh, avg_B[0,i,:]/(avg_rhouu[1,i]),'g-' )
        plt.plot( aphi[i,:,0]/maxaphibh, -myunb[i,:],'g-' )
        plt.plot( aphi[i,:,0]/maxaphibh, (avg_uu[1]*dxdxp[1,1])[i,:],'g--' )
        i=iofr(r2)
        #plt.plot( aphi[i,:,0]/maxaphibh, avg_B[0,i,:]/(avg_rhouu[1,i]),'b-' )
        plt.plot( aphi[i,:,0]/maxaphibh, -myunb[i,:],'b-' )
        plt.plot( aphi[i,:,0]/maxaphibh, (avg_uu[1]*dxdxp[1,1])[i,:],'b--' )
        plt.xlim( 0, 2 )
        plt.ylim( -0.5,2)
        plt.ylabel(r"$(1+u/\rho) u_t$")
        ax33.grid(True)
        #print maxaphibh
        #FIG 3
        plt.figure(3)
        plt.clf()
        i=iofr(r1)
        plt.plot( aphi[i,:]/maxaphibh, 'g' )
        plt.plot( aphi[i,:]/maxaphibh, 'gx' )
        i=iofr(r2)
        plt.plot( aphi[i,:]/maxaphibh, 'b' )
        plt.plot( aphi[i,:]/maxaphibh, 'bx' )
        plt.grid()
        ##############
        #
        #  FIGURE 5
        #
        ##############
        plt.figure(5)
        plt.clf()
        plt.plot(r[:,0,0],powjet,'m:',label=r'$P_{jet,EM+MA}$')
        #plt.plot(r[:,0,0],mdottot,'r')
        #plt.plot(r[:,0,0],powjet,'bx')
        plt.ylim(0,20)
        plt.xlim(rhor,30)
        #plt.plot(findroot2d(aphix[:,:,0]-maxaphibh,eout)+findroot2d(aphix[:,:,0]-maxaphibh,eout,isleft=False),'y--')
        #plt.plot(powxjet,'b--')
        #plt.plot(r[:,0,0],powxjetEM,'b--')
        #plt.plot(r[:,0,0],powxjetwind,'g')
        plt.plot(r[:,0,0],eouttot,'r',label=r'$P_{tot}$')
        plt.plot(r[:,0,0],powjetEM,'m-.',label=r'$P_{jet,EM}$')
        plt.plot(r[:,0,0],powjetKE,'m--',label=r'$P_{jet,KE}$')
        plt.plot(r[:,0,0],powjetEMKE,'m',label=r'$P_{jet,EMKE}$')
        #xxx
        #plt.plot(r[iofr(16):,0,0],powjetwind[iofr(16):],'g',label=r'$P_{unbound}(-u_t(1+\Gamma u_g/\rho)>1)$')
        #plt.plot(r[iofr(100):,0,0],powwind[iofr(100):],'c',label=r'$P_{jetwind}(\mu>1.005)$')
        plt.plot(r[iofr(10):,0,0],powjetwindEMKE[iofr(10):],'c--',label=r'$P_{jetwindEMKE}$')
        plt.legend(loc='upper right',ncol=3)
        plt.ylabel(r'Energy fluxes in jet region, $\mu>2$')
        plt.xlabel(r'$r$')
        #plt.plot(r[:,0,0],powjetEM+powjetMA,'k')
        plt.grid()
        plt.figure(8)
        plt.clf()
        plt.plot(r[:,0,0],hjet1aphi,'k')
        plt.plot(r[:,0,0],h[:,ny/4,0],'g')
        plt.plot(r[:,0,0],np.pi-hjet2aphi,'k:')
        plt.xlim(0,1500)
        #xxx
        ##############
        #
        #  FIGURE 6
        #
        ##############
        #
        plt.figure(6)
        plco(aphi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),colors='k',nc=30)
        #plc(daphi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=(0,),colors='r')
        d=500
        plt.xlim(0,d/2.); plt.ylim(-d,d)
        plc(aphi-maxaphibh,levels=(0,),colors='b',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjet1aphi[:,None,None],levels=(0,),colors='m',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjet2aphi[:,None,None],levels=(0,),colors='m',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hwind1aphi[:,None,None],levels=(0,),colors='c',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hwind2aphi[:,None,None],levels=(0,),colors='c',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjetwind1[:,None,None],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(h-hjetwind2[:,None,None],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(chophi(choplo(-avg_unb[:,:,0]-(1.0),0),0.001),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(avg_gamma,nc=50,xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True)
        #
        #plc(chophi((r/rhor)**0.75*(1-np.abs(np.cos(h))),1),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(avg_omegaf2*2*np.pi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True)
        #plc(avg_uu[1]*dxdxp[1][1],xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=(0,))
        #plc(avg_uu[1]*dxdxp[1][1],xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=(0,),colors='g')
        #plc(-avg_unb[:,:,0]-(1.0),levels=(0,0.01,0.1),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        #plc(avg_mu[:,:,0]-(1.0),levels=(0,0.01,0.1),colors='r',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        if False:
            plc(chophi(choplo(-avg_unb[:,:,0]-(1.0),0),0.001),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            rfd("fieldline0000.bin")
            #plc(np.log10(avg_rho),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            plc(avg_uu[1],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            #plc(daphi,levels=(0,),colors='r',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            #plc(h-hjet1aphia[:,None,None],levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            plc(h-hjetwind1[:,None,None],levels=(0,),colors='c',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            plc(h-hjetwind2[:,None,None],levels=(0,),colors='y',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            plc(chophi(choplo(-avg_sigma,0.1),10),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
            plc(chophi((r/rhor)**0.75*(1-np.abs(np.cos(h))),1),xcoord=r*np.sin(h),ycoord=r*np.cos(h))
    #xxx
    #plc(h,xcoord=r*np.sin(h),ycoord=r*np.cos(h))
    #plc(lrho,xcoord=r*np.sin(h),ycoord=r*np.cos(h))

def printjetwindpower(filehandle = None, r = None, stage = 0, powjet = 0, powwind = 0, muminjet = 0, muminwind = 0, md = 0, powjetEMKE=0, powjetwindEMKE=0, ftot=0, fsqtot=0, f30=0, fsq30=0, pjemtot=0, eoutEMtot = 0):
    if filehandle == None or r == None:
        raise( ValuseError("filehandle and r have to be specified") )
    #
    #foutpower.write( "#Name a Mdot   Pjet    Etajet  Psitot Psisqtot**0.5 Psijet Psisqjet**0.5 rstag Pjtotmax Pjtot1rstag Pjtot2rstag Pjtot4rstag Pjtot8rstag\n"  )
    #
    i = iofr(r)
    if stage == 0:
        #initial stage
        filehandle.write( "%s %f %f %f %f %f %f %f %f %f %f %f %f" % (os.path.basename(os.getcwd()), a, avg_ts[0], avg_te1[0], muminjet, muminwind, md, ftot, fsqtot, f30, fsq30, pjemtot, eoutEMtot) )
    if stage == 0 or stage == 1:
        #intermediate stage
        filehandle.write( " %f %f %f %f %f" % (powjetEMKE[i], powjetwindEMKE[i], powjet[i], powwind[i], r) )
    if stage == 2:
        #final stage
        filehandle.write( " %f %f %f %f %f\n" % (powjetEMKE[i], powjetwindEMKE[i], powjet[i], powwind[i], r) )
    #flush to disk just in case to make sure all is written
    filehandle.flush()
    os.fsync(filehandle.fileno())
    
    
def horcalc(which=1):
    """
    Compute root mean square deviation of disk body from equatorial plane
    """
    tiny=np.finfo(rho.dtype).tiny
    up=(gdet*rho*(h-np.pi/2)*which).sum(axis=1)
    dn=(gdet*rho*which).sum(axis=1)
    thetamid2d=up/(dn+tiny)+np.pi/2
    thetamid3d=np.empty((nx,ny,nz),dtype=h.dtype)
    hoverr3d=np.empty((nx,ny,nz),dtype=h.dtype)
    for j in np.arange(0,ny):
        thetamid3d[:,j] = thetamid2d
    up=(gdet*rho*(h-thetamid3d)**2*which).sum(axis=1)
    dn=(gdet*rho*which).sum(axis=1)
    hoverr2d= (up/(dn+tiny))**0.5
    for j in np.arange(0,ny):
        hoverr3d[:,j] = hoverr2d
    return((hoverr3d,thetamid3d))

def intangle(qty,hoverr=None,thetamid=np.pi/2,minbsqorho=None,which=1):
    #somehow gives slightly different answer than when computed directly
    if hoverr == None:
        hoverr = np.pi/2
        thetamid = np.pi/2
    integrand = qty
    insidehor = np.abs(h-thetamid)<hoverr
    if minbsqorho != None:
        insidebsqorho = bsq/rho>=minbsqorho
    else:
        insidebsqorho = 1
    integral=(integrand*insidehor*insidebsqorho*which).sum(axis=2).sum(axis=1)*_dx2*_dx3
    integral=scaletofullwedge(integral)
    return(integral)

# def inttheta(qty,dtheta=np.pi/2):
#     integrand = qty
#     insidedtheta = np.abs(h-np.pi/2)<=dtheta
#     integral=np.sum(np.sum(integrand*insidetheta,axis=2),axis=1)
#     return(integral)

    
def Qmri():
    """
    APPROXIMATELY Computes number of theta cells resolving one MRI wavelength
    """
    global bu,rho,uu,_dx2
    cvel()
    #corrected this expression to include both 2pi and dxdxp[3][3]
    #also corrected defition of va^2 to contain bsq+gam*ug term
    #need to figure out how to properly measure this in fluid frame
    vau2 = np.abs(bu[2])/np.sqrt(rho+bsq+gam*ug)
    omega = dxdxp[3][3]*uu[3]/uu[0]+1e-15
    lambdamriu2 = 2*np.pi * vau2 / omega
    res=lambdamriu2/_dx2
    return(res)

def Qmriavg(dir=2):
    """
    APPROXIMATELY Computes number of theta cells resolving one MRI wavelength
    """
    global avg_bu,avg_rho,avg_uu,_dx2
    #corrected this expression to include both 2pi and dxdxp[3][3]
    #also corrected defition of va^2 to contain bsq+gam*ug term
    #need to figure out how to properly measure this in fluid frame
    if dir == 2:
        vau2 = np.abs(avg_bu[2])/np.sqrt(avg_rho+avg_bsq+gam*avg_ug)
        #vau2 = np.abs(avg_B[1])/np.sqrt(avg_rho+avg_bsq+gam*avg_ug)
        omega = dxdxp[3][3]*np.abs(avg_uu[3])/avg_uu[0]+1e-15
        lambdamriu2 = 2*np.pi * vau2 / omega
        res=lambdamriu2/_dx2
    elif dir == 3:
        vau3 = np.abs(avg_bu[3])/np.sqrt(avg_rho+avg_bsq+gam*avg_ug)
        #vau3 = np.abs(avg_B[2])/np.sqrt(avg_rho+avg_bsq+gam*avg_ug)
        omega = dxdxp[3][3]*np.abs(avg_uu[3])/avg_uu[0]+1e-15
        lambdamriu3 = 2*np.pi * vau3 / omega
        res=lambdamriu3/_dx3
    else:
        pdb.set_trace()
    return(res)


def plco(myvar,xcoord=None,ycoord=None,ax=None,**kwargs):
    plt.clf()
    return plc(myvar,xcoord,ycoord,ax,**kwargs)

def plc(myvar,xcoord=None,ycoord=None,ax=None,**kwargs): #plc
    #xcoord = kwargs.pop('x1', None)
    #ycoord = kwargs.pop('x2', None)
    if(np.min(myvar)==np.max(myvar)):
        print("The quantity you are trying to plot is a constant = %g." % np.min(myvar))
        return
    cb = kwargs.pop('cb', False)
    nc = kwargs.pop('nc', 15)
    k = kwargs.pop('k',0)
    #cmap = kwargs.pop('cmap',cm.jet)
    isfilled = kwargs.pop('isfilled',False)
    if None != xcoord and None != ycoord:
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
    return res

def reinterp(vartointerp,extent,ncell,domask=1,isasymmetric=False):
    global xi,yi,zi
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    xraw=r*np.sin(h)
    yraw=r*np.cos(h)
    x=xraw[:,:,0].view().reshape(-1)
    y=yraw[:,:,0].view().reshape(-1)
    var=vartointerp[:,:,0].view().reshape(-1)
    #mirror
    x=np.concatenate((-x,x))
    y=np.concatenate((y,y))
    kval=min(vartointerp.shape[2]-1,nz/2)
    varmirror = vartointerp[:,:,kval].view().reshape(-1)
    if isasymmetric:
        varmirror *= -1.
    var=np.concatenate((varmirror,var))
    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    # grid the data.
    zi = griddata((x, y), var, (xi[None,:], yi[:,None]), method='linear')
    #zi[interior] = np.ma.masked
    if domask!=0:
        interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < (1+np.sqrt(1-a**2))*domask
        varinterpolated = ma.masked_where(interior, zi)
    else:
        varinterpolated = zi
    return(varinterpolated)

def reinterpxy(vartointerp,extent,ncell,domask=1,mirrorfactor=1):
    global xi,yi,zi
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    xraw=r*np.sin(h)*np.cos(ph)
    yraw=r*np.sin(h)*np.sin(ph)
    #2 cells below the midplane
    x=xraw[:,ny/2+1,:].view().reshape(-1)
    y=yraw[:,ny/2+1,:].view().reshape(-1)
    var=vartointerp[:,ny/2+1,:].view().reshape(-1)
    #mirror
    if nz*_dx3*dxdxp[3,3,0,0,0] < 0.99 * 2 * np.pi:
        x=np.concatenate((-x,x))
        y=np.concatenate((-y,y))
        var=np.concatenate((var*mirrorfactor,var))
    # define grid.
    xi = np.linspace(extent[0], extent[1], ncell)
    yi = np.linspace(extent[2], extent[3], ncell)
    # grid the data.
    zi = griddata((x, y), var, (xi[None,:], yi[:,None]), method='cubic')
    #zi[interior] = np.ma.masked
    if domask!=0:
        interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < (1+np.sqrt(1-a**2))*domask
        varinterpolated = ma.masked_where(interior, zi)
    else:
        varinterpolated = zi
    return(varinterpolated)
    
def ftr(x,xb,xf):
    return( amax(0.0*x,amin(1.0+0.0*x,1.0*(x-xb)/(xf-xb))) )
    
def mkframe(fname,ax=None,cb=True,vmin=None,vmax=None,len=20,ncell=800,pt=True,shrink=1,dostreamlines=True,downsample=4,density=2,dodiskfield=False,minlendiskfield=0.2,minlenbhfield=0.2,dorho=True,dovarylw=True,dobhfield=True,dsval=0.01,color='k',dorandomcolor=False,doarrows=True,lw=None,skipblankint=False,detectLoops=True,minindent=1,minlengthdefault=0.2,startatmidplane=True,showjet=False,arrowsize=1,startxabs=None,startyabs=None,populatestreamlines=True,useblankdiskfield=True,dnarrow=2,whichr=0.9,ncont=100,maxaphi=100,aspect=1.0):
    extent=(-len,len,-len/aspect,len/aspect)
    palette=cm.jet
    palette.set_bad('k', 1.0)
    #palette.set_over('r', 1.0)
    #palette.set_under('g', 1.0)
    rhor=1+(1-a**2)**0.5
    ihor = iofr(rhor)
    ilrho = reinterp(np.log10(rho),extent,ncell,domask=1.0)
    if not dostreamlines:
        aphi = fieldcalc()
        iaphi = reinterp(aphi,extent,ncell,domask=0)
        #maxabsiaphi=np.max(np.abs(iaphi))
        maxabsiaphi = maxaphi #50
        #ncont = 100 #30
        levs=np.linspace(-maxabsiaphi,maxabsiaphi,ncont)
    else:
        aphi = fieldcalc()
        iaphi = reinterp(aphi,extent,ncell,domask=0)
        Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
        Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
        Bp = B[3]*dxdxp[3,3]
        #
        Brnorm=Br
        Bhnorm=Bh*np.abs(r)
        Bpnorm=Bp*np.abs(r*np.sin(h))
        #
        Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
        BRnorm=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
        #
        iBz = reinterp(Bznorm,extent,ncell,domask=0.8)
        iBR = reinterp(BRnorm,extent,ncell,isasymmetric=True,domask=0.8) #isasymmetric = True tells to flip the sign across polar axis
        #
        if dorandomcolor:
            Ba=np.copy(B)
            cond = (B[1]<0)
            Ba[2,cond]*=-1
            Ba[3,cond]*=-1
            Ba[1,cond]*=-1
            Bar = dxdxp[1,1]*Ba[1]+dxdxp[1,2]*Ba[2]
            Bah = dxdxp[2,1]*Ba[1]+dxdxp[2,2]*Ba[2]
            Bap = Ba[3]*dxdxp[3,3]
            #
            Barnorm=Bar
            Bahnorm=Bah*np.abs(r)
            Bapnorm=Bap*np.abs(r*np.sin(h))
            #
            Baznorm=Barnorm*np.cos(h)-Bahnorm*np.sin(h)
            BaRnorm=Barnorm*np.sin(h)+Bahnorm*np.cos(h)
            #
            iBaz = reinterp(Baznorm,extent,ncell,domask=0.8)
            iBaR = reinterp(BaRnorm,extent,ncell,isasymmetric=True,domask=0.8) #isasymmetric = True tells to flip the sign across polar axis
        else:
            iBaz = None
            iBaR = None
        if showjet:
            imu = reinterp(mu,extent,ncell,domask=0.8)
        #
        if dovarylw:
            iibeta = reinterp(0.5*bsq/(gam-1)/ug,extent,ncell,domask=0)
            ibsqorho = reinterp(bsq/rho,extent,ncell,domask=0)
            ibsqo2rho = 0.5 * ibsqorho
        xi = np.linspace(extent[0], extent[1], ncell)
        yi = np.linspace(extent[2], extent[3], ncell)
        #myspeed=np.sqrt(iBR**2+iBz**2)
    #
    #myslines=streamplot(ti[:,0,0],tj[0,:,0],avg_gdetB[0,:,:,0].transpose(),avg_gdetB[1,:,:,0].transpose(),density=2,downsample=4,linewidth=1)
    #for c in cset2.collections:
    #    c.set_linestyle('solid')
    #CS = plt.contourf(xi,yi,zi,15,cmap=palette)
    if ax == None:
        ax = plt.gca()
        # CS = plt.imshow(ilrho, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower',vmin=vmin,vmax=vmax)
        # if not dostreamlines:
        #     cset2 = plt.contour(iaphi,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower',levels=levs)
        # else:
        #     lw = 0.5+1*ftr(np.log10(amax(ibsqo2rho,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
        #     lw += 1*ftr(np.log10(amax(iibeta,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
        #     lw *= ftr(np.log10(amax(iibeta,1e-6+0*iibeta)),-3.5,-3.4)
        #     # if t < 1500:
        #     #     lw *= ftr(ilrho,-2.,-1.9)
        #     fstreamplot(yi,xi,iBR,iBz,density=2,downsample=4,linewidth=lw,ax=ax,detectLoops=True,dodiskfield=False,dobhfield=True,startatmidplane=True,a=a)
        #     #streamplot(yi,xi,iBR,iBz,density=3,linewidth=1,ax=ax)
        # plt.xlim(extent[0],extent[1])
        # plt.ylim(extent[2],extent[3])
    if dorho:
        CS = ax.imshow(ilrho, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower',vmin=vmin,vmax=vmax)
    if showjet:
        ax.contour(imu,linewidths=0.5,colors='g', extent=extent,hold='on',origin='lower',levels=(2,))
        ax.contour(iaphi,linewidths=0.5,colors='b', extent=extent,hold='on',origin='lower',levels=(aphi[ihor,ny/2,0],))
    if not dostreamlines:
        cset2 = ax.contour(iaphi,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower',levels=levs)
        traj = None
    else:
        if dovarylw:
            if False:
                #old way
                lw = 0.5+1*ftr(np.log10(amax(ibsqo2rho,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
                lw += 1*ftr(np.log10(amax(iibeta,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
                lw *= ftr(np.log10(amax(iibeta,1e-6+0*iibeta)),-3.5,-3.4)
                # if t < 1500:
                lw *= ftr(iaphi,0.001,0.002)
            elif True:
                #new way, to avoid glitches in u_g in jet region to affect field line thickness
                lw1 = 2*ftr(np.log10(amax(ibsqo2rho,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
                lw2 = ftr(np.log10(amax(iibeta,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
                lw = 0.5 + amax(lw1,lw2)
                #lw *= ftr(np.log10(amax(iibeta,1e-6+0*iibeta)),-3.5,-3.4)
                # if t < 1500:
                lw *= ftr(iaphi,0.001,0.002)
        #pdb.set_trace()
        traj = fstreamplot(yi,xi,iBR,iBz,ua=iBaR,va=iBaz,density=density,downsample=downsample,linewidth=lw,ax=ax,detectLoops=detectLoops,dodiskfield=dodiskfield,dobhfield=dobhfield,startatmidplane=startatmidplane,a=a,minlendiskfield=minlendiskfield,minlenbhfield=minlenbhfield,dsval=dsval,color=color,doarrows=doarrows,dorandomcolor=dorandomcolor,skipblankint=skipblankint,minindent=minindent,minlengthdefault=minlengthdefault,arrowsize=arrowsize,startxabs=startxabs,startyabs=startyabs,populatestreamlines=populatestreamlines,useblankdiskfield=useblankdiskfield,dnarrow=dnarrow,whichr=whichr)
        #streamplot(yi,xi,iBR,iBz,density=3,linewidth=1,ax=ax)
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    #CS.cmap=cm.jet
    #CS.set_axis_bgcolor("#bdb76b")
    if True == cb:
        plt.colorbar(CS,ax=ax,shrink=shrink) # draw colorbar
    #plt.title(r'$\log_{10}\rho$ at $t = %4.0f$' % t)
    if True == pt:
        plt.title('log rho at t = %4.0f' % t)
    #if None != fname:
    #    plt.savefig( fname + '.png' )
    return(traj)

def mkframexy(fname,ax=None,cb=True,vmin=None,vmax=None,len=20,ncell=800,pt=True,shrink=1,dostreamlines=True,arrowsize=1):
    extent=(-len,len,-len,len)
    palette=cm.jet
    palette.set_bad('k', 1.0)
    #palette.set_over('r', 1.0)
    #palette.set_under('g', 1.0)
    ilrho = reinterpxy(np.log10(rho),extent,ncell,domask=1.0)
    aphi = fieldcalc()+rho*0
    iaphi = reinterpxy(aphi,extent,ncell)
    #maxabsiaphi=np.max(np.abs(iaphi))
    #maxabsiaphi = 100 #50
    #ncont = 100 #30
    #levs=np.linspace(-maxabsiaphi,maxabsiaphi,ncont)
    #cset2 = plt.contour(iaphi,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower',levels=levs)
    #for c in cset2.collections:
    #    c.set_linestyle('solid')
    #CS = plt.contourf(xi,yi,zi,15,cmap=palette)
    if dostreamlines:
        Br = dxdxp[1,1]*B[1]+dxdxp[1,2]*B[2]
        Bh = dxdxp[2,1]*B[1]+dxdxp[2,2]*B[2]
        Bp = B[3]*dxdxp[3,3]
        #
        Brnorm=Br
        Bhnorm=Bh*np.abs(r)
        Bpnorm=Bp*np.abs(r*np.sin(h))
        #
        Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
        BRnorm=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
        Bxnorm=BRnorm*np.cos(ph)-Bpnorm*np.sin(ph)
        Bynorm=BRnorm*np.sin(ph)+Bpnorm*np.cos(ph)
        #
        iBx = reinterpxy(Bxnorm,extent,ncell,domask=1,mirrorfactor=-1.)
        iBy = reinterpxy(Bynorm,extent,ncell,domask=1,mirrorfactor=-1.)
        iibeta = reinterpxy(0.5*bsq/(gam-1)/ug,extent,ncell,domask=0)
        ibsqorho = reinterpxy(bsq/rho,extent,ncell,domask=0)
        ibsqo2rho = 0.5 * ibsqorho
        xi = np.linspace(extent[0], extent[1], ncell)
        yi = np.linspace(extent[2], extent[3], ncell)
    if ax == None:
        CS = plt.imshow(ilrho, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower',vmin=vmin,vmax=vmax)
        plt.xlim(extent[0],extent[1])
        plt.ylim(extent[2],extent[3])
    else:
        CS = ax.imshow(ilrho, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower',vmin=vmin,vmax=vmax)
        if dostreamlines:
            lw = 0.5+1*ftr(np.log10(amax(ibsqo2rho,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
            lw += 1*ftr(np.log10(amax(iibeta,1e-6+0*ibsqorho)),np.log10(1),np.log10(2))
            lw *= ftr(np.log10(amax(iibeta,1e-6+0*iibeta)),-3.5,-3.4)
            # if t < 1500:
            #     lw *= ftr(ilrho,-2.,-1.9)
            lw *= ftr(iaphi,0.001,0.002)
            fstreamplot(yi,xi,iBx,iBy,density=1,downsample=1,linewidth=lw,detectLoops=True,dodiskfield=False,dobhfield=False,startatmidplane=False,a=a,arrowsize=arrowsize)
        ax.set_xlim(extent[0],extent[1])
        ax.set_ylim(extent[2],extent[3])
    #CS.cmap=cm.jet
    #CS.set_axis_bgcolor("#bdb76b")
    if True == cb:
        plt.colorbar(CS,ax=ax,shrink=shrink) # draw colorbar
    #plt.title(r'$\log_{10}\rho$ at $t = %4.0f$' % t)
    if True == pt:
        plt.title('log rho at t = %4.0f' % t)
    #if None != fname:
    #    plt.savefig( fname + '.png' )

def mainfunc(imgname):
    global xi,yi,zi,CS
    #grid3d("gdump")
    #rfd("fieldline0250.bin")
    xraw=r*np.sin(h)
    yraw=r*np.cos(h)
    lrhoraw=np.log10(rho)
    x=xraw[:,:,0].view().reshape(-1)
    y=yraw[:,:,0].view().reshape(-1)
    lrho=lrhoraw[:,:,0].view().reshape(-1)
    #mirror
    x=np.concatenate((-x,x))
    y=np.concatenate((y,y))
    lrho=np.concatenate((lrho,lrho))
    extent=(-41,41,-41,41)
    # define grid.
    xi = np.linspace(-41.0, 41.0, 800)
    yi = np.linspace(-41.0, 41.0, 800)
    # grid the data.
    zi = griddata((x, y), lrho, (xi[None,:], yi[:,None]), method='cubic')
    interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < 1+np.sqrt(1-a**2)
    #zi[interior] = np.ma.masked
    zim = ma.masked_where(interior, zi)
    palette=cm.jet
    palette.set_bad('k', 1.0)
    palette.set_over('r', 1.0)
    palette.set_under('g', 1.0)
    # contour the gridded data, plotting dots at the randomly spaced data points.
    cset2 = plt.contour(zi,15,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower')
    #for c in cset2.collections:
    #    c.set_linestyle('solid')
    #CS = plt.contourf(xi,yi,zi,15,cmap=palette)
    #CS = plt.imshow(zim, extent=[0.01,80,-40,40], cmap = palette, norm = colors.Normalize(vmin=-1,vmax=-0.2,clip = False))
    CS = plt.imshow(zim, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower')
    #CS.cmap=cm.jet
    #CS.set_axis_bgcolor("#bdb76b")
    plt.colorbar(CS) # draw colorbar
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.title('t = %f' % t)
    #plt.show()
    # rbf = Rbf(x[0:288:8,:,0].view().reshape(-1),y[0:288:8,:,0].view().reshape(-1),rho[0:288:8,:,0].view().reshape(-1),epsilon=2)
    # ZI = rbf( XI, YI )
    # # plot the result
    # n = plt.normalize(0.0, 40.0)
    # plt.subplot(1, 1, 1)
    # plt.imshow(XI, YI, ZI, cmap=cm.jet)
    # #plt.scatter(x, y, 100, z, cmap=cm.jet)
    # plt.title('RBF interpolation - multiquadrics')
    # plt.xlim(0, 40.0)
    # plt.ylim(0, 40.0)
    # plt.colorbar()
    # plt.figure()
    # plt.plot(r[:,0,0],np.log10(absflux),'b')
    # plt.legend(['Grid'])
    # plt.axis([r[0,0,0],100,-5,5])
    # plt.title('Grid plot')
    # plt.show()

def ravg(dumpname):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,avgU,avgTud10
    #read image
    ilist=np.array([128,144,160,176,192,208,224])
    for (j,i) in enumerate(ilist):
        fin = open( "dumps/%s-col%04d" % ( dumpname, i ) , "rb" )
        print("Reading %s-col%04d..." % (dumpname,i) )
        header = fin.readline().split()
        t = myfloat(np.float64(header[0]))
        nx = int(header[1])
        ny = int(header[2])
        nz = int(header[3])
        if j == 0:
            avgU=np.zeros((ilist.shape[0],nx,ny,nz),dtype=np.float32)
        _dx1=myfloat(float(header[7]))
        _dx2=myfloat(float(header[8]))
        _dx3=myfloat(float(header[9]))
        gam=myfloat(float(header[11]))
        a=myfloat(float(header[12]))
        Rin=myfloat(float(header[14]))
        Rout=myfloat(float(header[15]))
        if dumpname.endswith(".bin"):
            body = np.fromfile(fin,dtype=np.float64,count=-1)  #nx*ny*nz*11)
            gd = body.view().reshape((-1,nx,ny,nz),order='F')
            fin.close()
        else:
            fin.close()
            gd = np.loadtxt( "dumps/"+dumpname, 
                          dtype=np.float64, 
                          skiprows=1, 
                          unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
        #print np.max(gd)
        avgU[j:j+1] = gd[:,:,:,:].view() 
    avgTud10=avgU.sum(axis=0)
    return

def computeavg(qty):
    avgqty=intangle(gdet*qty)
    return(avgqty)

def doall():
    grid3d("gdump.bin",use2d=True)
    
    if np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) < 1.5:
        #lo-res 0.99 settings
        print( "Using hires a = 0.99 settings")
        ilist=np.arange(221,286)
        dtlist=100.+0*ilist
        dtlist[ilist==221]=22200-22167.6695855045
        dtlist[ilist==222]=22300-22231.9647756934
        dtlist[ilist==228]=22900-22890.8671337456
        dtlist[ilist==232]=23300-23292.5857662206
        dtlist[ilist==239]=24000-23951.5226133435
        dtlist[ilist==245]=24600-24594.4658011928
        dtlist[ilist==251]=25200-25124.6341346588
        dtlist[ilist==257]=25800-25799.8775611997
        dtlist[ilist==263]=26400-26330.0889135128
        dtlist[ilist==267]=26800-26763.9946654502
        dtlist[ilist==274]=27500-27406.4732593203
        dtlist[ilist==280]=28100-28097.4708711805
    elif np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) > 1.5:
        #lo-res 0.99 settings
        print( "Using lores a = 0.99 settings")
        ilist=np.arange(120,137)
        dtlist=100.+0*ilist

    print dtlist 

    for (j,i) in enumerate(ilist):
        dt=dtlist[j]
        ravg("avg%04d.bin" % i)
        if j == 0:
            FE=computeavg(avgTud10)*dt
        else:
            FE+=computeavg(avgTud10)*dt
    #get time average by dividing by the total averaging time
    FE /= dtlist.sum()
    np.save("fe.npy",FE)
    plt.clf()
    plt.plot(r[:,0,0],FE); 
    plt.xlim(rhor,20)
    plt.ylim(-15,15)

def rfloor(dumpname):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,dUfloor
    #read image
    fin = open( "dumps/" + dumpname, "rb" )
    print("Reading %s..." % dumpname)
    header = fin.readline().split()
    t = np.float64(header[0])
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gam=float(header[11])
    a=float(header[12])
    Rin=float(header[14])
    Rout=float(header[15])
    if dumpname.endswith(".bin"):
        body = np.fromfile(fin,dtype=np.float64,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        fin.close()
    else:
        fin.close()
        gd = np.loadtxt( "dumps/"+dumpname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    dUfloor = gd[:,:,:,:].view() 
    return

def rrdump(dumpname,write2xphi=False, whichdir = 3, flipspin = False):
    global nx,ny,nz,t,a,rho,ug,vu,vd,B,gd,gd1,numcols,gdetB,Ucons
    #print( "Reading " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[0])
    ny = int(header[1])
    nz = int(header[2])
    t  = float(header[3])
    a  = float(header[6])
    #nx+=8
    #ny+=8
    #nz+=8
    if dumpname.endswith(".bin"):
        body = np.fromfile(gin,dtype=np.float64,count=-1)  #nx*ny*nz*11)
        gdraw = body
        gin.close()
    else:
        gin.close()
        gdraw = np.loadtxt( "dumps/"+dump, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True )
    gd=gdraw.view().reshape((-1,nx,ny,nz), order='F')
    rho,ug = gd[0:2,:,:,:].view() 
    B = np.zeros_like(gd[4:8])
    vu = np.zeros_like(B)
    vu[1:4] = gd[2:5].view() #relative 4-velocity only has non-zero spatial components
    B[1:4] = gd[5:8].view()
    numcols = gd.shape[0]  #total number of columns is made up of (n prim vars) + (n cons vars) = numcols
    Ucons=gd[numcols/2:numcols/2+5+3] #conserved quantities
    gdetB = np.zeros_like(B)
    gdetB[1:4] = gd[numcols/2+5:numcols/2+5+3]  #gdetB starts with 5th conserved variable
    if 'gv3' in globals() and 'gn3' in globals(): 
        vd = mdot(gv3,vu)
        gamma = (1+mdot(mdot(gv3,vu),vu))**0.5
        etad = np.zeros_like(vu)
        etad[0] = -1/(-gn3[0,0])**0.5      #ZAMO frame velocity (definition)
        etau = mdot(gn3,etad)
        uu = gamma * etau + vu
        ud = mdot(gv3,uu)
    else:
        print( 'Metric (gv3, gn3) not defined, I am skipping the computation of uu and ud' )

    if flipspin:
        print( "Writing out spin-flip rdump...", )
        #write out a dump with flipped spin:
        gout = open( "dumps/" + dumpname + "spinflip", "wb" )
        header[6] = "%21.15g" % (-a)
        for headerel in header:
            s = "%s " % headerel
            gout.write( s )
        gout.write( "\n" )
        gout.flush()
        os.fsync(gout.fileno())
        #reshape the rdump content
        gd1 = gdraw.view().reshape((nz,ny,nx,-1),order='C')
        gd1.tofile(gout)
        gout.close()
        print( " done!" )

    if write2xphi and whichdir is not None:
        print( "Writing out 2xphi rdump...", )
        #write out a dump with twice as many cells in phi-direction:
        gout = open( "dumps/" + dumpname + "2xphi", "wb" )
        #double the number of phi-cells
        newnx = nx
        newny = ny
        newnz = nz
        if whichdir == 3:
            #refine phi-dir
            header[2] = "%d" % (2*nz)
            newnz = 2*nz
        elif whichdir == 2:
            #refine theta-dir
            header[1] = "%d" % (2*ny)
            newny = 2*ny
        for headerel in header:
            s = "%s " % headerel
            gout.write( s )
        gout.write( "\n" )
        gout.flush()
        os.fsync(gout.fileno())
        #reshape the rdump content
        gd1 = gdraw.view().reshape((nz,ny,nx,-1),order='C')
        #allocate memory for refined grid, nz' = 2*nz
        gd2 = np.zeros((newnz,newny,newnx,numcols),order='C',dtype=np.float64)
        if whichdir == 3:
            #copy even k's
            gd2[0::2,:,:,:] = gd1[:,:,:,:]
            #copy odd k's
            gd2[1::2,:,:,:] = gd1[:,:,:,:]
            #in the new cells, adjust gdetB[3] to be averages of immediately adjacent cells (this ensures divb=0)
            gdetB3index = numcols/2+5+2
            gd2[1:-1:2,:,:,gdetB3index] = 0.5*(gd1[:-1,:,:,gdetB3index]+gd1[1:,:,:,gdetB3index])
            gd2[-1,:,:,gdetB3index] = 0.5*(gd1[0,:,:,gdetB3index]+gd1[-1,:,:,gdetB3index])
        elif whichdir == 2:
            #copy even j's
            gd2[:,0::2,:,:] = gd1[:,:,:,:]
            #copy odd j's
            gd2[:,1::2,:,:] = gd1[:,:,:,:]
            #in the new cells, adjust gdetB[2] to be averages of immediately adjacent cells (this ensures divb=0)
            gdetB2index = numcols/2+5+1
            gd2[:,1:-1:2,:,gdetB2index] = 0.5*(gd1[:,:-1,:,gdetB2index]+gd1[:,1:,:,gdetB2index])
            gd2[:,-1,:,gdetB2index] = 0.5*(0.0+gd1[:,-1,:,gdetB2index])
        gd2.tofile(gout)
        gout.close()
        print( " done!" )
        

def fieldcalctoth():
    """
    Computes the field vector potential
    """
    daphi = -(gdetB[1]).sum(-1)*_dx2/nz
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    return(aphi)
   
def fieldcalcU(gdetB1=None):
    """
    Computes cell-centered vector potential
    """
    aphi=fieldcalcface(gdetB1)
    #center it properly in theta
    aphi[:,0:ny-1]=0.5*(aphi[:,0:ny-1]+aphi[:,1:ny]) 
    #special treatment for last cell since no cell at j = ny, and we know aphi[:,ny] should vanish
    aphi[:,ny-1] *= 0.5
    #and in r
    aphi[0:nx-1] = 0.5*(aphi[0:nx-1]  +aphi[1:nx])
    #aphi/=(nz*_dx3)  #<--correction 09/29/2011: this should be left commented out
    return(aphi)

def fieldcalcface(gdetB1=None):
    """
    Computes the field vector potential
    """
    global aphi
    if gdetB1 == None:
        gdetB1 = gdetB[1]
    #average in phi and add up
    daphi = (gdetB1).mean(-1)[:,:,None]*_dx2
    aphi = np.zeros_like(daphi)
    aphi[:,1:ny/2+1]=(daphi.cumsum(axis=1))[:,0:ny/2]
    #sum up from the other pole
    aphi[:,ny/2+1:ny]=(-daphi[:,::-1].cumsum(axis=1))[:,::-1][:,ny/2+1:ny]
    return(aphi)

def fieldcalcface2():
    """
    Computes the field vector potential
    """
    daphi = np.sum(gdetB[2], axis=2)[:,:,None]*_dx1*_dx3
    aphi=daphi.cumsum(axis=0)
    aphi-=daphi #correction for half-cell shift between face and center in theta
    #aphi[0:nx-1] = 0.5*(aphi[0:nx-1]+aphi[1:nx]) #and in r
    aphi/=(nz*_dx3)
    return(aphi)

def rdo(dump,oldfmt=False):
    """ Read in old dump format """
    rd(dump,oldfmt=True)

def rd(dump,oldfmt=False):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd
    global v1m,v1p,v2m,v2p,v3m,v3p,bsq,olddumpfmt
    #read image
    olddumpfmt = oldfmt
    fin = open( "dumps/" + dump, "rb" )
    header = fin.readline().split()
    t = myfloat(np.float64(header[0]))
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=myfloat(float(header[7]))
    _dx2=myfloat(float(header[8]))
    _dx3=myfloat(float(header[9]))
    gam=myfloat(float(header[11]))
    a=myfloat(float(header[12]))
    Rin=myfloat(float(header[14]))
    Rout=myfloat(float(header[15]))
    if dump.endswith(".bin"):
        body = np.fromfile(fin,dtype=np.float64,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        fin.close()
    else:
        fin.close()
        gd = np.loadtxt( "dumps/"+dump, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    gd=myfloat(gd)
    gc.collect()
    ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:,:].view() 
    vu=np.zeros_like(gd[0:4])
    B=np.zeros_like(gd[0:4])
    vu[1:4] = gd[11:14]
    B[1:4] = gd[14:17]
    if not oldfmt:
        pg,cs2,Sden = gd[17:20]
        U = gd[20:29]
        gdetB = np.zeros_like(B)
        gdetB[1:4] = U[5:8]
        divb = gd[29]
        uu = gd[30:34]
        ud = gd[34:38]
        bu = gd[38:42]
        bd = gd[42:46]
        bsq = mdot(bu,bd)
        v1m,v1p,v2m,v2p,v3m,v3p=gd[46:52]
        gdet=gd[53]
    else:
        U = gd[17:25]
        divb = gd[25]
        uu = gd[26:30]
        ud = gd[30:34]
        bu = gd[34:38]
        bd = gd[38:42]
        bsq = mdot(bu,bd)
        v1m,v1p,v2m,v2p,v3m,v3p=gd[42:48]
        gdet=gd[48]
        #gdetB = np.zeros_like(B)
        #gdetB[1:4] = U[5:8]
        gdetB = gdet*B

def rgfd(fieldlinefilename,**kwargs):
    if not os.path.isfile(os.path.join("dumps/", fieldlinefilename)):
        print( "File " + fieldlinefilename + " does not exist. Aborting." )
        return
    if 'gv3' not in globals():
        gdumpname = glob.glob( os.path.join("dumps/", "gdump*") )
        #read the 1st found file
        grid3d(os.path.basename(gdumpname[0]))
    rfd(fieldlinefilename,**kwargs)
    cvel()
    

def rfd(fieldlinefilename,**kwargs):
    #read information from "fieldline" file: 
    #Densities: rho, u, 
    #Velocity components: u1, u2, u3, 
    #Cell-centered magnetic field components: B1, B2, B3, 
    #Face-centered magnetic field components multiplied by metric determinant: gdetB1, gdetB2, gdetB3
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a,Rin,Rout,rho,lrho,ug,uu,uut,uu,B,uux,gdetB,rhor,r,h,ph,gdetF,fdbody
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
    fin = open( "dumps/" + fieldlinefilename, "rb" )
    header = fin.readline().split()
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
    #Spherical polar radius of the innermost radial cell
    Rin=myfloat(float(header[14]))
    #Spherical polar radius of the outermost radial cell
    Rout=myfloat(float(header[15]))
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
    uu=d[4:8,:,:,:]  #again, note uu[i] are 3-velocities (as read from the fieldline file)
    #multiply by u^t to get 4-velocities: u^i = u^t v^i
    uu[1:4]=uu[1:4] * uu[0]
    B = np.zeros_like(uu)
    #cell-centered magnetic field components
    B[1:4,:,:,:]=d[8:11,:,:,:]
    ii=11
    #if the input file contains additional data
    if d.shape[0]==14 or d.shape[0]==23:
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
    if d.shape[0]==20 or d.shape[0]==23:
        print("Loading flux data...")
        gdetF=np.zeros((4,3,nx,ny,nz))
        gdetF[1,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
        gdetF[2,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
        gdetF[3,0:3,:,:,:]=d[ii:ii+3,:,:,:]; ii=ii+3
    else:
        print("No data on gdetF, setting it to None.")
        gdetF = None
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
        gc.collect()


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


def decolumnify(dumpname):
    print( "Reading data from " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname + "-col0000", "rb" )
    header = gin.readline()
    gin.close()
    headersplt = header.split()
    nx = int(headersplt[1])
    ny = int(headersplt[2])
    nz = int(headersplt[3])
    gout = open( "dumps/" + dumpname, "wb" )
    gout.write(header)
    gout.flush()
    os.fsync(gout.fileno())
    flist = np.sort(glob.glob( os.path.join("dumps/", "gdump.bin-col*") ) )
    numfiles = flist.shape[0]
    gd = np.zeros((nz,ny,nx,numfiles),order='C',dtype=np.float64)
    for i,f in enumerate(flist):
        print( "Reading from " + f + " ..." )
        gin = open( f, "rb" )
        header = gin.readline()
        body = np.fromfile(gin,dtype=np.float64,count=-1)  #nx*ny*nz*1
        gd[:,:,:,i:i+1] = body.view().reshape((nz,ny,nx,-1),order='C')
        gin.close()
    print( "Writing to file..." )
    gd.tofile(gout)
    gout.close()
    print( "Done!" )

             
def myfloat(f,acc=1):
    """ acc=1 means np.float32, acc=2 means np.float64 """
    if acc==1:
        return( np.float32(f) )
    else:
        return( np.float64(f) )

def grid3d(dumpname,use2d=False,doface=False): #read grid dump file: header and body
    #The internal cell indices along the three axes: (ti, tj, tk)
    #The internal uniform coordinates, (x1, x2, x3), are mapped into the physical
    #non-uniform coordinates, (r, h, ph), which correspond to radius (r), polar angle (theta), and toroidal angle (phi).
    #There are more variables, e.g., dxdxp, which is the Jacobian of (x1,x2,x3)->(r,h,ph) transformation, that I can
    #go over, if needed.
    global nx,ny,nz,_startx1,_startx2,_startx3,_dx1,_dx2,_dx3,gam,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
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

def grid3dlight(dumpname): #read gdump: header and body
    global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
    print( "Reading grid from " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gin.close()
    #read gdump
    #
    gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True,
                      usecols=(0,1,2,3,4,5,6,7,8,105)).view().reshape((-1,nx,ny,nz), order='F')
    #gd = np.genfromtxt( "dumps/gdump", 
    #                 dtype=np.float64, 
    #                 skip_header=1, 
    #                 skip_footer=nx*ny*(nz-1),
    #                 unpack = True ).view().reshape((137,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph,gdet = gd[:,:,:,:].view() 
    #get the right order of indices by reversing the order of indices i,j(,k)
    #conn=gd[9:73].view().reshape((4,4,4,nx,ny,nz), order='F').transpose(2,1,0,3,4,5)
    #gn3 = gd[73:89].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    #gv3 = gd[89:105].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)
    #gdet = gd[105]
    #ck = gd[106:110].view().reshape((4,nx,ny,nz), order='F')
    #dxdxp = gd[110:136].view().reshape((4,4,nx,ny,nz), order='F').transpose(1,0,2,3,4)

def rdebug(debugfname):
    global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,a
    global fail0,floor0,limitgamma0,inflow0,failrho0,failu0,failrhou0,precgam0,precu0,toentropy0,tocold0,eosfail0
    global fail1,floor1,limitgamma1,inflow1,failrho1,failu1,failrhou1,precgam1,precu1,toentropy1,tocold1,eosfail1
    global fail2,floor2,limitgamma2,inflow2,failrho2,failu2,failrhou2,precgam2,precu2,toentropy2,tocold2,eosfail2
    global fail3,floor3,limitgamma3,inflow3,failrho3,failu3,failrhou3,precgam3,precu3,toentropy3,tocold3,eosfail3
    global dtot0, dtot1, dtot2, dtot3
    global lgdtot0, lgdtot1, lgdtot2, lgdtot3
    global failtot0, failtot1, failtot2, failtot3 
    global lgftot0, lgftot1, lgftot2, lgftot3 
    #read image
    fin = open( "dumps/" + debugfname, "rb" )
    header = fin.readline().split()
    t = np.float64(header[0])
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=myfloat(float(header[7]))
    _dx2=myfloat(float(header[8]))
    _dx3=myfloat(float(header[9]))
    gam=myfloat(float(header[11]))
    a=myfloat(float(header[12]))
    if debugfname.endswith(".bin"):
        body = np.fromfile(fin,dtype=np.float64,count=-1)  #nx*ny*nz*11)
        gd = body.view().reshape((-1,nx,ny,nz),order='F')
        fin.close()
    else:
        fin.close()
        gd = np.loadtxt( "dumps/"+debugfname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True ).view().reshape((-1,nx,ny,nz), order='F')
    gd=myfloat(gd)
    (
       fail0,floor0,limitgamma0,inflow0,failrho0,failu0,failrhou0,precgam0,precu0,toentropy0,tocold0,eosfail0,
       fail1,floor1,limitgamma1,inflow1,failrho1,failu1,failrhou1,precgam1,precu1,toentropy1,tocold1,eosfail1,
       fail2,floor2,limitgamma2,inflow2,failrho2,failu2,failrhou2,precgam2,precu2,toentropy2,tocold2,eosfail2,
       fail3,floor3,limitgamma3,inflow3,failrho3,failu3,failrhou3,precgam3,precu3,toentropy3,tocold3,eosfail3
    ) = gd[0:48,:,:,:].view() 
 
    # shows where *ever* failed or not
    lg1fail=np.log10(fail0+1)
    lg1tot=np.log10(fail0+failrho0+failu0+failrhou0+1)
    #
    lg1precgam=np.log10(precgam0+1)
    lg1precu=np.log10(precu0+1)
    #
    failtot0=fail0+failrho0+failu0+failrhou0
    failtot1=fail1+failrho1+failu1+failrhou1
    failtot2=fail2+failrho2+failu2+failrhou2
    failtot3=fail3+failrho3+failu3+failrhou3
    #
    lgftot0=np.log10(failtot0+1)
    lgftot1=np.log10(failtot1+1)
    lgftot2=np.log10(failtot2+1)
    lgftot3=np.log10(failtot3+1)
    #
    failtot0sum=np.sum(failtot0)
    failtot1sum=np.sum(failtot1)
    failtot2sum=np.sum(failtot2)
    failtot3sum=np.sum(failtot3)
    #
    print( "failtotsum(0,1,2,3): %10d, %10d, %10d, %10d" % (failtot0sum, failtot1sum, failtot2sum, failtot3sum) )
    #
    # absolute totals
    dtot0=fail0+floor0+limitgamma0+failrho0+failu0+failrhou0+precgam0+precu0
    dtot1=fail1+floor1+limitgamma1+failrho1+failu1+failrhou1+precgam1+precu1
    dtot2=fail2+floor2+limitgamma2+failrho2+failu2+failrhou2+precgam2+precu2
    dtot3=fail3+floor3+limitgamma3+failrho3+failu3+failrhou3+precgam3+precu3
    #
    lgdtot0=np.log10(dtot0+1)
    lgdtot1=np.log10(dtot1+1)
    lgdtot2=np.log10(dtot2+1)
    lgdtot3=np.log10(dtot3+1)
    #
    dtot0sum=np.sum(dtot0)
    dtot1sum=np.sum(dtot1)
    dtot2sum=np.sum(dtot2)
    dtot3sum=np.sum(dtot3)
    #
    print( "   dtotsum(0,1,2,3): %10d, %10d, %10d, %10d" % (dtot0sum, dtot1sum, dtot2sum, dtot3sum) )
    #


def rfdgrid(dumpname): #read gdump: header and body
    global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,conn,gn3,gv3,ck,dxdxp,gdet
    print( "Reading grid from " + "dumps/" + dumpname + " ..." )
    gin = open( "dumps/" + dumpname, "rb" )
    header = gin.readline().split()
    nx = int(header[1])
    ny = int(header[2])
    nz = int(header[3])
    _dx1=float(header[7])
    _dx2=float(header[8])
    _dx3=float(header[9])
    gin.close()
    #read gdump
    #
    gd = np.loadtxt( "dumps/" + dumpname, 
                      dtype=np.float64, 
                      skiprows=1, 
                      unpack = True,
                      usecols=(0,1,2,3,4,5,6,7,8)).view().reshape((-1,nx,ny,nz), order='F')
    #gd = np.genfromtxt( "dumps/gdump", 
    #                 dtype=float64, 
    #                 skip_header=1, 
    #                 skip_footer=nx*ny*(nz-1),
    #                 unpack = True ).view().reshape((137,nx,ny,nz), order='F')
    ti,tj,tk,x1,x2,x3,r,h,ph,gdet = gd[:,:,:,:].view() 

def compute_delta():
    """
    Returns a unit matrix
    """
    global delta
    if 'delta' in globals():
        return delta
    #
    delta = np.zeros_like((4,4,nx,ny,nz))
    for i in arange(0,4):
        delta[i,i] = 1
    return(delta)

def odot(a,b):
    """ Outer product of two vectors a^mu b_nu"""
    #the shape of the product is (4,4,nx,ny,max(a.nz,b.nz))
    outer_product = np.zeros(np.concatenate((np.array((4,4)),amax(a[0].shape,b[0].shape))),dtype=np.float32,order='F')
    for mu in np.arange(4):
        for nu in np.arange(4):
            outer_product[mu,nu] = a[mu]*b[nu]
    return(outer_product)

def mdot(a,b):
    """
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
    """
    if a.ndim == 4 and b.ndim == 4:
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

def fieldcalc(gdetB1=None):
    """
    Computes the field vector potential
    """
    #return((1-h[:,:,0]/np.pi)[:,:,None]*fieldcalcp()+(h[:,:,0]/np.pi)[:,:,None]*fieldcalcm())
    return(fieldcalcU(gdetB1))

def mysum2(vec):
    #return( vec[:,:,0][:,:,None]*nz )
    return( np.sum(vec, axis=2)[:,:,None] )

def fcalc():
    """
    Computes the field vector potential
    """
    daphi = np.sum(gdet*B[1],axis=2)*_dx2*_dx3
    aphi=daphi.cumsum(axis=1)
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalcp():
    """
    Computes the field vector potential
    """
    daphi = mysum2(gdet*B[1])*_dx2*_dx3
    aphi=daphi.cumsum(axis=1)
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalcm():
    """
    Computes the field vector potential
    """
    daphi = mysum2(gdet*B[1])*_dx2*_dx3
    aphi=(-daphi[:,::-1].cumsum(axis=1))[:,::-1]
    aphi/=(nz*_dx3)
    return(aphi)

def fieldcalc2():
    """
    Computes the field vector potential
    """
    #daphi = (gdet*B[1]).sum(2)*_dx2*_dx3
    #aphi=daphi.cumsum(axis=1)
    daphi = (-gdet*B[2]).sum(2)*_dx2*_dx3
    #np.zeros_like(B[1,:,:,0:1])
    daphi1 = (gdet[0]*B[1,0]).sum(1).cumsum(axis=0)*_dx2*_dx3
    daphi[0,:] += daphi1
    aphi=daphi.cumsum(axis=0)
    aphi/=(nz*_dx3)
    return(aphi[:,:,None])

def fieldcalc2U():
    """
    Computes the field vector potential
    """
    #daphi = (gdet*B[1]).sum(2)*_dx2*_dx3
    #aphi=daphi.cumsum(axis=1)
    daphi = (-gdetB[2]).sum(2)*_dx2*_dx3
    #np.zeros_like(B[1,:,:,0:1])
    daphi1 = (gdetB[1,0]).sum(1).cumsum(axis=0)*_dx2*_dx3
    daphi[0,:] += daphi1
    aphi=daphi.cumsum(axis=0)
    aphi=scaletofullwedge(aphi)
    return(aphi[:,:,None])

def horfluxcalc(ihor=None,minbsqorho=10):
    """
    Computes the absolute flux through the sphere i = ihor
    """
    global gdetB, _dx2, _dx3
    #1D function of theta only:
    dfabs = (np.abs(gdetB[1]*(bsq/rho>minbsqorho))).sum(2)*_dx2*_dx3
    fabs = dfabs.sum(axis=1)
    #account for the wedge
    fabs=scaletofullwedge(fabs)
    #fabs *= 
    if ihor == None:
        return(fabs)
    else:
        return(fabs[ihor])


def scaletofullwedge(val):
    return(val * 2*np.pi/(dxdxp[3,3,0,0,0]*nz*_dx3))

# def mdotcalc(whichi=None,minbsqorho=None):
#     mdotden = -gdet*rho*uu[1]
#     if minbsqorho != None:
#         mdotden[bsq/rho<minbsqorho<minbsqorho] = 0*mdotden[bsq/rho<minbsqorho<minbsqorho] 
#     mdottot = scaletofullwedge(np.sum(np.sum(mdotden,axis=2),axis=1)*_dx2*_dx3)
#     if whichi == None:
#         return(mdottot)
#     else:
#         return(mdottot[whichi])

def mdotcalc(ihor=None,**kwargs):
    """
    Computes the absolute flux through the sphere i = ihor
    """
    #1D function of theta only:
    md = intangle( -gdet*rho*uu[1], **kwargs)
    if ihor==None:
        return(md)
    else:
        return(md[ihor])

def diskfluxcalc(jmid,rmin=None,rmax=None):
    """
    Computes the absolute flux through the disk midplane at j = jmid
    """
    global gdetB,_dx1,_dx3,r
    #1D function of theta only:
    dfabs = (np.abs(gdetB[2,:,jmid,:])).sum(1)*_dx1*_dx3
    if rmax != None:
        dfabs = dfabs*(r[:,0,0]<=rmax)
    if rmin != None:
        dfabs = dfabs*(r[:,0,0]>=rmin)
    fabs = dfabs.sum(axis=0)
    fabs=scaletofullwedge(fabs)
    return(fabs)

def mfjhorvstime(ihor):
    """
    Returns a tuple (ts,fs,mdot,pjetem,pjettot): lists of times, horizon fluxes, and Mdot
    """
    flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline*.bin") ) )
    flist.sort()
    ts=np.empty(len(flist),dtype=np.float32)
    fs=np.empty(len(flist),dtype=np.float32)
    md=np.empty(len(flist),dtype=np.float32)
    jem=np.empty(len(flist),dtype=np.float32)
    jtot=np.empty(len(flist),dtype=np.float32)
    for findex, fname in enumerate(flist):
        print( "Reading " + fname + " ..." )
        rfd("../"+fname)
        cvel()
        Tcalcud()
        fs[findex]=horfluxcalc(ihor)
        md[findex]=mdotcalc(ihor)
        #EM
        jem[findex]=jetpowcalc(0)[ihor]
        #tot
        jtot[findex]=jetpowcalc(2)[ihor]
        ts[findex]=t
        #if os.path.isfile("lrho%04d.png" % findex):
        #    print( "Skipping " + fname + " as lrho%04d.png exists" % findex );
        #else:
        #    print( "Reinterpolating " + fname + " ..." )
        #    plt.figure(0)
        #    plt.clf()
        #    mkframe("lrho%04d" % findex, vmin=-8,vmax=0.2)
    print( "Done!" )
    return((ts,fs,md,jem,jtot))

def mergeqtyvstime(n):
    nqty = getqtyvstime(0,getnqty=True)
    for i in np.arange(n):
        #load each file
        fname = "qty2_%d_%d.npy" % (i, n)
        print( "Loading " + fname + " ..." )
        sys.stdout.flush()
        qtymemtemp = np.load( fname )
        #per-element sum relevant parts of each file
        if i == 0:
            qtymem = np.zeros((nqty,qtymemtemp.shape[1],qtymemtemp.shape[2]),dtype=np.float32)
        #1st index: whichqty
        #2nd index: whichdumpnumber
        qtymem[:qtymemtemp.shape[0],i::n] += qtymemtemp[:,i::n]
    fname = "qty2.npy"
    print( "Saving into " + fname + " ..." )
    sys.stdout.flush()
    np.save( fname , qtymem )
    print( "Done!" )
        

def getqtyvstime(ihor,horval=0.2,fmtver=2,dobob=0,whichi=None,whichn=None,docompute=False,getnqty=False):
    """
    Returns a tuple (ts,fs,mdot,pjetem,pjettot): lists of times, horizon fluxes, and Mdot
    """
    nqtyold=98+134*(dobob==1)
    nqty=98+134*(dobob==1)+32+1+9
    if getnqty:
        return(nqty)
    if whichn != None and (whichi < 0 or whichi > whichn):
        print( "whichi = %d shoudl be >= 0 and < whichn = %d" % (whichi, whichn) )
        return( -1 )
    if 'rho' in globals():
        tiny=np.finfo(rho.dtype).tiny
    else:
        tiny = np.finfo(np.float64).tiny
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    flist.sort()
    #store 1D data
    numtimeslices=len(flist)
    #np.seterr(invalid='raise',divide='raise')
    #
    print "Number of time slices: %d" % numtimeslices
    if whichi >=0 and whichi < whichn:
        fname = "qty2_%d_%d.npy" % (whichi, whichn)
    else:
        fname = "qty2.npy" 
    if fmtver == 2 and os.path.isfile( fname ):
        qtymem2=np.load( fname )
        loadednqty=qtymem2.shape[0]
        numtimeslices2 = qtymem2.shape[1]
        #require same number of variables, don't allow format changes on the fly for safety
        print "Number of previously saved time slices: %d" % numtimeslices2 
        if( numtimeslices2 >= numtimeslices ):
            print "Number of previously saved time slices is >= than of timeslices to be loaded, re-using previously saved time slices"
            # qtymem2[:,1493]=0.5*(qtymem2[:,1492]+qtymem2[:,1494])
            # qtymem2=qtymem2[:,:-1]
            # np.save("qty2_new.npy",qtymem2)  #kill bad frame
            # if nqty != loadednqty:
            #     qtymem=np.zeros((nqty,numtimeslices2,nx),dtype=np.float32)
            #     qtymem[:loadednqty] = qtymem2[:]
            #     del qtymem2
            #     return(qtymem)
            return(qtymem2)
        elif docompute==False:
            print "Read from %s" % fname
            # if nqty != loadednqty:
            #     qtymem=np.zeros((nqty,numtimeslices2,nx),dtype=np.float32)
            #     qtymem[:loadednqty] = qtymem2[:]
            #     del qtymem2
            #     return(qtymem)
            return(qtymem2)
        else:
            qtymem=np.zeros((nqty,numtimeslices,nx),dtype=np.float32)
            assert loadednqty == nqty or loadednqty+9 == nqty
            print "Number of previously saved time slices is < than of timeslices to be loaded, re-using previously saved time slices"
            qtymem[:loadednqty,0:numtimeslices2] = qtymem2[:,0:numtimeslices2]
            del qtymem2
            qtymem2=None
    elif fmtver == 1 and os.path.isfile("qty.npy"):
        qtymem2=np.load( "qty.npy" )
        numtimeslices2 = qtymem2.shape[1]
        print "Number of previously saved time slices: %d" % numtimeslices2 
        print "Instructed to use old format, reusing prev. saved slices"
        return(qtymem2)
    else:
        numtimeslices2 = 0
        qtymem=np.zeros((nqty,numtimeslices,nx),dtype=np.float32)
    #qty defs
    i=0
    ts=qtymem[i,:,0];i+=1
    #HoverR
    hoverr=qtymem[i];i+=1
    thetamid=qtymem[i];i+=1
    #rhosq:
    rhosqs=qtymem[i];i+=1
    rhosrhosq=qtymem[i];i+=1
    ugsrhosq=qtymem[i];i+=1
    uu0rhosq=qtymem[i];i+=1
    uus1rhosq=qtymem[i];i+=1
    uuas1rhosq=qtymem[i];i+=1
    uus3rhosq=qtymem[i];i+=1
    uuas3rhosq=qtymem[i];i+=1
    Bs1rhosq=qtymem[i];i+=1
    Bas1rhosq=qtymem[i];i+=1
    Bs2rhosq=qtymem[i];i+=1
    Bas2rhosq=qtymem[i];i+=1
    Bs3rhosq=qtymem[i];i+=1
    Bas3rhosq=qtymem[i];i+=1
    #2h
    gdetint2h=qtymem[i];i+=1
    rhos2h=qtymem[i];i+=1
    ugs2h=qtymem[i];i+=1
    uu02h=qtymem[i];i+=1
    uus12h=qtymem[i];i+=1
    uuas12h=qtymem[i];i+=1
    uus32h=qtymem[i];i+=1
    uuas32h=qtymem[i];i+=1
    Bs12h=qtymem[i];i+=1
    Bas12h=qtymem[i];i+=1
    Bs22h=qtymem[i];i+=1
    Bas22h=qtymem[i];i+=1
    Bs32h=qtymem[i];i+=1
    Bas32h=qtymem[i];i+=1
    #4h
    gdetint4h=qtymem[i];i+=1
    rhos4h=qtymem[i];i+=1
    ugs4h=qtymem[i];i+=1
    uu04h=qtymem[i];i+=1
    uus14h=qtymem[i];i+=1
    uuas14h=qtymem[i];i+=1
    uus34h=qtymem[i];i+=1
    uuas34h=qtymem[i];i+=1
    Bs14h=qtymem[i];i+=1
    Bas14h=qtymem[i];i+=1
    Bs24h=qtymem[i];i+=1
    Bas24h=qtymem[i];i+=1
    Bs34h=qtymem[i];i+=1
    Bas34h=qtymem[i];i+=1
    #2hor
    gdetint2hor=qtymem[i];i+=1
    rhos2hor=qtymem[i];i+=1
    ugs2hor=qtymem[i];i+=1
    bsqs2hor=qtymem[i];i+=1
    bsqorhos2hor=qtymem[i];i+=1
    bsqougs2hor=qtymem[i];i+=1
    uu02hor=qtymem[i];i+=1
    uus12hor=qtymem[i];i+=1
    uuas12hor=qtymem[i];i+=1
    uus32hor=qtymem[i];i+=1
    uuas32hor=qtymem[i];i+=1
    Bs12hor=qtymem[i];i+=1
    Bas12hor=qtymem[i];i+=1
    Bs22hor=qtymem[i];i+=1
    Bas22hor=qtymem[i];i+=1
    Bs32hor=qtymem[i];i+=1
    Bas32hor=qtymem[i];i+=1
    #Flux
    fstot=qtymem[i];i+=1
    fs2hor=qtymem[i];i+=1
    fsj5=qtymem[i];i+=1
    fsj10=qtymem[i];i+=1
    fsj20=qtymem[i];i+=1
    fsj30=qtymem[i];i+=1
    fsj40=qtymem[i];i+=1
    #Mdot
    mdtot=qtymem[i];i+=1
    md2h=qtymem[i];i+=1
    md4h=qtymem[i];i+=1
    md2hor=qtymem[i];i+=1
    md5=qtymem[i];i+=1
    md10=qtymem[i];i+=1
    md20=qtymem[i];i+=1
    md30=qtymem[i];i+=1
    md40=qtymem[i];i+=1
    mdrhosq=qtymem[i];i+=1
    mdtotbound=qtymem[i];i+=1
    #Edot
    edtot=qtymem[i];i+=1
    ed2h=qtymem[i];i+=1
    ed4h=qtymem[i];i+=1
    ed2hor=qtymem[i];i+=1
    edrhosq=qtymem[i];i+=1
    edma=qtymem[i];i+=1
    edtotbound=qtymem[i];i+=1
    edmabound=qtymem[i];i+=1
    #Pjet
    pjem5=qtymem[i];i+=1
    pjem10=qtymem[i];i+=1
    pjem20=qtymem[i];i+=1
    pjem30=qtymem[i];i+=1
    pjem40=qtymem[i];i+=1
    pjma5=qtymem[i];i+=1
    pjma10=qtymem[i];i+=1
    pjma20=qtymem[i];i+=1
    pjma30=qtymem[i];i+=1
    pjma40=qtymem[i];i+=1
    #new format?
    if qtymem.shape[0] > nqtyold:
        #yes!
        pjem_n_mu10=qtymem[i];i+=1
        pjem_n_mu5=qtymem[i];i+=1
        pjem_n_mu2=qtymem[i];i+=1
        pjem_n_mu1=qtymem[i];i+=1
        pjrm_n_mu10=qtymem[i];i+=1
        pjrm_n_mu5=qtymem[i];i+=1
        pjrm_n_mu2=qtymem[i];i+=1
        pjrm_n_mu1=qtymem[i];i+=1
        pjma_n_mu10=qtymem[i];i+=1
        pjma_n_mu5=qtymem[i];i+=1
        pjma_n_mu2=qtymem[i];i+=1
        pjma_n_mu1=qtymem[i];i+=1
        phiabsj_n_mu10=qtymem[i];i+=1
        phiabsj_n_mu5=qtymem[i];i+=1
        phiabsj_n_mu2=qtymem[i];i+=1
        phiabsj_n_mu1=qtymem[i];i+=1
        pjem_s_mu10=qtymem[i];i+=1
        pjem_s_mu5=qtymem[i];i+=1
        pjem_s_mu2=qtymem[i];i+=1
        pjem_s_mu1=qtymem[i];i+=1
        pjrm_s_mu10=qtymem[i];i+=1
        pjrm_s_mu5=qtymem[i];i+=1
        pjrm_s_mu2=qtymem[i];i+=1
        pjrm_s_mu1=qtymem[i];i+=1
        pjma_s_mu10=qtymem[i];i+=1
        pjma_s_mu5=qtymem[i];i+=1
        pjma_s_mu2=qtymem[i];i+=1
        pjma_s_mu1=qtymem[i];i+=1
        phiabsj_s_mu10=qtymem[i];i+=1
        phiabsj_s_mu5=qtymem[i];i+=1
        phiabsj_s_mu2=qtymem[i];i+=1
        phiabsj_s_mu1=qtymem[i];i+=1
        if i < qtymem.shape[0]:
            ldtot=qtymem[i];i+=1
        else:
            ldtot=None
        if i < qtymem.shape[0]:
            print( "Getting memory ready for gdetF's" )
            gdetF10=qtymem[i];i+=1
            gdetF11=qtymem[i];i+=1
            gdetF12=qtymem[i];i+=1
            gdetF20=qtymem[i];i+=1
            gdetF21=qtymem[i];i+=1
            gdetF22=qtymem[i];i+=1
            gdetF30=qtymem[i];i+=1
            gdetF31=qtymem[i];i+=1
            gdetF32=qtymem[i];i+=1
        else:
            print( "Setting gdetF's to None" )
            gdetF10=None
            gdetF11=None
            gdetF12=None
            gdetF20=None
            gdetF21=None
            gdetF22=None
            gdetF30=None
            gdetF31=None
            gdetF32=None
    else:
        print( "Oldish format: missing north/south jet power and flux" )
        sys.stdout.flush()
        pjem_n_mu10=None
        pjem_n_mu5=None
        pjem_n_mu2=None
        pjem_n_mu1=None
        pjrm_n_mu10=None
        pjrm_n_mu5=None
        pjrm_n_mu2=None
        pjrm_n_mu1=None
        pjma_n_mu10=None
        pjma_n_mu5=None
        pjma_n_mu2=None
        pjma_n_mu1=None
        phiabsj_n_mu10=None
        phiabsj_n_mu5=None
        phiabsj_n_mu2=None
        phiabsj_n_mu1=None
        pjem_s_mu10=None
        pjem_s_mu5=None
        pjem_s_mu2=None
        pjem_s_mu1=None
        pjrm_s_mu10=None
        pjrm_s_mu5=None
        pjrm_s_mu2=None
        pjrm_s_mu1=None
        pjma_s_mu10=None
        pjma_s_mu5=None
        pjma_s_mu2=None
        pjma_s_mu1=None
        phiabsj_s_mu10=None
        phiabsj_s_mu5=None
        phiabsj_s_mu2=None
        phiabsj_s_mu1=None
        ldtot=None
    if dobob == 1:
        print "Total number of quantities: %d+134 = %d" % (i, i+134)
    else:
        print "Total number of quantities: %d" % (i)
    if( whichi >=0 and whichn > 0 ):
        print "Doing every %d-th slice of %d" % (whichi, whichn)
    sys.stdout.flush()
    #end qty defs
    for findex, fname in enumerate(flist):
        if( whichi >=0 and whichn > 0 ):
            if( findex % whichn != whichi ):
                continue
        #skip pre-loaded time slices
        if findex < numtimeslices2: 
            continue
        #call garbage collector -- trying to get req'd memory under control
        gc.collect()
        print( "Reading " + fname + " ..." )
        sys.stdout.flush()
        rfd("../"+fname)
        print( "Computing " + fname + " ..." )
        sys.stdout.flush()
        cvel()
        Tcalcud()
        ts[findex]=t
        #HoverR
        diskcondition=bsq/rho<10
        hoverr3d,thetamid3d=horcalc(which=diskcondition)
        hoverr[findex]=hoverr3d.sum(2).sum(1)/(ny*nz)
        thetamid[findex]=thetamid3d.sum(2).sum(1)/(ny*nz)
        #rhosq:
        keywordsrhosq={'which': diskcondition}
        gdetint=intangle(gdet,**keywordsrhosq)
        rhosqint=intangle(gdet*rho**2,**keywordsrhosq)+tiny
        rhosqs[findex]=rhosqint
        maxrhosq2d=(rho**2*diskcondition).max(1)+tiny
        maxrhosq3d=np.empty_like(rho)
        for j in np.arange(0,ny):
            maxrhosq3d[:,j,:] = maxrhosq2d
        rhosrhosq[findex]=intangle(gdet*rho**2*rho,**keywordsrhosq)/rhosqint
        ugsrhosq[findex]=intangle(gdet*rho**2*ug,**keywordsrhosq)/rhosqint
        uu0rhosq[findex]=intangle(gdet*rho**2*uu[0],**keywordsrhosq)/rhosqint
        uus1rhosq[findex]=intangle(gdet*rho**2*uu[1],**keywordsrhosq)/rhosqint
        uuas1rhosq[findex]=intangle(gdet*rho**2*np.abs(uu[1]),**keywordsrhosq)/rhosqint
        uus3rhosq[findex]=intangle(gdet*rho**2*uu[3],**keywordsrhosq)/rhosqint
        uuas3rhosq[findex]=intangle(gdet*rho**2*np.abs(uu[3]),**keywordsrhosq)/rhosqint
        Bs1rhosq[findex]=intangle(gdetB[1]*rho**2,**keywordsrhosq)/rhosqint
        Bas1rhosq[findex]=intangle(np.abs(gdetB[1])*rho**2,**keywordsrhosq)/rhosqint
        Bs2rhosq[findex]=intangle(gdetB[2]*rho**2,**keywordsrhosq)/rhosqint
        Bas2rhosq[findex]=intangle(np.abs(gdetB[2])*rho**2,**keywordsrhosq)/rhosqint
        Bs3rhosq[findex]=intangle(gdetB[3]*rho**2,**keywordsrhosq)/rhosqint
        Bas3rhosq[findex]=intangle(np.abs(gdetB[3])*rho**2,**keywordsrhosq)/rhosqint
        #2h
        keywords2h={'hoverr': 2*horval, 'which': diskcondition}
        gdetint=intangle(gdet,**keywords2h)+tiny
        gdetint2h[findex]=gdetint
        rhos2h[findex]=intangle(gdet*rho,**keywords2h)/gdetint
        ugs2h[findex]=intangle(gdet*ug,**keywords2h)/gdetint
        uu02h[findex]=intangle(gdet*uu[0],**keywords2h)/gdetint
        uus12h[findex]=intangle(gdet*uu[1],**keywords2h)/gdetint
        uuas12h[findex]=intangle(gdet*np.abs(uu[1]),**keywords2h)/gdetint
        uus32h[findex]=intangle(gdet*uu[3],**keywords2h)/gdetint
        uuas32h[findex]=intangle(gdet*np.abs(uu[3]),**keywords2h)/gdetint
        Bs12h[findex]=intangle(gdetB[1],**keywords2h)/gdetint
        Bas12h[findex]=intangle(np.abs(gdetB[1]),**keywords2h)/gdetint
        Bs22h[findex]=intangle(gdetB[2],**keywords2h)/gdetint
        Bas22h[findex]=intangle(np.abs(gdetB[2]),**keywords2h)/gdetint
        Bs32h[findex]=intangle(gdetB[3],**keywords2h)/gdetint
        Bas32h[findex]=intangle(np.abs(gdetB[3]),**keywords2h)/gdetint
        #4h
        keywords4h={'hoverr': 4*horval, 'which': diskcondition}
        gdetint=intangle(gdet,**keywords4h)
        gdetint4h[findex]=gdetint+tiny
        rhos4h[findex]=intangle(gdet*rho,**keywords4h)/gdetint
        ugs4h[findex]=intangle(gdet*ug,**keywords4h)/gdetint
        uu04h[findex]=intangle(gdet*uu[0],**keywords4h)/gdetint
        uus14h[findex]=intangle(gdet*uu[1],**keywords4h)/gdetint
        uuas14h[findex]=intangle(gdet*np.abs(uu[1]),**keywords4h)/gdetint
        uus34h[findex]=intangle(gdet*uu[3],**keywords4h)/gdetint
        uuas34h[findex]=intangle(gdet*np.abs(uu[3]),**keywords4h)/gdetint
        Bs14h[findex]=intangle(gdetB[1],**keywords4h)/gdetint
        Bas14h[findex]=intangle(np.abs(gdetB[1]),**keywords4h)/gdetint
        Bs24h[findex]=intangle(gdetB[2],**keywords4h)/gdetint
        Bas24h[findex]=intangle(np.abs(gdetB[2]),**keywords4h)/gdetint
        Bs34h[findex]=intangle(gdetB[3],**keywords4h)/gdetint
        Bas34h[findex]=intangle(np.abs(gdetB[3]),**keywords4h)/gdetint
        #2hor
        keywords2hor={'hoverr': 2*hoverr3d, 'thetamid': thetamid3d, 'which': diskcondition}
        gdetint=intangle(gdet,**keywords2hor)
        gdetint2hor[findex]=gdetint+tiny
        rhos2hor[findex]=intangle(gdet*rho,**keywords2hor)/gdetint
        ugs2hor[findex]=intangle(gdet*ug,**keywords2hor)/gdetint
        bsqs2hor[findex]=intangle(bsq,**keywords2hor)/gdetint
        bsqorhos2hor[findex]=intangle(bsq/rho,**keywords2hor)/gdetint
        bsqougs2hor[findex]=intangle(bsq/ug,**keywords2hor)/gdetint
        uu02hor[findex]=intangle(gdet*uu[0],**keywords2hor)/gdetint
        uus12hor[findex]=intangle(gdet*uu[1],**keywords2hor)/gdetint
        uuas12hor[findex]=intangle(gdet*np.abs(uu[1]),**keywords2hor)/gdetint
        uus32hor[findex]=intangle(gdet*uu[3],**keywords2hor)/gdetint
        uuas32hor[findex]=intangle(gdet*np.abs(uu[3]),**keywords2hor)/gdetint
        Bs12hor[findex]=intangle(gdetB[1],**keywords2hor)/gdetint
        Bas12hor[findex]=intangle(np.abs(gdetB[1]),**keywords2hor)/gdetint
        Bs22hor[findex]=intangle(gdetB[2],**keywords2hor)/gdetint
        Bas22hor[findex]=intangle(np.abs(gdetB[2]),**keywords2hor)/gdetint
        Bs32hor[findex]=intangle(gdetB[3],**keywords2hor)/gdetint
        Bas32hor[findex]=intangle(np.abs(gdetB[3]),**keywords2hor)/gdetint
        #Flux
        fstot[findex]=horfluxcalc(minbsqorho=0)
        fs2hor[findex]==intangle(np.abs(gdetB[1]),**keywords2hor)
        fsj5[findex]=horfluxcalc(minbsqorho=5)
        fsj10[findex]=horfluxcalc(minbsqorho=10)
        fsj20[findex]=horfluxcalc(minbsqorho=20)
        fsj30[findex]=horfluxcalc(minbsqorho=30)
        fsj40[findex]=horfluxcalc(minbsqorho=40)
        #Mdot
        enth=1+ug*gam/rho
        mdtot[findex]=mdotcalc()
        mdtotbound[findex]=mdotcalc(which=(-enth*ud[0]<=1))
        md2h[findex]=mdotcalc(**keywords2h)
        md4h[findex]=mdotcalc(**keywords4h)
        md2hor[findex]=mdotcalc(**keywords2hor)
        md5[findex]=intangle(-gdet*rho*uu[1],minbsqorho=5)
        md10[findex]=intangle(-gdet*rho*uu[1],minbsqorho=10)
        md20[findex]=intangle(-gdet*rho*uu[1],minbsqorho=20)
        md30[findex]=intangle(-gdet*rho*uu[1],minbsqorho=30)
        md40[findex]=intangle(-gdet*rho*uu[1],minbsqorho=40)
        mdrhosq[findex]=scaletofullwedge(((-gdet*rho**2*rho*uu[1]*diskcondition).sum(1)/maxrhosq2d).sum(1)*_dx2*_dx3)
        #mdrhosq[findex]=(-gdet*rho**2*rho*uu[1]).sum(1).sum(1)/(-gdet*rho**2).sum(1).sum(1)*(-gdet).sum(1).sum(1)*_dx2*_dx3
        #Edot
        edtot[findex]=intangle(-gdet*Tud[1][0])
        edma[findex]=intangle(-gdet*TudMA[1][0])
        edtotbound[findex]=intangle(-gdet*Tud[1][0],which=(-enth*ud[0]<=1))
        edmabound[findex]=intangle(-gdet*TudMA[1][0],which=(-enth*ud[0]<=1))
        ed2h[findex]=intangle(-gdet*Tud[1][0],hoverr=2*horval)
        ed4h[findex]=intangle(-gdet*Tud[1][0],hoverr=4*horval)
        ed2hor[findex]=intangle(-gdet*Tud[1][0],hoverr=2*hoverr3d,thetamid=thetamid3d)
        edrhosq[findex]=scaletofullwedge(((-gdet*rho**2*Tud[1][0]).sum(1)/maxrhosq2d).sum(1)*_dx2*_dx3)
        #Pjet
        pjem5[findex]=jetpowcalc(0,minbsqorho=5)
        pjem10[findex]=jetpowcalc(0,minbsqorho=10)
        pjem20[findex]=jetpowcalc(0,minbsqorho=20)
        pjem30[findex]=jetpowcalc(0,minbsqorho=30)
        pjem40[findex]=jetpowcalc(0,minbsqorho=40)
        pjma5[findex]=jetpowcalc(1,minbsqorho=5)
        pjma10[findex]=jetpowcalc(1,minbsqorho=10)
        pjma20[findex]=jetpowcalc(1,minbsqorho=20)
        pjma30[findex]=jetpowcalc(1,minbsqorho=30)
        pjma40[findex]=jetpowcalc(1,minbsqorho=40)

        #new format?
        if qtymem.shape[0] > nqtyold:
            #yes!
            #north hemisphere
            pjem_n_mu10[findex]=jetpowcalc(0,minmu=10,donorthsouth=1)
            pjem_n_mu5[findex]=jetpowcalc(0,minmu=5,donorthsouth=1)
            pjem_n_mu2[findex]=jetpowcalc(0,minmu=2,donorthsouth=1)
            pjem_n_mu1[findex]=jetpowcalc(0,minmu=1,donorthsouth=1)
            pjrm_n_mu10[findex]=jetpowcalc(3,minmu=10,donorthsouth=1)
            pjrm_n_mu5[findex]=jetpowcalc(3,minmu=5,donorthsouth=1)
            pjrm_n_mu2[findex]=jetpowcalc(3,minmu=2,donorthsouth=1)
            pjrm_n_mu1[findex]=jetpowcalc(3,minmu=1,donorthsouth=1)
            pjma_n_mu10[findex]=jetpowcalc(1,minmu=10,donorthsouth=1)
            pjma_n_mu5[findex]=jetpowcalc(1,minmu=5,donorthsouth=1)
            pjma_n_mu2[findex]=jetpowcalc(1,minmu=2,donorthsouth=1)
            pjma_n_mu1[findex]=jetpowcalc(1,minmu=1,donorthsouth=1)
            phiabsj_n_mu10[findex]=jetpowcalc(4,minmu=10,donorthsouth=1)
            phiabsj_n_mu5[findex]=jetpowcalc(4,minmu=5,donorthsouth=1)
            phiabsj_n_mu2[findex]=jetpowcalc(4,minmu=2,donorthsouth=1)
            phiabsj_n_mu1[findex]=jetpowcalc(4,minmu=1,donorthsouth=1)
            #south hemisphere
            pjem_s_mu10[findex]=jetpowcalc(0,minmu=10,donorthsouth=-1)
            pjem_s_mu5[findex]=jetpowcalc(0,minmu=5,donorthsouth=-1)
            pjem_s_mu2[findex]=jetpowcalc(0,minmu=2,donorthsouth=-1)
            pjem_s_mu1[findex]=jetpowcalc(0,minmu=1,donorthsouth=-1)
            pjrm_s_mu10[findex]=jetpowcalc(3,minmu=10,donorthsouth=-1)
            pjrm_s_mu5[findex]=jetpowcalc(3,minmu=5,donorthsouth=-1)
            pjrm_s_mu2[findex]=jetpowcalc(3,minmu=2,donorthsouth=-1)
            pjrm_s_mu1[findex]=jetpowcalc(3,minmu=1,donorthsouth=-1)
            pjma_s_mu10[findex]=jetpowcalc(1,minmu=10,donorthsouth=-1)
            pjma_s_mu5[findex]=jetpowcalc(1,minmu=5,donorthsouth=-1)
            pjma_s_mu2[findex]=jetpowcalc(1,minmu=2,donorthsouth=-1)
            pjma_s_mu1[findex]=jetpowcalc(1,minmu=1,donorthsouth=-1)
            phiabsj_s_mu10[findex]=jetpowcalc(4,minmu=10,donorthsouth=-1)
            phiabsj_s_mu5[findex]=jetpowcalc(4,minmu=5,donorthsouth=-1)
            phiabsj_s_mu2[findex]=jetpowcalc(4,minmu=2,donorthsouth=-1)
            phiabsj_s_mu1[findex]=jetpowcalc(4,minmu=1,donorthsouth=-1)
            if ldtot is not None:
                ldtot[findex]=intangle(gdet*Tud[1][3])
            if gdetF10 is not None and gdetF is not None:
                print( "Assigning gdetF's in getqtyvstime()" )
                gdetF10[findex]=intangle(gdetF[1][0])
                gdetF11[findex]=intangle(gdetF[1][1])
                gdetF12[findex]=intangle(gdetF[1][2])
                gdetF20[findex]=intangle(gdetF[2][0])
                gdetF21[findex]=intangle(gdetF[2][1])
                gdetF22[findex]=intangle(gdetF[2][2])
                gdetF30[findex]=intangle(gdetF[3][0])
                gdetF31[findex]=intangle(gdetF[3][1])
                gdetF32[findex]=intangle(gdetF[3][2])
        #Bob's 1D quantities
        if dobob==1:
                dVF=_dx1*_dx2*_dx3
                dVA=_dx2*_dx3
                Dt=1
                TT=0
                RR=1
                TH=2
                PH=3
        	qtymem[i+0,findex]=intangle(Dt*dVF*gdet*rho,**keywords2hor)
        	qtymem[i+1,findex]=intangle(Dt*dVF*gdet*rho*rho,**keywords2hor)
        	qtymem[i+2,findex]=intangle(Dt*dVF*gdet*rho*ug,**keywords2hor)
        	qtymem[i+3,findex]=intangle(Dt*dVF*gdet*rho*bsq,**keywords2hor)
        
        	qtymem[i+4,findex]=intangle(Dt*dVF*gdet*rho*uu[1],**keywords2hor) #pr[2]
        	qtymem[i+5,findex]=intangle(Dt*dVF*gdet*rho*uu[2],**keywords2hor) #pr[3]
        	qtymem[i+6,findex]=intangle(Dt*dVF*gdet*rho*uu[3],**keywords2hor) #pr[4]
        
        	qtymem[i+7,findex]=intangle(Dt*dVF*gdet*rho*B[1],**keywords2hor) #pr[5]
        	qtymem[i+8,findex]=intangle(Dt*dVF*gdet*rho*B[2],**keywords2hor) #pr[6]
        	qtymem[i+9,findex]=intangle(Dt*dVF*gdet*rho*B[3],**keywords2hor) #pr[7]
        
        	#rho * u * u
        
        	qtymem[i+10,findex]=intangle(Dt*dVA*gdet*rho*(ud[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+11,findex]=intangle(Dt*dVA*gdet*rho*(ud[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+12,findex]=intangle(Dt*dVA*gdet*rho*(ud[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+13,findex]=intangle(Dt*dVA*gdet*rho*(ud[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+14,findex]=intangle(Dt*dVA*gdet*rho*(ud[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+15,findex]=intangle(Dt*dVA*gdet*rho*(ud[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+16,findex]=intangle(Dt*dVA*gdet*rho*(ud[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+17,findex]=intangle(Dt*dVA*gdet*rho*(ud[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+18,findex]=intangle(Dt*dVA*gdet*rho*(ud[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+19,findex]=intangle(Dt*dVA*gdet*rho*(ud[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+20,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+21,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+22,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+23,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+24,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+25,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+26,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+27,findex]=intangle(Dt*dVA*gdet*rho*(uu[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+28,findex]=intangle(Dt*dVA*gdet*rho*(uu[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+29,findex]=intangle(Dt*dVA*gdet*rho*(uu[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+30,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(uu[TT]),**keywords2hor)
        	qtymem[i+31,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(uu[RR]),**keywords2hor)
        	qtymem[i+32,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(uu[TH]),**keywords2hor)
        	qtymem[i+33,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT])*(uu[PH]),**keywords2hor)
        	qtymem[i+34,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(uu[RR]),**keywords2hor)
        	qtymem[i+35,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(uu[TH]),**keywords2hor)
        	qtymem[i+36,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR])*(uu[PH]),**keywords2hor)
        	qtymem[i+37,findex]=intangle(Dt*dVA*gdet*rho*(uu[TH])*(uu[TH]),**keywords2hor)
        	qtymem[i+38,findex]=intangle(Dt*dVA*gdet*rho*(uu[TH])*(uu[PH]),**keywords2hor)
        	qtymem[i+39,findex]=intangle(Dt*dVA*gdet*rho*(uu[PH])*(uu[PH]),**keywords2hor)
        
        
        	#UU * u * u
        
        	qtymem[i+40,findex]=intangle(Dt*dVA*gdet*ug*(ud[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+41,findex]=intangle(Dt*dVA*gdet*ug*(ud[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+42,findex]=intangle(Dt*dVA*gdet*ug*(ud[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+43,findex]=intangle(Dt*dVA*gdet*ug*(ud[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+44,findex]=intangle(Dt*dVA*gdet*ug*(ud[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+45,findex]=intangle(Dt*dVA*gdet*ug*(ud[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+46,findex]=intangle(Dt*dVA*gdet*ug*(ud[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+47,findex]=intangle(Dt*dVA*gdet*ug*(ud[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+48,findex]=intangle(Dt*dVA*gdet*ug*(ud[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+49,findex]=intangle(Dt*dVA*gdet*ug*(ud[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+50,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+51,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+52,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+53,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+54,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+55,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+56,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+57,findex]=intangle(Dt*dVA*gdet*ug*(uu[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+58,findex]=intangle(Dt*dVA*gdet*ug*(uu[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+59,findex]=intangle(Dt*dVA*gdet*ug*(uu[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+60,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(uu[TT]),**keywords2hor)
        	qtymem[i+61,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(uu[RR]),**keywords2hor)
        	qtymem[i+62,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(uu[TH]),**keywords2hor)
        	qtymem[i+63,findex]=intangle(Dt*dVA*gdet*ug*(uu[TT])*(uu[PH]),**keywords2hor)
        	qtymem[i+64,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(uu[RR]),**keywords2hor)
        	qtymem[i+65,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(uu[TH]),**keywords2hor)
        	qtymem[i+66,findex]=intangle(Dt*dVA*gdet*ug*(uu[RR])*(uu[PH]),**keywords2hor)
        	qtymem[i+67,findex]=intangle(Dt*dVA*gdet*ug*(uu[TH])*(uu[TH]),**keywords2hor)
        	qtymem[i+68,findex]=intangle(Dt*dVA*gdet*ug*(uu[TH])*(uu[PH]),**keywords2hor)
        	qtymem[i+69,findex]=intangle(Dt*dVA*gdet*ug*(uu[PH])*(uu[PH]),**keywords2hor)
        
        	#bsq * u * u
        
        	qtymem[i+70,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+71,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+72,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+73,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+74,findex]=intangle(Dt*dVA*gdet*bsq*(ud[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+75,findex]=intangle(Dt*dVA*gdet*bsq*(ud[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+76,findex]=intangle(Dt*dVA*gdet*bsq*(ud[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+77,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+78,findex]=intangle(Dt*dVA*gdet*bsq*(ud[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+79,findex]=intangle(Dt*dVA*gdet*bsq*(ud[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+80,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(ud[TT]),**keywords2hor)
        	qtymem[i+81,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(ud[RR]),**keywords2hor)
        	qtymem[i+82,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(ud[TH]),**keywords2hor)
        	qtymem[i+83,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(ud[PH]),**keywords2hor)
        	qtymem[i+84,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(ud[RR]),**keywords2hor)
        	qtymem[i+85,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(ud[TH]),**keywords2hor)
        	qtymem[i+86,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(ud[PH]),**keywords2hor)
        	qtymem[i+87,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TH])*(ud[TH]),**keywords2hor)
        	qtymem[i+88,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TH])*(ud[PH]),**keywords2hor)
        	qtymem[i+89,findex]=intangle(Dt*dVA*gdet*bsq*(uu[PH])*(ud[PH]),**keywords2hor)
        
        	qtymem[i+90,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(uu[TT]),**keywords2hor)
        	qtymem[i+91,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(uu[RR]),**keywords2hor)
        	qtymem[i+92,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(uu[TH]),**keywords2hor)
        	qtymem[i+93,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TT])*(uu[PH]),**keywords2hor)
        	qtymem[i+94,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(uu[RR]),**keywords2hor)
        	qtymem[i+95,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(uu[TH]),**keywords2hor)
        	qtymem[i+96,findex]=intangle(Dt*dVA*gdet*bsq*(uu[RR])*(uu[PH]),**keywords2hor)
        	qtymem[i+97,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TH])*(uu[TH]),**keywords2hor)
        	qtymem[i+98,findex]=intangle(Dt*dVA*gdet*bsq*(uu[TH])*(uu[PH]),**keywords2hor)
        	qtymem[i+99,findex]=intangle(Dt*dVA*gdet*bsq*(uu[PH])*(uu[PH]),**keywords2hor)
        
        	# b * b
        
        	qtymem[i+100,findex]=intangle(Dt*dVA*gdet*(bd[TT])*(bd[TT]),**keywords2hor)
        	qtymem[i+101,findex]=intangle(Dt*dVA*gdet*(bd[TT])*(bd[RR]),**keywords2hor)
        	qtymem[i+102,findex]=intangle(Dt*dVA*gdet*(bd[TT])*(bd[TH]),**keywords2hor)
        	qtymem[i+103,findex]=intangle(Dt*dVA*gdet*(bd[TT])*(bd[PH]),**keywords2hor)
        	qtymem[i+104,findex]=intangle(Dt*dVA*gdet*(bd[RR])*(bd[RR]),**keywords2hor)
        	qtymem[i+105,findex]=intangle(Dt*dVA*gdet*(bd[RR])*(bd[TH]),**keywords2hor)
        	qtymem[i+106,findex]=intangle(Dt*dVA*gdet*(bd[RR])*(bd[PH]),**keywords2hor)
        	qtymem[i+107,findex]=intangle(Dt*dVA*gdet*(bd[TH])*(bd[TH]),**keywords2hor)
        	qtymem[i+108,findex]=intangle(Dt*dVA*gdet*(bd[TH])*(bd[PH]),**keywords2hor)
        	qtymem[i+109,findex]=intangle(Dt*dVA*gdet*(bd[PH])*(bd[PH]),**keywords2hor)
        
        	qtymem[i+110,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bd[TT]),**keywords2hor)
        	qtymem[i+111,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bd[RR]),**keywords2hor)
        	qtymem[i+112,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bd[TH]),**keywords2hor)
        	qtymem[i+113,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bd[PH]),**keywords2hor)
        	qtymem[i+114,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bd[RR]),**keywords2hor)
        	qtymem[i+115,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bd[TH]),**keywords2hor)
        	qtymem[i+116,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bd[PH]),**keywords2hor)
        	qtymem[i+117,findex]=intangle(Dt*dVA*gdet*(bu[TH])*(bd[TH]),**keywords2hor)
        	qtymem[i+118,findex]=intangle(Dt*dVA*gdet*(bu[TH])*(bd[PH]),**keywords2hor)
        	qtymem[i+119,findex]=intangle(Dt*dVA*gdet*(bu[PH])*(bd[PH]),**keywords2hor)
        
        	qtymem[i+120,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bu[TT]),**keywords2hor)
        	qtymem[i+121,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bu[RR]),**keywords2hor)
        	qtymem[i+122,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bu[TH]),**keywords2hor)
        	qtymem[i+123,findex]=intangle(Dt*dVA*gdet*(bu[TT])*(bu[PH]),**keywords2hor)
        	qtymem[i+124,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bu[RR]),**keywords2hor)
        	qtymem[i+125,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bu[TH]),**keywords2hor)
        	qtymem[i+126,findex]=intangle(Dt*dVA*gdet*(bu[RR])*(bu[PH]),**keywords2hor)
        	qtymem[i+127,findex]=intangle(Dt*dVA*gdet*(bu[TH])*(bu[TH]),**keywords2hor)
        	qtymem[i+128,findex]=intangle(Dt*dVA*gdet*(bu[TH])*(bu[PH]),**keywords2hor)
        	qtymem[i+129,findex]=intangle(Dt*dVA*gdet*(bu[PH])*(bu[PH]),**keywords2hor)
        
        
        	#mass flux
        	qtymem[i+130,findex]=intangle(Dt*dVA*gdet*rho*(uu[TT]),**keywords2hor)
        	qtymem[i+131,findex]=intangle(Dt*dVA*gdet*rho*(uu[RR]),**keywords2hor)
        	qtymem[i+132,findex]=intangle(Dt*dVA*gdet*rho*(uu[TH]),**keywords2hor)
        	qtymem[i+133,findex]=intangle(Dt*dVA*gdet*rho*(uu[PH]),**keywords2hor)
        #END BOB's QUANTITIES
        #if os.path.isfile("lrho%04d.png" % findex):
        #    print( "Skipping " + fname + " as lrho%04d.png exists" % findex );
        #else:
        #    print( "Reinterpolating " + fname + " ..." )
        #    plt.figure(0)
        #    plt.clf()
        #    mkframe("lrho%04d" % findex, vmin=-8,vmax=0.2)
    print( "Saving to file..." )
    if( whichi >=0 and whichn > 0 ):
        np.save( "qty2_%d_%d.npy" % (whichi, whichn), qtymem )
    else:
        np.save( "qty2.npy", qtymem )
    print( "Done!" )
    return(qtymem)

def fhorvstime(ihor):
    """
    Returns a tuple (ts,fs,mdot): lists of times, horizon fluxes, and Mdot
    """
    flist = glob.glob( os.path.join("dumps/", "fieldline*.bin") )
    ts=np.empty(len(flist),dtype=np.float32)
    fs=np.empty(len(flist),dtype=np.float32)
    md=np.empty(len(flist),dtype=np.float32)
    for findex, fname in enumerate(flist):
        print( "Reading " + fname + " ..." )
        rfd("../"+fname)
        fs[findex]=horfluxcalc(ihor)
        md[findex]=mdotcalc(ihor)
        ts[findex]=t
    print( "Done!" )
    return((ts,fs,md))

def amax(arg1,arg2):
    arr1 = np.array(arg1)
    arr2 = np.array(arg2)
    ret=np.zeros_like(arr1)
    ret[arr1>=arr2]=arr1[arr1>=arr2]
    ret[arr2>arr1]=arr2[arr2>arr1]
    return(ret)
def amin(arg1,arg2):
    arr1 = np.array(arg1)
    arr2 = np.array(arg2)
    ret=np.zeros_like(arr1)
    ret[arr1<=arr2]=arr1[arr1<=arr2]
    ret[arr2<arr1]=arr2[arr2<arr1]
    return(ret)

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

def faraday():
    global fdd, fuu, omegaf1, omegaf2
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


def jetpowcalc(which=2,minbsqorho=10,minmu=None,donorthsouth=0):
    if which==0:
        jetpowden = -gdet*TudEM[1,0]
    if which==1:
        jetpowden = -gdet*TudMA[1,0]
    if which==2:
        jetpowden = -gdet*Tud[1,0]
    if which==3:
        #rest-mass flux
        jetpowden = gdet*rho*uu[1]
    if which==4:
        #phi (mag. flux)
        jetpowden = np.abs(gdetB[1])
    #jetpowden[tj>=ny-2] = 0*jetpowden[tj>=ny-2]
    #jetpowden[tj<1] = 0*jetpowden[tj<1]
    if minmu is None:
        jetpowden[bsq/rho<minbsqorho] = 0*jetpowden[bsq/rho<minbsqorho]
    else:
        #zero out outside jet (cut out low magnetization region)
        cond=(mu<minmu)
        #zero out bound region
        cond+=(1-isunbound)
        #zero out infalling region
        cond+=(uu[1]<=0.0)
        # 1 = north
        #-1 = south
        if donorthsouth==1:
            #NORTH
            #[zero out south hemisphere]
            cond += (tj>=ny/2)
        elif donorthsouth==-1:
            #SOUTH
            #[zero out north hemisphere]
            cond += (tj<ny/2)
        jetpowden[cond] = 0*jetpowden[cond]
    jetpowtot = scaletofullwedge(np.sum(np.sum(jetpowden,axis=2),axis=1)*_dx2*_dx3)
    #print "which = %d, minbsqorho = %g" % (which, minbsqorho)
    return(jetpowtot)
    
def plotit(ts,fs,md):
    #rc('font', family='serif')
    #plt.figure( figsize=(12,9) )
    fig,plotlist=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,9))
    #plt.subplots_adjust(hspace=0.4) #increase vertical spacing to avoid crowding
    plotlist[0].plot(ts,fs,label=r'$\Phi_{\rm h}/\Phi_{\rm i}$: Normalized Horizon Magnetic Flux')
    plotlist[0].plot(ts,fs,'r+') #, label=r'$\Phi_{\rm h}/0.5\Phi_{\rm i}$: Data Points')
    plotlist[0].legend(loc='lower right')
    #plt.xlabel(r'$t\;(GM/c^3)$')
    plotlist[0].set_ylabel(r'$\Phi_{\rm h}$',fontsize=16)
    plt.setp( plotlist[0].get_xticklabels(), visible=False)
    plotlist[0].grid(True)
    #
    #plotlist[1].subplot(212,sharex=True)
    plotlist[1].plot(ts,md,label=r'$\dot M_{\rm h}$: Horizon Accretion Rate')
    plotlist[1].plot(ts,md,'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
    plotlist[1].legend(loc='upper right')
    plotlist[1].set_xlabel(r'$t\;(GM/c^3)$')
    plotlist[1].set_ylabel(r'$\dot M_{\rm h}$',fontsize=16)
    
    #title("\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
    #      fontsize=16, color='r')
    plotlist[1].grid(True)
    fig.savefig('test.pdf')

def iofr(rval):
    res = interp1d(r[:,0,0], ti[:,0,0], kind='linear')
    return(np.floor(res(rval)+0.5))

def plotqtyvstime(qtymem,ihor=None,whichplot=None,ax=None,findex=None,fti=None,ftf=None,showextra=False,prefactor=100,epsFm=None,epsFke=None,epsetaj=None,epsFm30=None,sigma=None, usegaussianunits=False, aphi_j_val=0,showextraeta=False,plotFM30=False):
    global mdotfinavgvsr, mdotfinavgvsr5, mdotfinavgvsr10,mdotfinavgvsr20, mdotfinavgvsr30,mdotfinavgvsr40
    if ihor is None:
        ihor = iofr(rhor)
    nqtyold=98
    nqty=98+32+1
    ###############################
    #copy this from getqtyvstime()
    ###############################
    if qtymem.shape[0] == 71:
        print "Plot using old fmt"
        #old fmt
        i=0
        ts=qtymem[i,:,0];i+=1
        #HoverR
        hoverr=qtymem[i];i+=1
        thetamid=qtymem[i];i+=1
        #rhosq:
        rhosqs=qtymem[i];i+=1
        rhosrhosq=qtymem[i];i+=1
        ugsrhosq=qtymem[i];i+=1
        uu0rhosq=qtymem[i];i+=1
        uus1rhosq=qtymem[i];i+=1
        uuas1rhosq=qtymem[i];i+=1
        uus3rhosq=qtymem[i];i+=1
        uuas3rhosq=qtymem[i];i+=1
        Bs1rhosq=qtymem[i];i+=1
        Bas1rhosq=qtymem[i];i+=1
        Bs3rhosq=qtymem[i];i+=1
        Bas3rhosq=qtymem[i];i+=1
        #2h
        rhos2h=qtymem[i];i+=1
        ugs2h=qtymem[i];i+=1
        uu02h=qtymem[i];i+=1
        uus12h=qtymem[i];i+=1
        uuas12h=qtymem[i];i+=1
        uus32h=qtymem[i];i+=1
        uuas32h=qtymem[i];i+=1
        Bs12h=qtymem[i];i+=1
        Bas12h=qtymem[i];i+=1
        Bs32h=qtymem[i];i+=1
        Bas32h=qtymem[i];i+=1
        #4h
        rhos4h=qtymem[i];i+=1
        ugs4h=qtymem[i];i+=1
        uu04h=qtymem[i];i+=1
        uus14h=qtymem[i];i+=1
        uuas14h=qtymem[i];i+=1
        uus34h=qtymem[i];i+=1
        uuas34h=qtymem[i];i+=1
        Bs14h=qtymem[i];i+=1
        Bas14h=qtymem[i];i+=1
        Bs34h=qtymem[i];i+=1
        Bas34h=qtymem[i];i+=1
        #2hor
        rhos2hor=qtymem[i];i+=1
        ugs2hor=qtymem[i];i+=1
        uu02hor=qtymem[i];i+=1
        uus12hor=qtymem[i];i+=1
        uuas12hor=qtymem[i];i+=1
        uus32hor=qtymem[i];i+=1
        uuas32hor=qtymem[i];i+=1
        Bs12hor=qtymem[i];i+=1
        Bas12hor=qtymem[i];i+=1
        Bs32hor=qtymem[i];i+=1
        Bas32hor=qtymem[i];i+=1
        #Flux
        fstot=qtymem[i]*2;i+=1
        fsj5=qtymem[i]*2;i+=1
        fsj10=qtymem[i]*2;i+=1
        #Mdot
        mdtot=qtymem[i]*2;i+=1
        md2h=qtymem[i]*2;i+=1
        md4h=qtymem[i]*2;i+=1
        md2hor=qtymem[i]*2;i+=1
        md5=qtymem[i]*2;i+=1
        md10=qtymem[i]*2;i+=1
        mdrhosq=qtymem[i]*2;i+=1
        mdtotbound=qtymem[i]*2;i+=1
        #Edot
        edtot=qtymem[i]*2;i+=1
        ed2h=qtymem[i]*2;i+=1
        ed4h=qtymem[i]*2;i+=1
        ed2hor=qtymem[i]*2;i+=1
        edrhosq=qtymem[i]*2;i+=1
        edma=qtymem[i]*2;i+=1
        edtotbound=qtymem[i]*2;i+=1
        edmabound=qtymem[i]*2;i+=1
        #Pjet
        pjem5=qtymem[i];i+=1
        pjem10=qtymem[i];i+=1
        pjma5=qtymem[i];i+=1
        pjma10=qtymem[i];i+=1
    else:
        #new fmt
        #qty defs
        i=0
        ts=qtymem[i,:,0];i+=1
        #HoverR
        hoverr=qtymem[i];i+=1
        thetamid=qtymem[i];i+=1
        #rhosq:
        rhosqs=qtymem[i];i+=1
        rhosrhosq=qtymem[i];i+=1
        ugsrhosq=qtymem[i];i+=1
        uu0rhosq=qtymem[i];i+=1
        uus1rhosq=qtymem[i];i+=1
        uuas1rhosq=qtymem[i];i+=1
        uus3rhosq=qtymem[i];i+=1
        uuas3rhosq=qtymem[i];i+=1
        Bs1rhosq=qtymem[i];i+=1
        Bas1rhosq=qtymem[i];i+=1
        Bs2rhosq=qtymem[i];i+=1
        Bas2rhosq=qtymem[i];i+=1
        Bs3rhosq=qtymem[i];i+=1
        Bas3rhosq=qtymem[i];i+=1
        #2h
        gdetint2h=qtymem[i];i+=1
        rhos2h=qtymem[i];i+=1
        ugs2h=qtymem[i];i+=1
        uu02h=qtymem[i];i+=1
        uus12h=qtymem[i];i+=1
        uuas12h=qtymem[i];i+=1
        uus32h=qtymem[i];i+=1
        uuas32h=qtymem[i];i+=1
        Bs12h=qtymem[i];i+=1
        Bas12h=qtymem[i];i+=1
        Bs22h=qtymem[i];i+=1
        Bas22h=qtymem[i];i+=1
        Bs32h=qtymem[i];i+=1
        Bas32h=qtymem[i];i+=1
        #4h
        gdetint4h=qtymem[i];i+=1
        rhos4h=qtymem[i];i+=1
        ugs4h=qtymem[i];i+=1
        uu04h=qtymem[i];i+=1
        uus14h=qtymem[i];i+=1
        uuas14h=qtymem[i];i+=1
        uus34h=qtymem[i];i+=1
        uuas34h=qtymem[i];i+=1
        Bs14h=qtymem[i];i+=1
        Bas14h=qtymem[i];i+=1
        Bs24h=qtymem[i];i+=1
        Bas24h=qtymem[i];i+=1
        Bs34h=qtymem[i];i+=1
        Bas34h=qtymem[i];i+=1
        #2hor
        gdetint2hor=qtymem[i];i+=1
        rhos2hor=qtymem[i];i+=1
        ugs2hor=qtymem[i];i+=1
        bsqs2hor=qtymem[i];i+=1
        bsqorhos2hor=qtymem[i];i+=1
        bsqougs2hor=qtymem[i];i+=1
        uu02hor=qtymem[i];i+=1
        uus12hor=qtymem[i];i+=1
        uuas12hor=qtymem[i];i+=1
        uus32hor=qtymem[i];i+=1
        uuas32hor=qtymem[i];i+=1
        Bs12hor=qtymem[i];i+=1
        Bas12hor=qtymem[i];i+=1
        Bs22hor=qtymem[i];i+=1
        Bas22hor=qtymem[i];i+=1
        Bs32hor=qtymem[i];i+=1
        Bas32hor=qtymem[i];i+=1
        #Flux
        fstot=qtymem[i];i+=1
        fs2hor=qtymem[i];i+=1
        fsj5=qtymem[i];i+=1
        fsj10=qtymem[i];i+=1
        fsj20=qtymem[i];i+=1
        fsj30=qtymem[i];i+=1
        fsj40=qtymem[i];i+=1
        #Mdot
        mdtot=qtymem[i];i+=1
        md2h=qtymem[i];i+=1
        md4h=qtymem[i];i+=1
        md2hor=qtymem[i];i+=1
        md5=qtymem[i];i+=1
        md10=qtymem[i];i+=1
        md20=qtymem[i];i+=1
        md30=qtymem[i];i+=1
        md40=qtymem[i];i+=1
        mdrhosq=qtymem[i];i+=1
        mdtotbound=qtymem[i];i+=1
        #Edot
        edtot=qtymem[i];i+=1
        ed2h=qtymem[i];i+=1
        ed4h=qtymem[i];i+=1
        ed2hor=qtymem[i];i+=1
        edrhosq=qtymem[i];i+=1
        edma=qtymem[i];i+=1
        edtotbound=qtymem[i];i+=1
        edmabound=qtymem[i];i+=1
        #Pjet
        pjem5=qtymem[i];i+=1
        pjem10=qtymem[i];i+=1
        pjem20=qtymem[i];i+=1
        pjem30=qtymem[i];i+=1
        pjem40=qtymem[i];i+=1
        pjma5=qtymem[i];i+=1
        pjma10=qtymem[i];i+=1
        pjma20=qtymem[i];i+=1
        pjma30=qtymem[i];i+=1
        pjma40=qtymem[i];i+=1
        #new format?
        if qtymem.shape[0] > nqtyold:
            #yes!
            pjem_n_mu10=qtymem[i];i+=1
            pjem_n_mu5=qtymem[i];i+=1
            pjem_n_mu2=qtymem[i];i+=1
            pjem_n_mu1=qtymem[i];i+=1
            pjrm_n_mu10=qtymem[i];i+=1
            pjrm_n_mu5=qtymem[i];i+=1
            pjrm_n_mu2=qtymem[i];i+=1
            pjrm_n_mu1=qtymem[i];i+=1
            pjma_n_mu10=qtymem[i];i+=1
            pjma_n_mu5=qtymem[i];i+=1
            pjma_n_mu2=qtymem[i];i+=1
            pjma_n_mu1=qtymem[i];i+=1
            phiabsj_n_mu10=qtymem[i];i+=1
            phiabsj_n_mu5=qtymem[i];i+=1
            phiabsj_n_mu2=qtymem[i];i+=1
            phiabsj_n_mu1=qtymem[i];i+=1
            pjem_s_mu10=qtymem[i];i+=1
            pjem_s_mu5=qtymem[i];i+=1
            pjem_s_mu2=qtymem[i];i+=1
            pjem_s_mu1=qtymem[i];i+=1
            pjrm_s_mu10=qtymem[i];i+=1
            pjrm_s_mu5=qtymem[i];i+=1
            pjrm_s_mu2=qtymem[i];i+=1
            pjrm_s_mu1=qtymem[i];i+=1
            pjma_s_mu10=qtymem[i];i+=1
            pjma_s_mu5=qtymem[i];i+=1
            pjma_s_mu2=qtymem[i];i+=1
            pjma_s_mu1=qtymem[i];i+=1
            phiabsj_s_mu10=qtymem[i];i+=1
            phiabsj_s_mu5=qtymem[i];i+=1
            phiabsj_s_mu2=qtymem[i];i+=1
            phiabsj_s_mu1=qtymem[i];i+=1
            if i < qtymem.shape[0]:
                ldtot=qtymem[i];i+=1
            else:
                print( "Oldish format: missing ldtot" )
                ldtot=None
            #derived
            pjke_n_mu2 = pjem_n_mu2 + pjma_n_mu2 - pjrm_n_mu2
            pjke_s_mu2 = pjem_s_mu2 + pjma_s_mu2 - pjrm_s_mu2
            pjke_mu2 = pjke_n_mu2 + pjke_s_mu2
            pjke_n_mu1 = pjem_n_mu1 + pjma_n_mu1 - pjrm_n_mu1
            pjke_s_mu1 = pjem_s_mu1 + pjma_s_mu1 - pjrm_s_mu1
            pjke_mu1 = pjke_n_mu1 + pjke_s_mu1
            phiabsj_mu2 = phiabsj_n_mu2 + phiabsj_s_mu2
            phiabsj_mu1 = phiabsj_n_mu1 + phiabsj_s_mu1
            if i < qtymem.shape[0]:
                print( "Assigning gdetF's in plotqtyvstime()" )
                gdetF10=qtymem[i];i+=1
                gdetF11=qtymem[i];i+=1
                gdetF12=qtymem[i];i+=1
                gdetF20=qtymem[i];i+=1
                gdetF21=qtymem[i];i+=1
                gdetF22=qtymem[i];i+=1
                gdetF30=qtymem[i];i+=1
                gdetF31=qtymem[i];i+=1
                gdetF32=qtymem[i];i+=1
            else:
                print( "Oldish format: missing gdetF1,2,3" )
                gdetF10=np.zeros_like(edtot)
                gdetF11=gdetF10
                gdetF12=gdetF10
                gdetF20=gdetF10
                gdetF21=gdetF10
                gdetF22=gdetF10
                gdetF30=gdetF10
                gdetF31=gdetF10
                gdetF32=gdetF10
        else:
            print( "Oldish format: missing north/south jet power and flux" )
            sys.stdout.flush()
            pjem_n_mu10=None
            pjem_n_mu5=None
            pjem_n_mu2=None
            pjem_n_mu1=None
            pjrm_n_mu10=None
            pjrm_n_mu5=None
            pjrm_n_mu2=None
            pjrm_n_mu1=None
            pjma_n_mu10=None
            pjma_n_mu5=None
            pjma_n_mu2=None
            pjma_n_mu1=None
            phiabsj_n_mu10=None
            phiabsj_n_mu5=None
            phiabsj_n_mu2=None
            phiabsj_n_mu1=None
            pjem_s_mu10=None
            pjem_s_mu5=None
            pjem_s_mu2=None
            pjem_s_mu1=None
            pjrm_s_mu10=None
            pjrm_s_mu5=None
            pjrm_s_mu2=None
            pjrm_s_mu1=None
            pjma_s_mu10=None
            pjma_s_mu5=None
            pjma_s_mu2=None
            pjma_s_mu1=None
            phiabsj_s_mu10=None
            phiabsj_s_mu5=None
            phiabsj_s_mu2=None
            phiabsj_s_mu1=None
            ldtot=None
            #derived
            pjke_n_mu2=None
            pjke_s_mu2=None
            pjke_mu2=None
            pjke_n_mu1=None
            pjke_s_mu1=None
            pjke_mu1=None
            phiabsj_mu2=None
            phiabsj_mu1=None
            gdetF10=np.zeros_like(edtot)
            gdetF11=gdetF10
            gdetF12=gdetF10
            gdetF20=gdetF10
            gdetF21=gdetF10
            gdetF22=gdetF10
            gdetF30=gdetF10
            gdetF31=gdetF10
            gdetF32=gdetF10
            ldtot=gdetF10
    #end qty defs
    ##############################
    #end copy
    ##############################
    #
    #rc('font', family='serif')
    #plt.figure( figsize=(12,9) )
    if ftf is not None:
        iti = 1000 #dummy
        itf = 2000 #dummy
        if ftf > ts[-1]:
            ftf = ts[-1]
        print fti, ftf
        dotavg=1
    elif os.path.isfile(os.path.join("titf.txt")):
        dotavg=1
        gd1 = np.loadtxt( "titf.txt",
                          dtype=np.float64, 
                          skiprows=1, 
                          unpack = True )
        iti = gd1[0]
        itf = gd1[1]
        fti = gd1[2]
        ftf = gd1[3]
    else:
        print( "Warning: titf.txt not found: using default numbers for averaging" )
        dotavg=1
        iti = 3000
        itf = 4000
        fti = 8000
        ftf = 1e5

    #mdotiniavg = timeavg(mdtot[:,ihor]-md10[:,ihor],ts,fti,ftf)
    #mdotfinavg = (mdtot[:,ihor]-md10[:,ihor])[(ts<ftf)*(ts>=fti)].sum()/(mdtot[:,ihor]-md10[:,ihor])[(ts<ftf)*(ts>=fti)].shape[0]
    if True and (gdetF11!=0).any() and timeavg(gdetF11,ts,fti,fti+1.).any():
        print( "plotqtyvstime(): using gdetF11 and gdetF10 to compute edot and mdot" )
        #avoid changing originals by making copy
        edtot=np.copy(edtot)
        mdtot=np.copy(mdtot)
        edtot[:-1] = -0.5*(gdetF11[:-1]+gdetF11[1:])
        mdtot[:-1] = -0.5*(gdetF10[:-1]+gdetF10[1:])
        edtot -= mdtot

    mdotiniavgvsr = timeavg(mdtot,ts,iti,itf)
    mdotfinavgvsr = timeavg(mdtot,ts,fti,ftf)
    # full (disk + jet) accretion rate
    mdtotvsr = mdotfinavgvsr
    edtotvsr = timeavg(edtot,ts,fti,ftf)
    edmavsr = timeavg(edma,ts,fti,ftf)
    if ldtot is not None:
        ldtotvsr = timeavg(ldtot,ts,fti,ftf)
    else:
        ldtotvsr = None
    #########
    mdotfinavgvsr5 = timeavg(mdtot[:,:]-md5[:,:],ts,fti,ftf)
    mdotfinavgvsr10 = timeavg(mdtot[:,:]-md10[:,:],ts,fti,ftf)
    mdotfinavgvsr20 = timeavg(mdtot[:,:]-md20[:,:],ts,fti,ftf)
    mdotfinavgvsr30 = timeavg(mdtot[:,:]-md30[:,:],ts,fti,ftf)
    mdotiniavgvsr30 = timeavg(mdtot[:,:]-md30[:,:],ts,iti,itf)
    mdotfinavgvsr40 = timeavg(mdtot[:,:]-md40[:,:],ts,fti,ftf)
    mdotiniavg = np.float64(mdotiniavgvsr30)[r[:,0,0]<10].mean()
    mdotfinavg = np.float64(mdotfinavgvsr30)[r[:,0,0]<10].mean()
    pjetiniavg = timeavg(pjem30[:,ihor],ts,iti,itf)
    pjetfinavg = timeavg(pjem30[:,ihor],ts,fti,ftf)
    pjemfinavgvsr = timeavg(edtot-edma,ts,fti,ftf)
    pjemtot = edtot-edma
    pjemfinavgvsr5 = timeavg(pjem5[:,:],ts,fti,ftf)
    pjemfinavgvsr10 = timeavg(pjem10[:,:],ts,fti,ftf)
    pjemfinavgvsr20 = timeavg(pjem20[:,:],ts,fti,ftf)
    pjemfinavgvsr30 = timeavg(pjem30[:,:],ts,fti,ftf)
    pjemfinavgvsr40 = timeavg(pjem40[:,:],ts,fti,ftf)
    pjemfinavgtot = timeavg((edtot-edma)[:,ihor],ts,fti,ftf)
    pjmafinavgvsr = timeavg(edma[:,:],ts,fti,ftf)
    pjmafinavgvsr5 = timeavg(pjma5[:,:],ts,fti,ftf)
    pjmafinavgvsr10 = timeavg(pjma10[:,:],ts,fti,ftf)
    pjmafinavgvsr20 = timeavg(pjma20[:,:],ts,fti,ftf)
    pjmafinavgvsr30 = timeavg(pjma30[:,:],ts,fti,ftf)
    pjmafinavgvsr40 = timeavg(pjma40[:,:],ts,fti,ftf)
    pjtotfinavgvsr = pjemfinavgvsr + pjmafinavgvsr
    pjtotfinavgvsr5 = pjemfinavgvsr5 + pjmafinavgvsr5
    pjtotfinavgvsr10 = pjemfinavgvsr10 + pjmafinavgvsr10
    pjtotfinavgvsr20 = pjemfinavgvsr20 + pjmafinavgvsr20
    pjtotfinavgvsr30 = pjemfinavgvsr30 + pjmafinavgvsr30
    pjtotfinavgvsr40 = pjemfinavgvsr40 + pjmafinavgvsr40

    if pjke_mu1 is not None:
        pjke_mu1_avg = timeavg(pjke_mu1,ts,fti,ftf)
    else:
        pjke_mu1_avg = -1e11+mdtotvsr*0
    if pjke_mu2 is not None:
        pjke_mu2_avg = timeavg(pjke_mu2,ts,fti,ftf)
    else:
        pjke_mu2_avg = -1e11+mdtotvsr*0

    #radius of stagnation point (Pjmabsqorho5(rstag) = 0)
    indices=ti[:,0,0][pjmafinavgvsr5>0]
    if indices.shape[0]>0:
        istag=indices[0]
        rstag=r[istag,0,0]
        i2stag=iofr(2*rstag)
        i4stag=iofr(4*rstag)
        i8stag=iofr(8*rstag)
        pjtotfinavgvsr5max    = np.max(pjtotfinavgvsr5)
        pjtotfinavgvsr5rstag  = pjtotfinavgvsr5[istag]
        pjtotfinavgvsr5r2stag = pjtotfinavgvsr5[i2stag]
        pjtotfinavgvsr5r4stag = pjtotfinavgvsr5[i4stag]
        pjtotfinavgvsr5r8stag = pjtotfinavgvsr5[i8stag]
    else:
        istag=0
        rstag=0
        i2stag=0
        i4stag=0
        i8stag=0
        pjtotfinavgvsr5max    = np.max(pjtotfinavgvsr5)
        pjtotfinavgvsr5rstag  = 0
        pjtotfinavgvsr5r2stag = 0
        pjtotfinavgvsr5r4stag = 0
        pjtotfinavgvsr5r8stag = 0

    
    

    fstotfinavg = timeavg(fstot[:,ihor],ts,fti,ftf)
    fstotsqfinavg = timeavg(fstot[:,ihor]**2,ts,fti,ftf)**0.5
        
    fsj30finavg = timeavg(fsj30[:,ihor],ts,fti,ftf)
    fsj30sqfinavg = timeavg(fsj30[:,ihor]**2,ts,fti,ftf)**0.5
    
    fc=0
    if showextra:
        ofc = 1
        clr = 'r'
    else:
        ofc = 0
        clr = 'k'

    if epsFm is not None and epsFke is not None:
        FMraw    = mdtot[:,ihor]
        FM       = epsFm * mdtot[:,ihor]
        FM30     = epsFm30 * (mdtot-md30)[:,ihor]
        mdotfinavg = timeavg(FM,ts,fti,ftf)
        FMavg    = epsFm * timeavg(mdtot[:,ihor],ts,fti,ftf,sigma=sigma)
        FMiniavg = epsFm * timeavg(mdtot[:,ihor],ts,iti,itf) 
        FEraw = -edtot[:,ihor]
        FE= epsFke*(FMraw-FEraw)
    else:
        FMiniavg = mdotiniavg
        FMavg = mdotfinavg
        FM = mdtot[:,ihor]-md30[:,ihor]
        FM30 = FM
        FE = pjemtot[:,ihor]
    if showextra:
        lst = 'solid'
    else:
        lst = 'dashed'
    tmax=min(ts[-1],max(fti,ftf))
    #######################
    #
    # Mdot ***
    #
    #######################
    if whichplot == 1:
        if dotavg:
            if len(FMavg.shape)==0:
                ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+FMavg,color=(ofc,fc,fc),linestyle=lst)
            else:
                ax.plot(ts[(ts<ftf)*(ts>=fti)],FMavg[(ts<ftf)*(ts>=fti)],color=(ofc,fc,fc),linestyle=lst)
            if(iti>fti):
                ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+mdotiniavg,color=(ofc,fc,fc))
                
        if plotFM30:
            print( "Plotting FM30 instead of FM" )
            ax.plot(ts,np.abs(FM30),clr,label=r'$\dot Mc^2$')
        else:
            ax.plot(ts,np.abs(FM),clr,label=r'$\dot Mc^2$')
        if findex != None:
            if not isinstance(findex,tuple):
                ax.plot(ts[findex],np.abs(FM)[findex],'o',mfc='r')
            else:
                for fi in findex:
                    ax.plot(ts[fi],np.abs(FM)[fi],'o',mfc='r')#,label=r'$\dot M$')
        #ax.legend(loc='upper left')
        ax.set_ylabel(r'$\dot Mc^2$',fontsize=16,labelpad=9)
        plt.setp( ax.get_xticklabels(), visible=False)
        ax.set_xlim(ts[0],tmax)
        if showextra:
            plt.legend(loc='upper left',bbox_to_anchor=(0.05,0.95),ncol=1,borderaxespad=0,frameon=True,labelspacing=0)
    #######################
    #
    # Pjet
    #
    #######################
    if whichplot == 2:
        ax.plot(ts,(pjem10[:,ihor]),label=r'P_{\rm j}$')
        #ax.legend(loc='upper left')
        if dotavg:
            ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+pjetfinavg)#,label=r'$\langle P_{\rm j}\rangle$')
        ax.set_ylabel(r'$P_{\rm j}$',fontsize=16)
        plt.setp( ax.get_xticklabels(), visible=False)
        ax.set_xlim(ts[0],tmax)
    #######################
    #
    # eta instantaneous
    #
    #######################
    if whichplot == 3:
        ax.plot(ts,(pjem10[:,ihor]/(mdtot[:,ihor]-md10[:,ihor])))#,label=r'$P_{\rm j}/\dot M$')
        #ax.legend(loc='upper left')
        if dotavg:
            ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+pjetfinavg/mdotfinavg)#,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
        ax.set_ylim(0,4)
        #ax.set_xlabel(r'$t\;(GM/c^3)$')
        ax.set_ylabel(r'$P_{\rm j}/\dot M$',fontsize=16)
        ax.set_xlim(ts[0],tmax)
    #######################
    #
    # eta ***
    #
    #######################
    if whichplot == 4:
        etabh = prefactor*FE/FMavg
        etabh_nosigma = prefactor*FE/mdotfinavg
        etaj = prefactor*pjke_mu2[:,iofr(100)]/FMavg
        if epsetaj is not None:
            etaj *= epsetaj
            print( "epsetaj = %g" % epsetaj )
        etaw = prefactor*(pjke_mu1-pjke_mu2)[:,iofr(100)]/FMavg
        etabh2 = prefactor*FE/FMiniavg
        etaj2 = prefactor*pjke_mu2[:,iofr(100)]/mdotiniavg
        etaw2 = prefactor*(pjke_mu1-pjke_mu2)[:,iofr(100)]/mdotiniavg
        if(1 and iti>fti):
            #use mdot averaged over the same time interval for iti<t<=itf
            icond=(ts>=iti)*(ts<itf)
            etabh[icond]=etabh2[icond]
            etaj[icond]=etaj2[icond]
            etaw[icond]=etaw2[icond]
        if dotavg:
            etaj_avg = timeavg(etaj,ts,fti,ftf)
            etabh_avg = timeavg(etabh,ts,fti,ftf,sigma=sigma)
            etabh_avg_nosigma = timeavg(etabh_nosigma,ts,fti,ftf)
            etaw_avg = timeavg(etaw,ts,fti,ftf)
            ptot_avg = timeavg(pjemtot[:,ihor],ts,fti,ftf)
            if showextra:
                ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaj_avg,'--',color=(fc,fc+0.5*(1-fc),fc)) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            #if len(etabh_avg.shape)==0:
            ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etabh_avg_nosigma,color=(ofc,fc,fc),linestyle=lst) 
            #else:
            #    ax.plot(ts[(ts<ftf)*(ts>=fti)],etabh_avg[(ts<ftf)*(ts>=fti)],color=(ofc,fc,fc),linestyle=lst) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            #ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaw_avg,'-.',color=(fc,fc+0.5*(1-fc),fc)) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            if(iti>fti):
                etaj2_avg = timeavg(etaj2,ts,iti,itf)
                etabh2_avg = timeavg(etabh2,ts,iti,itf)
                etaw2_avg = timeavg(etaw2,ts,iti,itf)
                ptot2_avg = timeavg(pjemtot[:,ihor],ts,iti,itf)
                if showextra:
                    ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+etaj2_avg,'--',color=(fc,fc+0.5*(1-fc),fc))
                ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+etabh2_avg,color=(ofc,fc,fc))
                #ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaw2_avg,'-.',color=(fc,fc+0.5*(1-fc),fc)) 
        if showextraeta:
            ax.plot(ts,etabh,'r',label=r'$\eta$')
        else:
            ax.plot(ts,etabh,clr,label=r'$\eta$')

        if showextra or showextraeta:
            l1,=ax.plot(ts,etaj,'g--',label=r'$\eta_{\rm jet}$')
            l2,=ax.plot(ts,etaw,'b-.',label=r'$\eta_{\rm wind}$')
            l1.set_dashes([10,5])
            l2.set_dashes([10,3,2,3])
        if findex != None:
            if not isinstance(findex,tuple):
                if showextra or showextraeta:
                    ax.plot(ts[findex],etaj[findex],'gs')
                ax.plot(ts[findex],etabh[findex],'o',mfc='r')
                if showextra or showextraeta:
                    ax.plot(ts[findex],etaw[findex],'bv')
            else:
                for fi in findex:
                    if showextra or showextraeta:
                        ax.plot(ts[fi],etaw[fi],'bv')#,label=r'$\dot M$')
                        ax.plot(ts[fi],etaj[fi],'gs')#,label=r'$\dot M$')
                    ax.plot(ts[fi],etabh[fi],'o',mfc='r')#,label=r'$\dot M$')
        #ax.legend(loc='upper left')
        #ax.set_ylim(0,2)
        ax.set_xlabel(r'$t\;[r_g/c]$',fontsize=16)
        if prefactor == 100:
            ax.set_ylabel(r'$\eta\ [\%]$',fontsize=16,ha='left',labelpad=20)
        else:
            ax.set_ylabel(r'$\eta$',fontsize=16,labelpad=16)
        ax.set_xlim(ts[0],tmax)
        if showextra or showextraeta:
            leg=plt.legend(loc='upper right',bbox_to_anchor=(0.95,0.97),ncol=3,borderpad = 0,borderaxespad=0,frameon=False,labelspacing=0)
            for t in leg.get_texts():
                t.set_fontsize(16)    # the legend text fontsize
  
        if len(etabh_avg_nosigma.shape)==0:
            print( "eta_BH = %g, eta_j = %g, eta_w = %g, eta_jw = %g, mdot = %g, ptot_BH = %g, ti = %g, tf = %g" % ( etabh_avg_nosigma, etaj_avg, etaw_avg, etaj_avg + etaw_avg, mdotfinavg, ptot_avg, fti, ftf ) )
        if iti > fti:
            print( "eta_BH2 = %g, eta_j2 = %g, eta_w2 = %g, eta_jw2 = %g, mdot2 = %g, ptot2_BH = %g" % ( etabh2_avg, etaj2_avg, etaw2_avg, etaj2_avg + etaw2_avg, mdotiniavg, ptot2_avg ) )

    #######################
    #
    # eta NEW ***
    #
    #######################
    if whichplot == 6:
        etabh = prefactor*pjemtot[:,ihor]/mdotfinavg
        etaj = prefactor*pjke_mu2[:,iofr(100)]/mdotfinavg
        etaw = prefactor*(pjke_mu1-pjke_mu2)[:,iofr(100)]/mdotfinavg
        etabh2 = prefactor*pjemtot[:,ihor]/mdotiniavg
        etaj2 = prefactor*pjke_mu2[:,iofr(100)]/mdotiniavg
        etaw2 = prefactor*(pjke_mu1-pjke_mu2)[:,iofr(100)]/mdotiniavg
        if(1 and iti>fti):
            #use mdot averaged over the same time interval for iti<t<=itf
            icond=(ts>=iti)*(ts<itf)
            etabh[icond]=etabh2[icond]
            etaj[icond]=etaj2[icond]
            etaw[icond]=etaw2[icond]
        if dotavg:
            etaj_avg = timeavg(etaj,ts,fti,ftf)
            etabh_avg = timeavg(etabh,ts,fti,ftf)
            etaw_avg = timeavg(etaw,ts,fti,ftf)
            ptot_avg = timeavg(pjemtot[:,ihor],ts,fti,ftf)
            if showextra:
                ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaj_avg,'--',color=(fc,fc+0.5*(1-fc),fc)) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etabh_avg,color=(ofc,fc,fc)) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            #ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaw_avg,'-.',color=(fc,fc+0.5*(1-fc),fc)) 
            #,label=r'$\langle P_j\rangle/\langle\dot M\rangle$')
            if(iti>fti):
                etaj2_avg = timeavg(etaj2,ts,iti,itf)
                etabh2_avg = timeavg(etabh2,ts,iti,itf)
                etaw2_avg = timeavg(etaw2,ts,iti,itf)
                ptot2_avg = timeavg(pjemtot[:,ihor],ts,iti,itf)
                if showextra:
                    ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+etaj2_avg,'--',color=(fc,fc+0.5*(1-fc),fc))
                ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+etabh2_avg,color=(ofc,fc,fc))
                #ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+etaw2_avg,'-.',color=(fc,fc+0.5*(1-fc),fc)) 
        ax.plot(ts,etabh,clr,label=r'$\eta_{\rm BH}$')
        if showextra:
            ax.plot(ts,etaj,'g--',label=r'$\eta_{\rm jet}$')
            ax.plot(ts,etaw,'b-.',label=r'$\eta_{\rm wind}$')
        if findex != None:
            if not isinstance(findex,tuple):
                if showextra:
                    ax.plot(ts[findex],etaj[findex],'gs')
                ax.plot(ts[findex],etabh[findex],'o',mfc='r')
                if showextra:
                    ax.plot(ts[findex],etaw[findex],'bv')
            else:
                for fi in findex:
                    if showextra:
                        ax.plot(ts[fi],etaw[fi],'bv')#,label=r'$\dot M$')
                        ax.plot(ts[fi],etaj[fi],'gs')#,label=r'$\dot M$')
                    ax.plot(ts[fi],etabh[fi],'o',mfc='r')#,label=r'$\dot M$')
        #ax.legend(loc='upper left')
        #ax.set_ylim(0,2)
        ax.set_xlabel(r'$t\;[r_g/c]$',fontsize=16)
        ax.set_ylabel(r'$\eta\ [\%]$',fontsize=16,ha='left',labelpad=20)
        ax.set_xlim(ts[0],tmax)
        if showextra:
            plt.legend(loc='upper left',bbox_to_anchor=(0.05,0.95),ncol=1,borderpad = 0,borderaxespad=0,frameon=True,labelspacing=0)


        print( "eta_BH = %g, eta_j = %g, eta_w = %g, eta_jw = %g, mdot = %g, ptot_BH = %g" % ( etabh_avg, etaj_avg, etaw_avg, etaj_avg + etaw_avg, mdotfinavg, ptot_avg ) )
        if iti > fti:
            print( "eta_BH2 = %g, eta_j2 = %g, eta_w2 = %g, eta_jw2 = %g, mdot2 = %g, ptot2_BH = %g" % ( etabh2_avg, etaj2_avg, etaw2_avg, etaj2_avg + etaw2_avg, mdotiniavg, ptot2_avg ) )

        #xxx
    #######################
    #
    # \Phi ***
    #
    #######################
    if whichplot == 5:
        if usegaussianunits == True:
            unitsfactor = (4*np.pi)**0.5*2*np.pi
        else:
            unitsfactor = 1.
        omh = a / (2*(1+(1-a**2)**0.5))
        phibh=fstot[:,ihor]/4/np.pi/FMavg**0.5
        phibh_nosigma=fstot[:,ihor]/4/np.pi/mdotfinavg**0.5
        phij=phiabsj_mu2[:,iofr(100)]/4/np.pi/mdotfinavg**0.5
        phiw=(phiabsj_mu1-phiabsj_mu2)[:,iofr(100)]/4/np.pi/mdotfinavg**0.5
        phibh2=fstot[:,ihor]/4/np.pi/mdotiniavg**0.5
        phij2=phiabsj_mu2[:,iofr(100)]/4/np.pi/mdotiniavg**0.5
        phiw2=(phiabsj_mu1-phiabsj_mu2)[:,iofr(100)]/4/np.pi/mdotiniavg**0.5
        if(1 and iti>fti):
            #use phi averaged over the same time interval for iti<t<=itf
            icond=(ts>=iti)*(ts<itf)
            phibh[icond]=phibh2[icond]
            phij[icond]=phij2[icond]
            phiw[icond]=phiw2[icond]
        if dotavg:
            phibh_avg = timeavg(phibh**2,ts,fti,ftf,sigma=sigma)**0.5
            phibh_avg_nosigma = timeavg(phibh_nosigma**2,ts,fti,ftf)**0.5
            fstot_avg = timeavg(fstot[:,ihor]**2,ts,fti,ftf)**0.5
            if showextra:
                ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+unitsfactor*timeavg(phij**2,ts,fti,ftf)**0.5,'--',color=(fc,fc+0.5*(1-fc),fc))
            #if len(phibh_avg.shape)==0:
            ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+unitsfactor*phibh_avg_nosigma,color=(ofc,fc,fc),linestyle=lst)
            #else:
            #    ax.plot(ts[(ts<ftf)*(ts>=fti)],unitsfactor*phibh_avg[(ts<ftf)*(ts>=fti)],color=(ofc,fc,fc),linestyle=lst)
            #ax.plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+unitsfactor*timeavg(phiw**2,ts,fti,ftf)**0.5,'-.',color=(fc,fc,1))
            if(iti>fti):
                phibh2_avg = timeavg(phibh2**2,ts,iti,itf)**0.5
                fstot2_avg = timeavg(fstot[:,ihor]**2,ts,iti,itf)**0.5
                if showextra:
                    ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+unitsfactor*timeavg(phij2**2,ts,iti,itf)**0.5,'--',color=(fc,fc+0.5*(1-fc),fc))
                ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+unitsfactor*phibh2_avg,color=(ofc,fc,fc))
                #ax.plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+unitsfactor*timeavg(phiw2**2,ts,iti,itf)**0.5,'-.',color=(fc,fc,1))
        #To approximately get efficiency:
        #ax.plot(ts,unitsfactor*2./3.*np.pi*omh**2*np.abs(fsj30[:,ihor]/4/np.pi)**2/mdotfinavg)
        #prefactor to get sqrt(eta): (2./3.*np.pi*omh**2)**0.5
        ax.plot(ts,unitsfactor*phibh,clr,label=r'$\phi_{\rm BH}$')
        ax.set_xlim(ts[0],tmax)
        if showextra:
            ax.plot(ts,unitsfactor*phij,'g--',label=r'$\phi_{\rm jet}$')
        #ax.plot(ts,unitsfactor*phiw,'b-.',label=r'$\phi_{\rm wind}$')
        if findex != None:
            if not isinstance(findex,tuple):
                if showextra:
                    ax.plot(ts[findex],unitsfactor*phij[findex],'gs')
                ax.plot(ts[findex],unitsfactor*phibh[findex],'o',mfc='r')
                #ax.plot(ts[findex],unitsfactor*phiw[findex],'bv')
            else:
                for fi in findex:
                    if showextra:
                        ax.plot(ts[fi],unitsfactor*phij[fi],'gs')
                    ax.plot(ts[fi],unitsfactor*phibh[fi],'o',mfc='r')
                    #ax.plot(ts[fi],unitsfactor*phiw[fi],'bv')
        #ax.legend(loc='upper left')
        #ax.set_ylabel(r'$\ \ \ k\Phi_j/\langle\dot M\rangle^{\!1/2}$',fontsize=16)
        ax.set_ylabel(r'$\phi$',fontsize=16,labelpad=16)
        if showextra:
            plt.legend(loc='upper left',bbox_to_anchor=(0.05,0.95),ncol=1,borderpad = 0,borderaxespad=0,frameon=True,labelspacing=0)
        plt.setp( ax.get_xticklabels(), visible=False )
        if len(phibh_avg_nosigma.shape)==0:
            print( "phi_BH = %g, fstot = %g" % ( phibh_avg_nosigma, fstot_avg ) )
        if iti > fti:
            print( "phi2_BH = %g, fstot2 = %g" % ( phibh2_avg, fstot2_avg ) )

    if whichplot == -1:
        etajetavg = pjetfinavg/mdotfinavg
        foutpower = open( "pjet_power_%s.txt" %  os.path.basename(os.getcwd()), "w" )
        #foutpower.write( "#Name a Mdot   Pjet    Etajet  Psitot Psisqtot**0.5 Psijet Psisqjet**0.5 rstag Pjtotmax Pjtot1rstag Pjtot2rstag Pjtot4rstag Pjtot8rstag\n"  )
        foutpower.write( "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (os.path.basename(os.getcwd()), a, mdotfinavg, 
                                                                       pjetfinavg, etajetavg, fstotfinavg, 
                                                                       fstotsqfinavg, fsj30finavg, fsj30sqfinavg, 
                                                                       rstag, pjtotfinavgvsr5max, pjtotfinavgvsr5rstag, 
                                                                       pjtotfinavgvsr5r2stag, pjtotfinavgvsr5r4stag, pjtotfinavgvsr5r8stag) )
        #flush to disk just in case to make sure all is written
        foutpower.flush()
        os.fsync(foutpower.fileno())
        foutpower.close()
        return( mdotfinavg, fstotfinavg, fstotsqfinavg, fsj30finavg, fsj30sqfinavg, pjemfinavgtot )

    if whichplot == -2:
        return( mdtotvsr, edtotvsr, edmavsr, ldtotvsr )

    if whichplot == -199:
        horavg = timeavg(hoverr,ts,fti,ftf)
        return( horavg )

    if whichplot == -200:
        #XXX compute edmavsr without polar regions with avg_aphi < avg_phi[iofr(rhor),aphi_j_val]
        if aphi_j_val > 0 and os.path.isfile( "avg2d.npy" ):
            avgmem = get2davg(usedefault=1)
            assignavg2dvars(avgmem)
            #sum in phi and theta
            edtotEM = edtotvsr - edmavsr
            eoutcumMA = scaletofullwedge(nz*(-gdet*avg_TudMA[1,0]*_dx2*_dx3).sum(axis=2)).cumsum(axis=1)
            eoutcumEM = scaletofullwedge(nz*(-gdet*avg_TudEM[1,0]*_dx2*_dx3).sum(axis=2)).cumsum(axis=1)
            moutcum = scaletofullwedge(nz*(gdet*avg_rhouu[1]*_dx2*_dx3).sum(axis=2)).cumsum(axis=1)
            if aphi_j_val > 0:
                #flux in x2-direction integrated in x1-x3 plane
                #MA energy
                eperpMA = scaletofullwedge(nz*(-gdet*avg_TudMA[2,0]*_dx1*_dx3).sum(axis=2)).cumsum(axis=0)
                #set correction to zero at outer radius (take r=20) where correction is negligible
                iref = iofr(20)
                eperpMA = eperpMA-eperpMA[iref:iref+1]
                #move in x2 to the correct face-location
                eperpMAdn = 0.5*(eperpMA[:,aphi_j_val]+eperpMA[:,aphi_j_val-1])
                eperpMAup = 0.5*(eperpMA[:,ny-1-aphi_j_val]+eperpMA[:,ny-aphi_j_val])
                #eperpMAdn comes with negative sign to account for energy flowing into domain there
                eperpMA = eperpMAup - eperpMAdn
                #shift to correct radial location by half cell right (so located at cell center instead of cell face)
                eperpMA[1:] = 0.5*(eperpMA[:-1]+eperpMA[1:])
                #EM energy
                eperpEM = scaletofullwedge(nz*(-gdet*avg_TudEM[2,0]*_dx1*_dx3).sum(axis=2)).cumsum(axis=0)
                #set correction to zero at outer radius (take r=20) where correction is negligible
                iref = iofr(20)
                eperpEM = eperpEM-eperpEM[iref:iref+1]
                #move in x2 to the correct face-location
                eperpEMdn = 0.5*(eperpEM[:,aphi_j_val]+eperpEM[:,aphi_j_val-1])
                eperpEMup = 0.5*(eperpEM[:,ny-1-aphi_j_val]+eperpEM[:,ny-aphi_j_val])
                #eperpEMdn comes with negative sign to account for energy flowing into domain there
                eperpEM = eperpEMup - eperpEMdn
                #shift to correct radial location by half cell right (so located at cell center instead of cell face)
                eperpEM[1:] = 0.5*(eperpEM[:-1]+eperpEM[1:])
                #mass
                #Mdot
                mperp = scaletofullwedge(nz*(gdet*avg_rhouu[2]*_dx1*_dx3).sum(axis=2)).cumsum(axis=0)
                #set correction to zero at outer radius (take r=20) where correction is negligible
                iref = iofr(20)
                mperp = mperp-mperp[iref:iref+1]
                #move in x2 to the correct face-location
                mperpdn = 0.5*(mperp[:,aphi_j_val]+mperp[:,aphi_j_val-1])
                mperpup = 0.5*(mperp[:,ny-1-aphi_j_val]+mperp[:,ny-aphi_j_val])
                #mperpdn comes with negative sign to account for energy flowing into domain there
                mperp = mperpup - mperpdn
                #shift to correct radial location by half cell right (so located at cell center instead of cell face)
                mperp[1:] = 0.5*(mperp[:-1]+mperp[1:])
            else:
                eperpMA = 0*edtotEM
                eperpEM = 0*edtotEM
                mperp = 0*edtotEM
            edtotMA = cutout_along_aphi(eoutcumMA,aphi_j_val=aphi_j_val) - eperpMA
            #comment next line out if don't want to cut out pieces of EM energy flux
            #edtotEM = cutout_along_aphi(eoutcumEM,aphi_j_val=aphi_j_val) - eperpEM
            mdtotvsr = -(cutout_along_aphi(moutcum,aphi_j_val=aphi_j_val) - mperp)
            edtotvsr = edtotEM + edtotMA
            edmavsr = edtotMA

        return( mdtotvsr, edtotvsr, edmavsr, ldtotvsr,
                mdotfinavgvsr5, mdotfinavgvsr10, mdotfinavgvsr20, mdotfinavgvsr30, mdotfinavgvsr40,
                pjemfinavgvsr5, pjemfinavgvsr10, pjemfinavgvsr20, pjemfinavgvsr30, pjemfinavgvsr40,
                pjmafinavgvsr5, pjmafinavgvsr10, pjmafinavgvsr20, pjmafinavgvsr30, pjmafinavgvsr40,
                fstotfinavg, fstotsqfinavg,
                pjke_mu2_avg, pjke_mu1_avg,
                timeavg(gdetF10,ts,fti,ftf), timeavg(gdetF11,ts,fti,ftf), timeavg(gdetF12,ts,fti,ftf))
 
    if whichplot == -300:
        #BL metric g_rr
        rh = 1+(1-a**2)**0.5
        r1d = r[:,0,0]
        Sigma = r**2+a**2*np.cos(h)**2
        Delta = r**2-2*r+a**2
        gdrr = Sigma/Delta
        gdtt = -(1-2*r/Sigma)
        #
        uutKS = uu[0]
        uurKS = dxdxp[1,1]*uu[1]
        uuhKS = dxdxp[2,2]*uu[2] + dxdxp[2,1]*uu[1]
        uupKS = dxdxp[3,3]*uu[3]
        uurBL = uurKS
        uutBL = uutKS - 2*(r/Delta)*uurKS - (a/Delta) * uupKS
        vurBL = uurBL / uutBL
        vrBL = gdrr**0.5*vurBL 
        plt.figure(2)
        plt.loglog( r1d, -vrBL[:,ny/2].mean(-1), 'r', label=r"$t=%g$" % t )
        plt.xlim( rh, 100 )
        plt.ylim( 1e-3, 2 )
        plt.legend( loc = 'lower left' )
       
    if whichplot == -3:
        #BL metric g_rr
        rh = 1+(1-a**2)**0.5
        r1d = r[:,0,0]
        Sigma = r**2+a**2*np.cos(h)**2
        Delta = r**2-2*r+a**2
        gdrr = Sigma/Delta
        gdtt = -(1-2*r/Sigma)
        #
        uutKS = uu02h
        uurKS = dxdxp[1,1,None,:,ny/2,0]*uus12h
        uupKS = dxdxp[3,3,None,:,ny/2,0]*uus32h
        uurBL = uurKS
        uutBL = uutKS - 2*(r/Delta)[None,:,ny/2,0]*uurKS - (a/Delta)[None,:,ny/2,0] * uupKS
        vurBL = uurBL / uutBL
        vrBL = gdrr[None,:,ny/2,0]**0.5*vurBL 
        t1 = 0.5e4
        t2 = 1e4
        avg1_vr = timeavg(vrBL,ts,t1,t2) 
        t1 = 1e4
        t2 = 1.5e4
        avg2_vr = timeavg(vrBL,ts,t1,t2) 
        t1 = 1.5e4
        t2 = 2e4
        avg3_vr = timeavg(vrBL,ts,t1,t2) 
        plt.figure(1)
        plt.loglog( r1d, -avg1_vr, 'r', label=r"$5,000<t<10,000$" )
        plt.loglog( r1d, -avg2_vr, 'g', label=r"$10,000<t<15,000$" )
        plt.loglog( r1d, -avg3_vr, 'b', label=r"$15,000<t<20,000$" )
        plt.xlim( rh, 200 )
        plt.ylim( 1e-3, 2 )
        plt.grid()
        plt.ylabel( r"$v_{\hat r}$", fontsize = 20 )
        plt.xlabel( r"$r$", fontsize = 20 )
        plt.legend( loc = 'upper right' )

    if whichplot == -4:
        rh = 1+(1-a**2)**0.5
        r1d = r[:,0,0]
        plt.figure(1)
        plt.loglog( r1d, timeavg(rhos2h,ts,5000,10000), 'r', label=r"$5,000<t<10,000$" )
        plt.loglog( r1d, timeavg(rhos2h,ts,10000,15000), 'g', label=r"$10,000<t<15,000$" )
        plt.loglog( r1d, timeavg(rhos2h,ts,15000,20000), 'b', label=r"$15,000<t<20,000$" )
        plt.xlabel(r"$r$", fontsize = 15)
        plt.ylabel(r"$\rho$", fontsize = 15)
        plt.xlim( rh, 100 )
        plt.ylim( 1e-2, 15 )
        plt.grid()
        plt.legend( loc = 'lower left' )

    #if whichplot == -5:
        
    if whichplot == None:
        fig,plotlist=plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(12,16),num=1)
        #plt.clf()
        plottitle = "a = %g: %s" % ( a, os.path.basename(os.getcwd()) )
        plt.suptitle( plottitle )
        plt.subplots_adjust(hspace=0.1) #increase vertical spacing to avoid crowding
        print fstot[:,ihor].shape
        plotlist[0].plot(ts,fstot[:,ihor],label=r'$\Phi_{\rm h,tot}$')
        #plotlist[0].plot(ts,fsj5[:,ihor],label=r'$\Phi_{\rm h,5}$')
        plotlist[0].plot(ts,fsj30[:,ihor],label=r'$\Phi_{\rm h,30}$')
        #plotlist[0].plot(ts,fs,'r+') #, label=r'$\Phi_{\rm h}/0.5\Phi_{\rm i}$: Data Points')
        if dotavg:
            plotlist[0].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+fstotsqfinavg,label=r'$\langle \Phi^2_{\rm h,tot}\rangle^{1/2}$')
            plotlist[0].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+fsj30sqfinavg,label=r'$\langle \Phi^2_{\rm h,30}\rangle^{1/2}$')
        plotlist[0].legend(loc='upper left')
        #plt.xlabel(r'$t\;(GM/c^3)$')
        plotlist[0].set_ylabel(r'$\Phi_{\rm h}$',fontsize=16)
        plt.setp( plotlist[0].get_xticklabels(), visible=False)
        plotlist[0].grid(True)
        #
        #plotlist[1].subplot(212,sharex=True)
        #plotlist[1].plot(ts,np.abs(mdtot[:,ihor]),label=r'$\dot M_{\rm h,tot}$')
        #plotlist[1].plot(ts,np.abs(mdtot[:,ihor]-md5[:,ihor]),label=r'$\dot M_{\rm h,tot,bsqorho<5}$')
        plotlist[1].plot(ts,np.abs(mdtot[:,ihor]-md10[:,ihor]),label=r'$\dot M_{{\rm h,tot}, b^2/rho<10}$')
        plotlist[1].plot(ts,np.abs(mdtot[:,ihor]-md30[:,ihor]),label=r'$\dot M_{{\rm h,tot}, b^2/rho<30}$')
        if dotavg:
            #plotlist[1].plot(ts[(ts<itf)*(ts>=iti)],0*ts[(ts<itf)*(ts>=iti)]+mdotiniavg,label=r'$\langle \dot M_{{\rm h,tot}, b^2/\rho<10}\rangle_{i}$')
            plotlist[1].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+mdotfinavg,label=r'$\langle \dot M_{{\rm h,tot}, b^2/\rho<10}\rangle_{f}$')
        #plotlist[1].plot(ts,np.abs(md2h[:,ihor]),label=r'$\dot M_{\rm h,2h}$')
        #plotlist[1].plot(ts,np.abs(md4h[:,ihor]),label=r'$\dot M_{\rm h,4h}$')
        #plotlist[1].plot(ts,np.abs(md2hor[:,ihor]),label=r'$\dot M_{\rm h,2hor}$')
        #plotlist[1].plot(ts,np.abs(mdrhosq[:,ihor]),label=r'$\dot M_{\rm h,rhosq}$')
        #plotlist[1].plot(ts,np.abs(md5[:,ihor]),label=r'$\dot M_{\rm h,5}$')
        plotlist[1].plot(ts,np.abs(md10[:,ihor]),label=r'$\dot M_{\rm h,10}$')
        plotlist[1].plot(ts,np.abs(md30[:,ihor]),label=r'$\dot M_{\rm h,30}$')
        #plotlist[1].plot(ts,np.abs(md[:,ihor]),'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
        plotlist[1].legend(loc='upper left')
        #plotlist[1].set_xlabel(r'$t\;(GM/c^3)$')
        plotlist[1].set_ylabel(r'$\dot M_{\rm h}$',fontsize=16)
        plt.setp( plotlist[1].get_xticklabels(), visible=False)

        plotlist[2].plot(ts,(pjem10[:,ihor]),label=r'$P_{\rm j,em10}$')
        plotlist[2].plot(ts,(pjem30[:,ihor]),label=r'$P_{\rm j,em30}$')
        if dotavg:
            plotlist[2].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+pjetfinavg,label=r'$\langle P_{{\rm j,em30}\rangle_{f}}$')
        plotlist[2].legend(loc='upper left')
        #plotlist[2].set_xlabel(r'$t\;(GM/c^3)$')
        plotlist[2].set_ylabel(r'$P_{\rm j}$',fontsize=16)

        #plotlist[3].plot(ts,(pjem10[:,ihor]/mdtot[:,ihor]),label=r'$P_{\rm j,em10}/\dot M_{\rm tot}$')
        #plotlist[3].plot(ts,(pjem5[:,ihor]/(mdtot[:,ihor]-md5[:,ihor])),label=r'$P_{\rm j,em5}/\dot M_{{\rm tot},b^2/\rho<5}$')
        plotlist[3].plot(ts,(pjem30[:,ihor]/mdotfinavg),label=r'$\dot \eta_{10}=P_{\rm j,em10}/\dot M_{{\rm tot},b^2/\rho<30}$')
        if dotavg:
            #plotlist[3].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+pjetfinavg/mdotiniavg,label=r'$\langle P_j\rangle/\langle\dot M_i\rangle_{f}$')
            plotlist[3].plot(ts[(ts<ftf)*(ts>=fti)],0*ts[(ts<ftf)*(ts>=fti)]+pjetfinavg/mdotfinavg,'r',label=r'$\langle P_j\rangle/\langle\dot M_f\rangle_{f}$')
        #plotlist[3].set_ylim(0,6)
        plotlist[3].legend(loc='upper left')
        plotlist[3].set_xlabel(r'$t\;(GM/c^3)$')
        #plotlist[3].set_ylabel(r'$P_{\rm j}/\dot M_{\rm h}$',fontsize=16)
        plotlist[3].set_ylabel(r'$\eta_{\rm jet}$',fontsize=16)

        #title("\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
        #      fontsize=16, color='r')
        plotlist[0].grid(True)
        plotlist[1].grid(True)
        plotlist[2].grid(True)
        plotlist[3].grid(True)
        fig.savefig('pjet1_%s.pdf' % os.path.basename(os.getcwd()) )

        #density/velocity/hor figure
        #!!!rhor=1+(1-a**2)**0.5
        fig,plotlist=plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(12,16),num=2)
        #plt.clf()
        plottitle = r"\rho,u^r,h/r: a = %g: %s" % ( a, os.path.basename(os.getcwd()) )
        plt.suptitle( plottitle )
        plt.subplots_adjust(hspace=0.1) #increase vertical spacing to avoid crowding
        #print fstot[:,ihor].shape
        plotlist[0].plot(ts,hoverr[:,ihor],label=r'$(h/r)_{\rm h}$')
        plotlist[0].plot(ts,hoverr[:,iofr(2)],label=r'$(h/r)_{\rm 2}$') ##### continue here
        plotlist[0].plot(ts,hoverr[:,iofr(4)],label=r'$(h/r)_{\rm 4}$')
        plotlist[0].plot(ts,hoverr[:,iofr(8)],label=r'$(h/r)_{\rm 8}$')
        #lotlist[0].plot(ts,hoverr[:,iofr(10)],label=r'$(h/r)_{\rm 10}$')
        #plotlist[0].plot(ts,hoverr[:,iofr(12)],label=r'$(h/r)_{\rm 12}$')
        #plotlist[0].plot(ts,hoverr[:,iofr(15)],label=r'$(h/r)_{\rm 15}$')
        #thetamid
        plotlist[0].plot(ts,(thetamid-np.pi/2)[:,ihor],'--',label=r'$\theta_{\rm h}$')
        plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(2)],'--',label=r'$\theta_{\rm 2}$') ##### continue here
        plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(4)],'--',label=r'$\theta_{\rm 4}$')
        plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(8)],'--',label=r'$\theta_{\rm 8}$')
        #plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(10)],'--',label=r'$\theta_{\rm 10}$')
        #plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(12)],'--',label=r'$\theta_{\rm 12}$')
        #plotlist[0].plot(ts,(thetamid-np.pi/2)[:,iofr(15)],'--',label=r'$\theta_{\rm 15}$')
        #plotlist[0].plot(ts,fs,'r+') #, label=r'$\Phi_{\rm h}/0.5\Phi_{\rm i}$: Data Points')
        #legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plotlist[0].legend(loc='upper right',ncol=4)
        #plt.xlabel(r'$t\;(GM/c^3)$')
        plotlist[0].set_ylabel(r'$h/r$',fontsize=16)
        plt.setp( plotlist[0].get_xticklabels(), visible=False)
        plotlist[0].grid(True)
        #
        #plotlist[1].subplot(212,sharex=True)
        plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,ihor],label=r'$-u^r_{\rm h}$')
        plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(2)],label=r'$-u^r_{\rm 2}$') ##### continue here
        plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(4)],label=r'$-u^r_{\rm 4}$')
        plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(8)],label=r'$-u^r_{\rm 8}$')
        #plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(10)],label=r'$-u^r_{\rm 10}$')
        #plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(12)],label=r'$-u^r_{\rm 12}$')
        #plotlist[1].plot(ts,(-uus12hor*dxdxp[1][1][:,0,0])[:,iofr(15)],label=r'$-u^r_{\rm 15}$')
        plotlist[1].legend(loc='upper right')
        #plotlist[1].set_xlabel(r'$t\;(GM/c^3)$')
        plotlist[1].set_ylabel(r'$u^r$',fontsize=16)
        plt.setp( plotlist[1].get_xticklabels(), visible=False)

        plotlist[2].plot(ts,rhos2hor[:,ihor],label=r'$\rho_{\rm h}$')
        plotlist[2].plot(ts,rhos2hor[:,iofr(2)],label=r'$\rho_{\rm 2}$') ##### continue here
        plotlist[2].plot(ts,rhos2hor[:,iofr(4)],label=r'$\rho_{\rm 4}$')
        plotlist[2].plot(ts,rhos2hor[:,iofr(8)],label=r'$\rho_{\rm 8}$')
        #plotlist[2].plot(ts,rhos2hor[:,iofr(10)],label=r'$\rho_{\rm 10}$')
        #plotlist[2].plot(ts,rhos2hor[:,iofr(12)],label=r'$\rho_{\rm 12}$')
        #plotlist[2].plot(ts,rhos2hor[:,iofr(15)],label=r'$\rho_{\rm 15}$')
        plotlist[2].legend(loc='upper left')
        #plotlist[2].set_xlabel(r'$t\;(GM/c^3)$')
        plotlist[2].set_ylabel(r'$\rho$',fontsize=16)

        plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,ihor],label=r'$u^r_{\rm h}$')
        plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(2)],label=r'$u^r_{\rm 2}$') ##### continue here
        plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(4)],label=r'$u^r_{\rm 4}$')
        plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(8)],label=r'$u^r_{\rm 8}$')
        #plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(10)],label=r'$u^r_{\rm 10}$')
        #plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(12)],label=r'$u^r_{\rm 12}$')
        #plotlist[3].plot(ts,(ugs2hor/rhos2hor)[:,iofr(15)],label=r'$u^r_{\rm 15}$')
        plotlist[3].legend(loc='upper left')
        plotlist[3].set_xlabel(r'$t\;(GM/c^3)$')
        plotlist[3].set_ylabel(r'$u_g/\rho$',fontsize=16)

        #title("\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
        #      fontsize=16, color='r')
        plotlist[0].grid(True)
        plotlist[1].grid(True)
        plotlist[2].grid(True)
        plotlist[3].grid(True)
        fig.savefig('pjet2_%s.pdf' % os.path.basename(os.getcwd()) )
    
        plt.figure(3)
        plt.clf()
        plt.plot(r[:,0,0],mdotfinavgvsr,label=r'$\dot M_{\rm tot}$')
        plt.plot(r[:,0,0],mdotfinavgvsr5,label=r'$\dot M_{b^2/\rho<5}$')
        plt.plot(r[:,0,0],mdotfinavgvsr10,label=r'$\dot M_{b^2/\rho<10}$')
        plt.plot(r[:,0,0],mdotfinavgvsr20,label=r'$\dot M_{b^2/\rho<20}$')
        plt.plot(r[:,0,0],mdotfinavgvsr30,label=r'$\dot M_{b^2/\rho<30}$')
        plt.plot(r[:,0,0],mdotfinavgvsr40,label=r'$\dot M_{b^2/\rho<40}$')
        plt.xlim(1+(1-a**2)**0.5,20)
        plt.ylim(0,np.max(mdotfinavgvsr[r[:,0,0]<20]))
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('pjet3_%s.pdf' % os.path.basename(os.getcwd()) )

        plt.figure(4)
        plt.clf()
        rmax=50
        plt.plot(r[:,0,0],pjemfinavgvsr,'b',label=r'$\dot Pem_{\rm tot}$')
        plt.plot(r[:,0,0],pjemfinavgvsr5,'g',label=r'$\dot Pem_{b^2/\rho>5}$')
        plt.plot(r[:,0,0],pjemfinavgvsr10,'r',label=r'$\dot Pem_{b^2/\rho>10}$')
        plt.plot(r[:,0,0],pjemfinavgvsr20,'c',label=r'$\dot Pem_{b^2/\rho>20}$')
        plt.plot(r[:,0,0],pjemfinavgvsr30,'m',label=r'$\dot Pem_{b^2/\rho>30}$')
        plt.plot(r[:,0,0],pjemfinavgvsr40,'y',label=r'$\dot Pem_{b^2/\rho>40}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr,'b--',label=r'$\dot P_{\rm tot}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr5,'g--',label=r'$\dot P_{b^2/\rho>5}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr10,'r--',label=r'$\dot P_{b^2/\rho>10}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr20,'c--',label=r'$\dot P_{b^2/\rho>20}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr30,'m--',label=r'$\dot P_{b^2/\rho>30}$')
        plt.plot(r[:,0,0],pjtotfinavgvsr40,'y--',label=r'$\dot P_{b^2/\rho>40}$')
        plt.xlim(1+(1-a**2)**0.5,rmax)
        plt.ylim(0,np.max(pjemfinavgvsr[r[:,0,0]<rmax]))
        plt.legend(loc='lower right',ncol=2)
        plt.grid()
        plt.savefig('pjet4_%s.pdf' % os.path.basename(os.getcwd()) )

def gaussf( x, x0, sigma ):
    """ Returns normalized Gaussian centered at x0 with stdev sigma """
    return( np.exp(-0.5*(x-x0)**2/sigma**2) / (np.sqrt(2*np.pi)*sigma) )

def timeavg( qty, ts, fti, ftf, step = 1, sigma = None ):
    cond = (ts<ftf)*(ts>=fti)
    if sigma is None:
        #use masked array to remove any stray NaN's
        qtycond = np.ma.masked_array(qty[cond],np.isnan(qty[cond]))
        qtycond = qtycond[::step]
        qtyavg = qtycond.mean(axis=0,dtype=np.float64)
    else:
        qtym = np.ma.masked_array(qty,np.isnan(qty))
        qtyavg=np.zeros_like(qtym)
        for (i0,t0) in enumerate(ts):
            mygauss_at_t0 = gaussf( ts, t0, sigma )
            #assumes uniform spacing in time
            if qtym.ndim == 2:
                mygauss_at_t0 = mygauss_at_t0[:,None]
            if t0 >= fti and t0 <= ftf:
                qtyavg[i0] += (qtym * mygauss_at_t0)[cond].sum(axis=0) / mygauss_at_t0[cond].sum()
        qtyavg[ts<fti] += qtyavg[ts>=fti][0]
        qtyavg[ts>ftf] += qtyavg[ts<=ftf][-1]
        #pdb.set_trace()

    return( qtyavg )

def getstagparams(var=None,rmax=20,doplot=1,doreadgrid=1,usedefault=1,fixupnearaxis=False):
    if doreadgrid:
        grid3d("gdump.bin",use2d=True)
    avgmem = get2davg(usedefault=usedefault)
    assignavg2dvars(avgmem)
    #a large enough distance that floors are not applied, yet close enough that reaches inflow equilibrium
    rnoflooradded=rmax
    #radial index and radius of stagnation surface
    sol = avg_uu[1]*(r-rmax)
    istag = np.floor( findroot2d(sol, ti, axis = 1, isleft=True, fallback = 1, fallbackval = iofr(rnoflooradded)) + 0.5 )
    jstag = np.floor( findroot2d(sol, tj, axis = 1, isleft=True, fallback = 1, fallbackval = iofr(rnoflooradded)) + 0.5 )
    rstag = findroot2d( sol, r, axis = 1, isleft=True, fallback = 1, fallbackval = rnoflooradded )
    hstag = findroot2d( sol, h, axis = 1, isleft=True, fallback = 1, fallbackval = np.pi/2.)
    if fixupnearaxis:
        for j in np.array([1,-2]):
            rstag[j]=0.5*(rstag[j-1]+rstag[j+1])
            istag[j]=iofr(rstag[j])
            hstag[j]=h[istag[j],j,0]
    if doplot:
        plt.figure(1)
        plt.clf()
        plt.plot(hstag,rstag)
        plt.figure(2)
        plco(avg_uu[1],levels=(0,),colors='k',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plt.plot(rstag*np.sin(hstag),rstag*np.cos(hstag))
        plt.xlim(0,10)
        plt.ylim(-5,5)
    #cond=(ti==istag[None,:,None])*(tj==jstag[None,:,None])
    #print( zip(r[cond],rstag) )
    if var is not None:
        varstag = findroot2d( avg_uu[1], var, axis = 1, isleft=True, fallback = 0)
        return varstag
    else:
        return istag, jstag, hstag, rstag

def get_dUfloor( floordumpno, maxrinflowequilibrium = 20, aphi_j_val=0 ):
    """ maxrsteady should be chosen to be on the outside of the inflow equilibrium region """
    global nx, ny, nz
    RR=0
    TH=1
    PH=2
    cachefname = "dumps/failfloorv2dudump%04d.npz" % floordumpno
    if os.path.isfile(cachefname):
        #if already pre-computed floor info, reuse it
        print("Reading %s..." % os.path.basename(cachefname))
        npzfile = np.load(cachefname)
        UfloorAin = npzfile['UfloorAin']
        UfloorAout = npzfile['UfloorAout']
        nx = npzfile['nx']
        ny = npzfile['ny']
        nz = npzfile['nz']
        return( UfloorAin, UfloorAout )
    #if no precomputed info
    rfloor( "failfloordudump%04d.bin" % floordumpno )
    #add back in rest-mass energy to conserved energy
    dUfloor[1] -= dUfloor[0]
    condin = np.ones_like(dUfloor)
    condin[0] = (avg_uu[1]<0)*(r[:,:,0:1]<maxrinflowequilibrium)
    condin[1] = (avg_uu[1]<0)*(r[:,:,0:1]<maxrinflowequilibrium)
    # condin[0] = (avg_rhouu[1]<0)*(r[:,:,0:1]<maxrinflowequilibrium)
    # condin[1] = (-avg_TudMA[1,0]<0)*(r[:,:,0:1]<maxrinflowequilibrium)
    #condin[1] = ((avg_rhouu+gam*avg_uguu)[1]<0)*(r[:,:,0:1]<maxrinflowequilibrium)
    condin[2:]=condin[0:1]
    #uncomment this if don't want to use stagnation surface
    #condin = (r[:,:,0:1]<maxrinflowequilibrium)
    condout = 1 - condin
    #XXX change below to account for limited range in theta
    #Integrate in radius
    UfloorAout = (dUfloor*condout[:,:,:,:]).sum(1+PH)  #*(tj!=0)*(tj!=ny-1)
    UfloorAin = (dUfloor*condin[:,:,:,:]).sum(1+PH) #*(tj!=0)*(tj!=ny-1)
    if not os.path.isfile(cachefname):
        np.savez(cachefname, UfloorAin=UfloorAin, UfloorAout=UfloorAout, nx=nx, ny=ny, nz=nz)
    return( UfloorAin, UfloorAout )

def plotfluxes(doreload=1,aphi_j_val=0):
    global DF,DF1,DF2,qtymem,qtymem1,qtymem2
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
    plt.figure(4)
    gs = GridSpec(2, 2)
    gs.update(left=0.09, right=0.94, top=0.95, bottom=0.1, wspace=0.01, hspace=0.04)
    ax1 = plt.subplot(gs[-2,-1])
    os.chdir("/home/atchekho/run/rtf2_15r34_2pi_a0.99gg500rbr1e3_0_0_0") 
    if not doreload:
        DF=DF1
        qtymem=qtymem1
    takeoutfloors(fti=7000,ftf=30500,#fti=22095,ftf=28195,
        ax=ax1,dolegend=False,doreload=doreload,plotldtot=False,lw=2,aphi_j_val=aphi_j_val)
    if doreload:
        DF1=DF
        qtymem1=qtymem
    plt.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/10., 
             0.85*ax1.get_ylim()[1], r"$(\mathrm{b})$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    # plt.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/2., 
    #          0.85*ax1.get_ylim()[1], r"$a=%g$" % a, size=20, rotation=0.,
    #          ha="center", va="center",
    #          color='k',weight='regular',bbox=bbox_props
    #          )
    plt.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/2., 
             -7, r"$\langle F_E\rangle<0$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular'#,bbox=bbox_props
             )
    plt.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/2., 
             12, r"$\langle F_M\!\rangle$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular'#,bbox=bbox_props
             )
    # ax1r = ax1.twinx()
    # ax1r.set_ylim(ax1.get_ylim())
    # ax1r.set_yticks((ymax/2,ymax))
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(16)
    plt.setp( ax1.get_yticklabels(), visible=False )
    ax1.set_ylim((-19,19))
    ax2 = plt.subplot(gs[-2,-2])
    os.chdir("/home/atchekho/run/rtf2_15r34.475_a0.5_0_0_0") 
    if not doreload:
        DF=DF2
        qtymem=qtymem2
    takeoutfloors(fti=10300,ftf=1e5,ax=ax2,dolegend=False,doreload=doreload,plotldtot=False,lw=2)
    if doreload:
        DF2=DF
        qtymem2=qtymem
    plt.text(ax2.get_xlim()[0]+(ax2.get_xlim()[1]-ax2.get_xlim()[0])/10., 
             0.85*ax2.get_ylim()[1], r"$(\mathrm{a})$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    # plt.text(ax2.get_xlim()[0]+(ax2.get_xlim()[1]-ax2.get_xlim()[0])/2., 
    #          0.85*ax2.get_ylim()[1], r"$a=%g$" % a, size=20, rotation=0.,
    #          ha="center", va="center",
    #          color='k',weight='regular',bbox=bbox_props
    #          )
    plt.text(ax2.get_xlim()[0]+(ax2.get_xlim()[1]-ax2.get_xlim()[0])/2., 
             5.8, r"$\langle F_E\rangle>0$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular'#,bbox=bbox_props
             )
    plt.text(ax2.get_xlim()[0]+(ax2.get_xlim()[1]-ax2.get_xlim()[0])/2., 
             15, r"$\langle F_M\!\rangle$", size=20, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular'#,bbox=bbox_props
             )
    ax2.set_ylim((-19,19))
    ax2.set_ylabel("Fluxes",fontsize=20,ha="left",labelpad=10)
    #ax2r = ax2.twinx()
    #ax2r.set_ylim(ax2.get_ylim())
    for label in ax2.get_xticklabels() + ax2.get_yticklabels(): #+ ax2r.get_yticklabels():
        label.set_fontsize(16)
    plt.savefig("fig4.eps",bbox_inches='tight',pad_inches=0.02)

def get_dFfloor(Dt, Dno, dotakeoutfloors=True,aphi_j_val=0, ndim=1, is_output_cell_center = True):
    """ Returns the (scaled to full wedge) flux correction due to floor activations and fixups, 
    requires gdump to be loaded [grid3d("gdump.bin",use2d=True)], and arrays, Dt and Dno, 
    set up."""
    #initialize with zeros
    global nx, ny, nz
    cachefname = "dumps/floorv2info.npz"
    #r-,th-, and phi- indices
    RR=0
    TH=1
    PH=2
    DT = 0
    DF = None
    if dotakeoutfloors:
        if os.path.isfile(cachefname):
            #if previously cached floor info, reuse it
            print("Reading %s..." % os.path.basename(cachefname))
            npzfile = np.load(cachefname)
            npzfile_Dt = npzfile['Dt']
            npzfile_Dno = npzfile['Dno']
            npzfile_DUin = npzfile['DUin']
            npzfile_DUout = npzfile['DUout']
            nx = npzfile['nx']
            ny = npzfile['ny']
            nz = npzfile['nz']
            if( Dt.shape == npzfile_Dt.shape and (Dt == npzfile_Dt).all() and
                Dno.shape == npzfile_Dno.shape and (Dno == npzfile_Dno).all() ):
                DUin = np.copy( npzfile_DUin )
                DUout = np.copy( npzfile_DUout )
                DF = DUin, DUout
                del npzfile_Dt
                del npzfile_Dno
                del npzfile_DUin
                del npzfile_DUout
                del npzfile
                #gc.collect()
            else:
                print( "Floor information (Dt or Dno) has changed since last time, skipping cache file (%s)\n" % cachefname )
        if DF is None:
            for (i,iDT) in enumerate(Dt):
                gc.collect() #try to clean up memory if not used
                iDUin, iDUout = get_dUfloor( Dno[i], aphi_j_val=aphi_j_val )
                #
                if iDT > 0:
                    DT += iDT
                if i==0:
                    DUin = iDUin
                    DUout = iDUout
                else:
                    DUin += iDUin * np.sign(iDT)
                    DUout += iDUout * np.sign(iDT)
            #average in time
            DUin /= DT
            DUout /= DT
            DUin *= scaletofullwedge(1.)
            DUout *= scaletofullwedge(1.)
            #save the floor info as cache file
            np.savez(cachefname, Dt=Dt, Dno=Dno, DUin=DUin, DUout=DUout, nx=nx, ny=ny, nz=nz)
            #at this point, don't know DF, so place two components of correction instead
            DF = DUin, DUout
        if ndim == 1:
            #convert DUin/DUout into DF
            DFout = DUout.cumsum(1+RR)
            DFin  = DUin.cumsum(1+RR)
            #
            DF = (DFin-DFin[:,nx-1:nx]) + DFout
            #
            if is_output_cell_center:
                #This needs to be moved half a cell to the right for correct centering
                DF[:,1:] = 0.5*(DF[:,:-1]+DF[:,1:])
            DF = DF.cumsum(1+TH)  #*(tj!=0)*(tj!=ny-1)
            if aphi_j_val == 0:
                #use unrestricted (full) sum
                DF = DF[:,:,ny-1]
            else:
                DF = cutout_along_aphi(DF,aphi_j_val=aphi_j_val)
    else:
        if ndim == 1:
            DF = np.zeros((8,nx),dtype=np.float64)
        elif ndim == 2:
            DF = np.zeros((8,nx,ny),dtype=np.float64)
        else:
            DF = np.zeros((8,nx,ny,nz),dtype=np.float64)
    return( DF )


def takeoutfloors(ax=None,doreload=1,dotakeoutfloors=1,dofeavg=0,fti=None,ftf=None,isinteractive=1,returndf=0,dolegend=True,plotldtot=True,lw=1,plotFem=False,writefile=True,doplot=True,aphi_j_val=0, ndim=1, is_output_cell_center = True, correct99 = False,**kwargs):
    global dUfloor, etad0, DF
    #Mdot, E, L
    grid3d("gdump.bin",use2d=True)
    #get base name of the current dir
    bn = os.path.basename(os.getcwd())
    pn = bn
    rbr = 100.
    rin=15
    Dt = None
    Dno = None
    betamin = 100
    qtymemloc = kwargs.pop('qtymem', None)
    if np.abs(a - 0.99)<1e-4 and bn=="rtf2_10r22.82_a0.99_n4_0_0_0":
        #lo-res 0.99 settings
        print( "Using a = 0.99 (rtf2_10r22.82_a0.99_n4_0_0_0) settings")
        dt = 100.
        Dt = np.array([11400.-9845.48387175465,
                       9800.-8071.47359453292,
                       8000.-7000.,
                       -(8000.-7000.)])
        Dno = np.array([114,
                        98,
                        80,
                        70])
        lfti = 7000.
        lftf = 20000.
        pn = "A0.99R10" 
        rin = 10
        rmax = 22.82
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) < 1.5:
        #hi-res 0.99 settings
        print( "Using hires a = 0.99 settings")
        #DTd = 22800-22231.9647756934 + 22200-22167.669585504507268 #22119.452438349220756
        dt = 100.
        # Dt = np.array([dt,-dt])
        # Dno = np.array([224.,223.])
        # fti = 14700.
        # ftf = 25000.
        Dt = np.array([28200-28097.4708711805,
                       28000-27406.4732593203,
                       27400-26763.9946654502,
                       26700-26330.0889135128,
                       26300-25799.8775611997,
                       25700-25124.6341346588,
                       25100-24594.4658011928,
                       24500-23951.5226133435,
                       23900-23292.5857662206,
                       23200-22890.8671337456,
                       22800-22231.9647756934,
                       22200-22167.6695855045])
        Dno = np.array([282,
                        280,
                        274,
                        267,
                        263,
                        257,
                        251,
                        245,
                        239,
                        232,
                        228,
                        222])
        pn="A0.99fc"
        # Dt = np.array([10800-10000,
        #                -(10800-10000)])
        # Dno = np.array([108,
        #                 100])
        # Dt = np.array([15100.-14674.9425787851])
        # Dno = np.array([151])
        # Dt = np.array([15800.-15167.5967825024])
        # Dno = np.array([158])
        # Dt = np.array([20700.-20000.,
        #                -(20700.-20000.)])
        # Dno = np.array([207.,
        #                 200.])
        # Dt = np.array([26300-25799.8775611997])
        # Dno = np.array([263])
        # Dt = np.array([26300-25800,
        #                -(26300-25800)])
        # Dno = np.array([263,
        #                 258])
        # lfti = 25800.
        # lftf = 26300.
        lfti = 7000.
        lftf = 30500.
        rbr = 1000.
        rin = 15
        rmax = 34
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) > 1.5:
        #lo-res 0.99 settings
        print( "Using lores a = 0.99 settings")
        dt = 100.
        Dt = np.array([13700-11887.3058391312,
                       11800.-11547.5107224568,
                       11500-9727.2561911212,
                       9700-8435.61370926043,
                       8400-6593.28942686595])
        Dno = np.array([137,
                        118,
                        115,
                        97,
                        84])
        # Dt = np.array([13700-11887.3058391312])
        # Dno = np.array([137])
        lfti = 6000.
        lftf = 1.e5
        pn="A0.99"
        rin = 15
        rmax = 34
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_20r45.35_0_0_0":
        print( "Using a = 0.9 (rtf2_20r45.35_0_0_0) settings")
        Dt = np.array([15100.-13152.0607139353,
                       13000.-10814.6119443408,
                       (10800.-9900.),
                       -(10800.-9900.)])
        Dno = np.array([151,
                        130,
                        108,
                        99])
        lfti = 8000.
        lftf = 15695.
        pn="A0.9R20"
        rin = 20
        rmax = 45.35
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_pi_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_pi_0_0_0) settings")
        Dt = np.array([
                       19800.-18629.2371784244,
                       18600.-18175.097618259,
                       18100.-17627.2495379077,
                       17600.-17047.4200574454,
                       17000.-16797.2958418814,
                       #16719.1587139405
                       #16700.-16672.2798069042,
                       16600.-15685.5761992012,
                       15600.-13898.007844829,
                       13800.-12927.303703666,
                       12900.-11126.259907352,
                       11100.-9337.26983629822,
                       9300.-8000.,
                       -(9300.-8000.)])
        Dno = np.array([198,
                        186,
                        181,
                        176,
                        170,
                        #167,
                        166,
                        156,
                        138,
                        129,
                        111,
                        93,
                        80])
        # lfti = 8000.
        # lftf = 15695.
        lfti = 8000.
        lftf = 1.e5
        pn="A0.9f"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0_2xr_newdiagkra":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0_2xr_newdiagkra) settings")
        Dt = np.array([16200-14876.761363014,
                       14800-13491.2552634378,
                       13400-12179.7086440425,
                       12100-10828.5870873105,
                       10800-9707.41586935387,
                       9700-9485.91473253505,
                       9400-8113.52370529932,
                       8110-8000.,
                      -(8110-8000.)
                       ])
        Dno = np.array([162,
                        148,
                        134,
                        121,
                        108,
                        97,
                        94,
                        81,
                        80
                        ])
        # lfti = 8000.
        # lftf = 15695.
        lfti = 8000.
        lftf = 1.e5
        pn="A0.9$h_r$"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_betax0.5_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_betax0.5_0_0_0) settings")
        # Dt = np.array([24400.-22382.7667797495,
        #                22300.-19989.7763978112,
        #                19900.-19551.4693631323,
        #                19500.-17262.4788610232,
        #                17200.-14969.7954456044,
        #                14900.-12645.0462217411,
        #                12600.-10447.0306188678,
        #                10400.-10000.,
        #              -(10400.-10000.)])
        # Dno = np.array([244,
        #                 223,
        #                 199,
        #                 195,
        #                 172,
        #                 149,
        #                 126,
        #                 104,
        #                 100])
        # lfti = 10000.
        # lftf = 50000.
        Dt = np.array([24400.-22382.7667797495,
                       22300.-19989.7763978112,
                       19900.-19551.4693631323,
                       19500.-17262.4788610232,
                       17200.-15000.,
                     -(17200.-15000.)])
        Dno = np.array([244,
                        223,
                        199,
                        195,
                        172,
                        150])
        lfti = 16000.
        lftf = 50000.
        pn="A0.9N200"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
        betamin = 200
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_betax2_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_betax2_0_0_0) settings")
        Dt = np.array([14300.-12041.7226584439,
                       12000.-9647.13921353543,
                       9600.-7345.22404370437,
                       7300.-6000.,
                     -(7300.-6000.)])
        Dno = np.array([143,
                        120,
                        96,
                        73,
                        60])
        lfti = 8000.
        lftf = 50000.
        pn="A0.9N50"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
        betamin = 50
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_betax4_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_betax4_0_0_0) settings")
        Dt = np.array([17300.-15195.9206754056,
                       15100.-12862.478404495,
                       12800.-10435.1795933295,
                       10400.-8164.22610799513,
                       8100.-6000.,
                     -(8100.-6000.)])
        Dno = np.array([173,
                        151,
                        128,
                        104,
                        81,
                        60])
        # Dt = np.array([17300.-15195.9206754056,
        #                15100.-12862.478404495,
        #                12800.-10435.1795933295])
        # Dno = np.array([173,
        #                 151,
        #                 128])
        lfti = 8000.
        lftf = 50000.
        pn="A0.9N25"
    # elif np.abs(a - (-0.9))<1e-4:
    #     print( "Using a = -0.9 settings")
    #     Dt = np.array([12300.-10106.5605346438,
    #                    10100.-8039.58198814953,
    #                    8000.-7000.,
    #                   -(8000.-7000.)])
    #     Dno = np.array([123,
    #                     101,
    #                     80,
    #                     70])
    #     lfti = 7000.
    #     lftf = 20000.
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
        betamin = 25
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r34.1_0_0_0_spinflip":
        #rtf2_15r37.1a-0.9_0_0_0
        #with less failfloordudumps:
        print( "Using a = -0.9 (rtf2_15r34.1_0_0_0_spinflip) settings")
        Dt = np.array([18700.0-16478.4789523558,
                       16400.0-14207.0184709617])
        Dno = np.array([187,
                        164])
        lfti = 14207.
        lftf = 20000.
        pn="A-0.9flip"
        rin = 15
        rmax = 34.1
        simti = lfti
        simtf = lftf
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r37.1a-0.9_0_0_0":
        #rtf2_15r37.1a-0.9_0_0_0
        #with less failfloordudumps:
        print( "Using a = -0.9 (rtf2_15r37.1a-0.9_0_0_0) settings")
        Dt = np.array([12300.-10106.5605346438,
                       10100.-8039.58198814953,
                       8000.-7000.,
                      -(8000.-7000.)])
        Dno = np.array([123,
                        101,
                        80,
                        70])
        lfti = 8000.
        lftf = 1.e5
        pn="A-0.9"
        rin = 15
        rmax = 37.1
        simti = 0
        simtf = lftf
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r37.1a-0.9_2xphi_0_0_0":
        #with less failfloordudumps:
        print( "Using a = -0.9 (rtf2_15r37.1a-0.9_2xphi_0_0_0) settings")
        Dt = np.array([14700.-13022.9649275961,
                       13000.-11322.912779019,
                       11300.-9622.93054762303,
                       9600.-8000.00574860238])
        Dno = np.array([146,
                        129,
                        112,
                        95])
        lfti = 8000.
        lftf = 1.e5
        pn="A-0.9$h_\\varphi$"
        rin = 15
        rmax = 37.1
        simti = 8000.
        simtf = lftf
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r37.1a-0.9_lr_0_0_0":
        print( "Using a = -0.9 (rtf2_15r37.1a-0.9_lr_0_0_0) settings")
        Dt = np.array([17500.-15712.9370781614,
                        12000.-10208.5400456126,
                        10200.-10000.,
                        -(10200.-10000.)])
        Dno = np.array([175,
                       120,
                       102,
                       100])
        lfti = 8000.
        lftf = 1.e5
        pn="A-0.9$l_\\theta$"
        rin = 15
        rmax = 37.1
        simti = 0
        simtf = lftf
    elif np.abs(a - (-0.5))<1e-4 and bn == "rtf2_15r36.21_a-0.5_0_0_0":
        print( "Using a = -0.5 (rtf2_15r36.21_a-0.5_0_0_0) settings")
        Dt = np.array([14200-13393.5929462345,
                       13300-11452.1038814141,
                       11400-8786.54757354983,
                        8700-8000.,
                      -(8700-8000.)])
        Dno = np.array([142,
                        133,
                        114,
                        87,
                        80])
        lfti = 10000.
        lftf = 20000.
        pn="A-0.5"
        simti = 0
        simtf = lftf
        rin=15
        rmax=36.21
    elif np.abs(a - (-0.2))<1e-4 and bn == "rtf2_15r35.64_a-0.2_0_0_0":
        print( "Using a = -0.5 (rtf2_15r35.64_a-0.2_0_0_0) settings")
        Dt = np.array([15100-12221.0353104236,
                       12200-10000.,
                     -(12200-10000.)])
        Dno = np.array([151,
                        122,
                        100])
        lfti = 10000.
        lftf = 20000.
        pn="A-0.2"
        rin = 15
        rmax = 35.64
        simti = 0
        simtf = lftf
    elif np.abs(a + 0.9)<1e-4 and bn == "rtf2_15r37.1a-0.9_0_0_0_0.5xr_newdiagkra":
        print( "Using a = -0.9 (rtf2_15r37.1a-0.9_0_0_0_0.5xr_newdiagkra) settings")
        Dt = np.array([18700-13805.9605171409,
                       13800-8901.06032520801,
                       8900-8000.,
                     -(8900-8000.)])
        Dno = np.array([187,
                        138,
                        89,
                        80])
        lfti = 8000.
        lftf = 1.e5
        pn="A-0.9$l_r$"
        rin = 15
        rmax = 37.1
        simti = 0
        simtf = lftf
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r37.1a-0.9_0_0_0_2xth":
        print( "Using a = -0.9 (rtf2_15r37.1a-0.9_0_0_0_2xth) settings")
        Dt = np.array([16500.-14770.1296385559,
                       14700.-13021.5966066291,
                       13000.-12840.6076022518,
                       12800.-12413.742287205,
                       12400.-12328.5640517402])
        Dno = np.array([165,
                        147,
                        130,
                        128,
                        124])
        lfti = 12328.
        lftf = 1.e5
        pn="A-0.9$h_\\theta$"
        rin = 15
        rmax = 37.1
        simti = 12328.
        simtf = lftf
    elif np.abs(a - (-0.9))<1e-4 and bn == "rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2":
        print( "Using a = -0.9 (rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2) settings")
        Dt = np.array([20000.-18923.6303395794,
                       18900.-17690.7770842922,
                       #17600.-17526.7920937424,
                       17500.-16525.8251918482,
                       16500.-15364.1834020439,
                       15300.-14177.7085736113,
                       14100.-13223.7286827588,
                       13200.-12200.2252164064,
                       12200.-11035.9098660778,
                       11000.-9895.29490728404,
                       9800.-8778.69431599574,
                       8700.-8000.,
                       -(8700.-8000.)])
        Dno = np.array([200,
                        189,
                        #176,
                        175,
                        165,
                        153,
                        141,
                        132,
                        122,
                        110,
                        98,
                        87,
                        80])
        lfti = 8000.
        lftf = 1.e5
        pn="A-0.9f"
        rbr = 1000.
        rin = 15
        rmax = 37.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf_15r34.1_0_0_0":
        print( "Using a = 0.9 (rtf_15r34.1_0_0_0) settings")
        Dt = np.array([15800.-15685.1591357819,
                       15600.-13413.903124605,
                       13400.-11142.7284161051,
                       11100.-8692.5709730908,
                       8600.-8000.,
                     -(8600.-8000.)])
        Dno = np.array([158,
                        156,
                        134,
                        111,
                        86,
                        80])
        lfti = 8000.
        lftf = 20000.
        pn="A0.9old"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_2xphi_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_2xphi_0_0_0) settings")
        Dt = np.array([11600-9999.2977584592,
                       9900-8355.23482702302])
        Dno = np.array([116,
                        99])
        lfti = 8000.
        lftf = 20000.
        pn="A0.9$h_\\varphi$"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_4xphi_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_4xphi_0_0_0) settings")
        Dt = np.array([14600-13552.905360015,
                       13500-12470.3849071082,
                       12400-11764.8381755944,
                       11700-10691.9735460019,
                       10600-9590.1113948539,
                       9500-8500.00310181627])
        Dno = np.array([145,
                        134,
                        123,
                        116,
                        105,
                        94])
        lfti = 8500.
        lftf = 100000.
        pn="A0.9$h^2_\\varphi$"
        rin = 15
        rmax = 34.1
        simti = 8500
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0_0.5xr_newdiagkra":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0_0.5xr_newdiagkra) settings")
        Dt = np.array([20300-16800.3973115124,
                       15500-15245.7891969861,
                       15200-12104.8225887105,
                       12100-9814.17250577102,
                       9800-8000.,
                     -(9800-8000.)])
        Dno = np.array([203,
                        155,
                        152,
                        121,
                        98,
                        80])
        lfti = 8000.
        lftf = 1.e5
        pn="A0.9$l_r$"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0) settings")
        Dt = np.array([22200.-19991.2576631864,
                       19900.-19520.2475545571,
                       19500.-17174.3949708944,
                       17100.-15198.3870293434,
                       15100.-14207.0184709617,
                       14200.-11819.493630548,
                       11800.-9525.17879311185,
                       9500.-8281.06561787569,
                       8200.-8000.,
                     -(8200.-8000.)])
        Dno = np.array([222,
                        199,
                        195,
                        171,
                        151,
                        142,
                        118,
                        95,
                        82,
                        80])
        lfti = 8000.
        lftf = 1e5
        pn="A0.9"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_lr_0_0_0":
        print( "Using a = 0.9 (rtf2_15r34.1_lr_0_0_0) settings")
        Dt = np.array([13900.-11092.5104022445,
                       11000.-10000.,
                       -(11000.-10000.)])
        Dno = np.array([139,
                        110,
                        100])
        lfti = 8000.
        lftf = 1e5
        pn="A0.9$l_\\theta$"
        rin = 15
        rmax = 34.1
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0_2xth":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0_2xth) settings")
        Dt = np.array([#21000.-19516.4625795943,
                       19500.-18404.7747547703,
                       18400.-16641.8415831742,
                       16500.-14875.0849054248,
                       14800.-14701.7703617868,
                       14700.-14292.8766863536])
        Dno = np.array([#210,
                        195,
                        184,
                        165,
                        148,
                        147])
        #lfti = 17500.
        #lfti = 14215.
        lfti = 14207.
        lftf = 1e5
        pn="A0.9$h_\\theta$"
        rin = 15
        rmax = 34.1
        simti = 14207.
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0_2xth2xphi":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0_2xth2xphi) settings")
        # Dt = np.array([#18243.1710113689,
        #                18200.-17725.310836,
        #                17700.-17179.8585982052,
        #                #17113.1745686056,
        #                17100.-16913.0491381982,
        #                16900.-16712.9233648069,
        #                16700.-16158.5902847868,
        #                16100.-15958.5669463911,
        #                15900.-15758.5456875678,
        #                15700.-15558.4981952456,
        #                15500.-15358.3836961663,
        #                15300.-14804.1559946433,
        #                14800.-14600.0030362192,
        #                14600.-14539.327413648,
        #                #14530.093538046,
        #                14500.-14207.0184709617])
        # Dno = np.array([181, #[from dumps/ dir
        #                 176, # ...
        #                 170, # ...
        #                 168, # ...
        #                 166, # ...
        #                 160, # ...
        #                 158, # ...
        #                 156, # ...
        #                 154, # ...
        #                 152, # ...
        #                 147, # from dumps/ dir]
        #                 146, #[from dumps/fldbackup/ dir]
        #                 145])#[from dumps/ dir]
        Dt = np.array([19400.-18976.9614912956,
                       18900.-18881.6126574549,
                       18800.-18418.9418783789
                       #18243.1710113689,
                       #18200.-17725.310836# ,
                       #17700.-17179.8585982052 #,
                       # #17113.1745686056,
                       # 17100.-16913.0491381982,
                       # 16900.-16712.9233648069,
                       # 16700.-16158.5902847868,
                       # 16100.-15958.5669463911,
                       # 15900.-15758.5456875678,
                       # 15700.-15558.4981952456,
                       # 15500.-15358.3836961663,
                       # 15300.-14804.1559946433,
                       # 14800.-14600.0030362192,
                       # 14600.-14539.327413648,
                       # #14530.093538046,
                       # 14500.-14207.0184709617
                       ])
        Dno = np.array([193,
                        188,
                        187
                #181# , #[from dumps/ dir
                        # 176 #, ...
                        # 170, # ...
                        # 168, # ...
                        # 166, # ...
                        # 160, # ...
                        # 158, # ...
                        # 156, # ...
                        # 154, # ...
                        # 152, # ...
                        # 147, # from dumps/ dir]
                        # 146, #[from dumps/fldbackup/ dir]
                        # 145
                        ])#[from dumps/ dir]
        #Dt*=1.5 #do this in order to roughly get flat curves for range of data 14207-18420  (using floor info 18420-19300); barely changes efficiency (Dt*=1: eta=90.3%; Dt*=1.5: eta=90.6%)
        #nz=64
        #_dx3=0.5/64.
        lfti=14207.
        #lfti=17000.
        #lfti = 18420.
        lftf = 1e5
        pn="A0.9$h_{\\theta}h_{\\varphi}$"
        rin = 15
        rmax = 34.1
        #simti = 14207.
        simti = lfti
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_0_0_0_2xphi_newdiagnau":
        print( "Using a = 0.9 (rtf2_15r34.1_0_0_0_2xphi_newdiagnau) settings")
        Dt = np.array([22500.-21392.5177548623,
                       21300.-20902.3522686028,
                       20900.-19774.6269601892,
                       19700.-18620.2226337403,
                       18600.-18434.8036279833,
                       18400.-17388.3330474263,
                       17300.-17106.484654906,
                       17100.-16650.1119902541,
                       #16623.2713347645
                       16600.-16368.2447692806,
                       #16234.0117336642
                       #16207.1697816043
                       #16193.7487169041
                       #16140.058900599
                       16100.-15334.5561626554,
                       #15300.-15227.1533381102,
                       15200.-14998.9547710076,
                       #14972.1071483046
                       #14945.2597480547
                       #14918.4142929998
                       #14891.5677858923
                       14800.-14207.0184709617
                       ])
        Dno = np.array([225,
                        213,
                        209,
                        197,
                        186,
                        184,
                        173,
                        171,
                        166,
                        161,
                        #153,
                        152,
                        148
                        ])#[from dumps/ dir]
        lfti=14207.
        lftf = 1e5
        pn="A0.9$h_{\\varphi}$"
        rin = 15
        rmax = 34.1
        #simti = 14207.
        simti = lfti
        simtf = lftf
    elif np.abs(a - 0.9)<1e-4 and bn == "rtf2_15r34.1_betax0.5_0_0_0_2xphi_restart15000":
        print( "Using a = 0.9 (rtf2_15r34.1_betax0.5_0_0_0_2xphi_restart15000) settings")
        Dt = np.array([23200-21626.7045853424,
                       21600-21180.2758247238,
                       21100-19607.2567643133,
                       19600-17970.484301684,
                       17900-17332.6888281801,
                       17300-16673.3266984864,
                       16600-15000.0049914467
                       ])
        Dno = np.array([231,
                        215,
                        210,
                        195,
                        178,
                        172,
                        165
                        ])#[from dumps/ dir]
        #lfti=14207.
        lfti=16000.
        #lfti = 18420.
        lftf = 1e5
        pn="A0.9N200$h_\\varphi$"
        rin = 15
        rmax = 34.1
        #simti = 14207.
        simti = 15000
        simtf = lftf
        betamin=200
    elif np.abs(a - 0.5)<1e-4:
        print( "Using a = 0.5 settings")
        dt1 = 13000.-10279.
        dt2 = 10200.-10000.
        Dt = np.array([
                19500-17592.5204772097,
                17500-16952.0863127404,
                16900-15010.5538754437,
                15000-13069.8145093629,
                dt1,
                dt2,
                -dt2])
        Dno = np.array([
                195,
                175,
                169,
                150,
                130,
                102,
                100])
        pn="A0.5"
        lfti = 10000.
        lftf = 1e5
        rin = 15
        rmax = 34.475
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.2)<1e-4:
        print( "Using a = 0.2 settings")
        Dt = np.array([19300-18061.8964506732,
                       18000-17406.5234843135,
                       17400-15396.4283777091,
                       15300-13364.19837589,
                       13300.-10366.5933313178])
        Dno = np.array([
                193,
                180,
                174,
                153,
                133])
        lfti = 10000.
        lftf = 1e5
        pn="A0.2"
        rin = 15
        rmax = 35
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.1)<1e-4:
        print( "Using a = 0.1 settings")
        Dt = np.array([20000.-18698.2882595555,
                       18600.-15552.4457340615,
                       15500.-12361.7572701913,
                       12300.-10000.,
                       -(12300.-10000.)])
        Dno = np.array([200,
                        186,
                        155,
                        123,
                        100])
        lfti = 10000.
        lftf = 20000.
        pn="A0.1"
        rin = 15
        rmax = 35
        simti = 0
        simtf = lftf
    elif np.abs(a - 0.0)<1e-4:
        print( "Using a = 0.0 settings")
        Dt = np.array([18500.-15700.2418591157,
                       15700.-12587.5162658605,
                       12500.-10000.,
                       -(12500.-10000.)])
        Dno = np.array([185,
                        157,
                        125,
                        100])
        lfti = 10000.
        lftf = 20000.
        pn="A0.0"
        rin = 15
        rmax = 35
        simti = 0
        simtf = lftf
    elif bn == "thickdisk7":
        print( "Using a = %g (thickdisk7) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A94BfN40"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    elif bn == "thickdisk8":
        print( "Using a = %g (thickdisk8) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A94BfN100C1"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    elif bn == "thickdisk11":
        print( "Using a = %g (thickdisk11) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 6000.
        lftf = 12000.
        pn="A94BfN100C2"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    elif bn == "thickdisk12":
        print( "Using a = %g (thickdisk12) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 6000.
        lftf = 1e5
        pn="A94BfN100C3"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    elif bn == "thickdisk13":
        print( "Using a = %g (thickdisk13) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A94BfN100C4"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    elif bn == "thickdiskrr2":
        print( "Using a = %g (thickdiskrr2) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A-0.94BfN30"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
        betamin = 30
        rbr = 500
    elif bn == "thickdiskr1":
        print( "Using a = %g (thickdiskr1) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A0.94BfN30"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
        betamin = 30
        rbr = 500
    elif bn == "thickdiskr2":
        print( "Using a = %g (thickdiskr2) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A0.94BfN30r"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
        betamin = 30
        rbr = 500
    elif bn == "thickdisk9":
        print( "Using a = %g (thickdisk9) settings" % a )
        Dt = None
        Dno = None
        #lfti = 14215.
        lfti = 8000.
        lftf = 1e5
        pn="A94BpN100"
        rin = 10
        rmax = 100
        simti = 0.
        simtf = lftf
        dotakeoutfloors=0
    else:
        print( "Unknown case: a = %g, using defaults..." % a )
        lfti = 10000.
        lftf = 20000.
        dotakeoutfloors = 0
        pn="Unknown"
        rin = -1
        rmax = -1
        simti = -1
        simtf = -1
    #os.path.basename(os.getcwd())
    if fti is None or ftf is None:
        fti = lfti
        ftf = lftf

    #dotakeoutfloors=False

    # if dotakeoutfloors:
    #     istag, jstag, hstag, rstag = getstagparams(rmax=20,doplot=0,doreadgrid=0)
    #dotakeoutfloors=1
    RR=0
    TH=1
    PH=2
    if doreload:
        #XXX this returns array of zeros if dotakeoutfloors == False or 0.
        DF = get_dFfloor(Dt, Dno, dotakeoutfloors=dotakeoutfloors,aphi_j_val=aphi_j_val, ndim=ndim, is_output_cell_center = is_output_cell_center)
    #trust resolution from floor information
    nxf=nx
    nyf=ny
    nzf=nz

    #RETURN: if requested 2D information
    if ndim == 2:
        DUin, DUout = DF
        #convert DUin/DUout into DF
        #DUin = DUin.cumsum(1+TH)  #*(tj!=0)*(tj!=ny-1)
        #DUout = DUout.cumsum(1+TH)  #*(tj!=0)*(tj!=ny-1)
        DFin  = DUin.cumsum(1+RR)
        DFout = DUout.cumsum(1+RR)
        #
        DF = (DFin-DFin[:,nx-1:nx]) + DFout
        #
        if is_output_cell_center:
            #This needs to be moved half a cell to the right for correct centering
            DF[:,1:] = 0.5*(DF[:,:-1]+DF[:,1:])
        return DF

    #RETURN: if requested raw 2D information
    if ndim == -2:
        return DF

    DFfloor0 = DF[0]
    DFfloor1 = DF[1]
    DFfloor4 = DF[4]

    if doreload:
        etad0 = -1/(-gn3[0,0])**0.5
        #!!!rhor = 1+(1-a**2)**0.5
        ihor = iofr(rhor)
        if qtymemloc is None:
            qtymem=getqtyvstime(ihor,0.2)
        else:
            qtymem = qtymemloc
        real_tf = qtymem[0,-1,0] #+(qtymem[0,-1,0]-qtymem[0,-2,0]) #fix simulation end time mismatch
        if ftf > real_tf and qtymem[0,-1,0] > 0:
            #last_t + dt:
            ftf = real_tf
        if simtf > real_tf and qtymem[0,-1,0] > 0:
            #last_t + dt:
            simtf = real_tf
    else:
        qtymem = qtymemloc

    #at this time we have the floor information, now get averages:
    #mdtotvsr, edtotvsr, edmavsr, ldtotvsr = plotqtyvstime( qtymem, whichplot = -2, fti=fti, ftf=ftf )
    #XXX

    if np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) < 1.5 and bn == "rtf2_15r34_2pi_a0.99gg500rbr1e3_0_0_0" and correct99:
        #face vs. center correction
        mdtotvsr, edtotvsr, edmavsr, ldtotvsr,\
                    mdotfinavgvsr5, mdotfinavgvsr10, mdotfinavgvsr20, mdotfinavgvsr30, mdotfinavgvsr40, \
                    pjemfinavgvsr5, pjemfinavgvsr10, pjemfinavgvsr20, pjemfinavgvsr30, pjemfinavgvsr40, \
                    pjmafinavgvsr5, pjmafinavgvsr10, pjmafinavgvsr20, pjmafinavgvsr30, pjmafinavgvsr40, \
                    fstotfinavg, fstotsqfinavg, \
                    pjke_mu2_avg, pjke_mu1_avg, \
                    gdetF10, gdetF11, gdetF12 \
                    = plotqtyvstime( qtymem, whichplot = -200, fti=31568.8571637753, ftf=1e5, aphi_j_val=aphi_j_val )
                    #= plotqtyvstime( qtymem, whichplot = -200, fti=32000, ftf=32500, aphi_j_val=aphi_j_val )
        F11 = -edtotvsr-mdtotvsr
        #properly center flux in a cell
        gdetF11c=np.copy(gdetF11)
        gdetF11c[:-1]=0.5*(gdetF11[:-1]+gdetF11[1:])
        energy_flux_correction_factor = gdetF11c/F11

    mdtotvsr, edtotvsr, edmavsr, ldtotvsr,\
                mdotfinavgvsr5, mdotfinavgvsr10, mdotfinavgvsr20, mdotfinavgvsr30, mdotfinavgvsr40, \
                pjemfinavgvsr5, pjemfinavgvsr10, pjemfinavgvsr20, pjemfinavgvsr30, pjemfinavgvsr40, \
                pjmafinavgvsr5, pjmafinavgvsr10, pjmafinavgvsr20, pjmafinavgvsr30, pjmafinavgvsr40, \
                fstotfinavg, fstotsqfinavg, \
                pjke_mu2_avg, pjke_mu1_avg, \
                gdetF10, gdetF11, gdetF12 \
                = plotqtyvstime( qtymem, whichplot = -200, fti=fti, ftf=ftf, aphi_j_val=aphi_j_val )

    horavg = plotqtyvstime( qtymem, whichplot = -199, fti=fti, ftf=ftf, aphi_j_val=aphi_j_val )
    
    if np.abs(a - 0.99)<1e-4 and scaletofullwedge(1.0) < 1.5 and bn == "rtf2_15r34_2pi_a0.99gg500rbr1e3_0_0_0" and correct99:
        #correct energy flux for face vs. center
        edtotvsr = (edtotvsr+mdtotvsr)*energy_flux_correction_factor - mdtotvsr
    elif True and (gdetF11!=0).any() and timeavg(qtymem[132],qtymem[0,:,0],fti,fti+1.).any():
        print( "takeoutfloors(): using gdetF11 and gdetF10 to compute edot and mdot" )
        edtotvsr[:-1] = -0.5*(gdetF11[:-1]+gdetF11[1:])
        mdtotvsr[:-1] = -0.5*(gdetF10[:-1]+gdetF10[1:])
        edtotvsr -= mdtotvsr
        
    FEMKE = -(edtotvsr+mdtotvsr)
    FKE = -(edmavsr+mdtotvsr)
    FKE10 = -((edmavsr-pjmafinavgvsr5) + mdotfinavgvsr5)

    #electromagnetic flux
    FEM=-(edtotvsr-edmavsr)

    if dotakeoutfloors == False:
        #Jon's method for when there are no floors available:
        #Remove matter contribution with b^2/rho < 20 and later evaluate fluxes at r = 5
        #Mass accretion rate (only inside b^2 < 20)
        mdtotvsr = mdotfinavgvsr20
        #Outward EM energy flux
        edemvsr = edtotvsr - edmavsr
        #Add the EM part and the low-magnetized MA part (that has bsq/rho<=20)
        #to get the *corrected* total energy flux
        edtotvsr = edtotvsr - pjmafinavgvsr20

    if dofeavg:
        FE=np.load("fe.npy")
    #edtotvsr-=FE
    #avgmem = get2davg(usedefault=1)
    #assignavg2dvars(avgmem)
    #edtotvsr = -(gdet[:,1:ny-1,0:1]*avg_Tud[1][0][:,1:ny-1,0:1]*_dx2*_dx3*nz).sum(-1).sum(-1)
    #!!!rhor = 1+(1-a**2)**0.5
    rh=rhor
    ihor = iofr(rhor)
    #FIGURE: mass
    if isinteractive:
        if ax is None and doplot:
            plt.figure(1)
            plt.clf()
        if ax is None and doplot:
            plt.plot(r[:,0,0],mdtotvsr,'b--',label=r"$F_M$ (raw)",lw=2)
    #pdb.set_trace()
    Fmuncorr=mdtotvsr
    Feuncorr=-edtotvsr
    Fm=(mdtotvsr+DFfloor0)
    Fe=-(edtotvsr+DFfloor1)
    if ldtotvsr is not None:
        #** definition of ldtot: \int(gdet*Tud[1][3]): 
        #   when u^r < 0 and u_\varphi > 0 (usual for prograde disk inflow), then ldtot < 0
        #defined as positive when ang. mom. flows *into* BH
        Fl=-(ldtotvsr+DFfloor4)
        Flphi=Fl/dxdxp[3][3][:,0,0]  #convert L_x3 into L_\varphi
        #spin-up parameter
        #taken from Gammie, Shapiro, McKinney 2003
        spar = (Flphi-2*a*Fe)/Fm
    else:
        #put obviously bad number in
        Fl=edtotvsr*0-1e11
        spar=edtotvsr*0-1e11
    if dotakeoutfloors:
        if isinteractive and doplot:
            plt.plot(r[:,0,0],Fm,'b',label=r"$F_M$",lw=2)
    if isinteractive and ax is None and doplot:
        plt.plot(r[:,0,0],-edtotvsr,'r--',label=r"$F_E$ (raw)",lw=2)
    if dofeavg and isinteractive and ax is None and doplot:
        plt.plot(r[:,0,0],FE,'k--',label=r"$F_E$",lw=2)
    if dotakeoutfloors:
        if isinteractive and doplot:
            plt.plot(r[:,0,0],Fe,'r',label=r"$F_E$",lw=2)
            if plotFem:
                #plt.plot(r[:,0,0],-FEM,'r--',label=r"$-F_{EM}$",lw=2)
                #plt.plot(r[:,0,0],Fm*0+Fm[iofr(5)]-Fe[iofr(5)],'r-.',label=r"$(F_{M}-F_{E})[r=5r_g]$",lw=2)
                plt.plot(r[:,0,0],-FEM+Fe[iofr(5)]-Fm[iofr(5)],'c',label=r"-$F_{EM}+F_{EMKE}(5)$",lw=2)
                plt.plot(r[:,0,0],FKE,'b',label=r"$F_{KE}$",lw=2)
                plt.plot(r[:,0,0],FEMKE,'m',label=r"$F_{EMKE}$",lw=2)
                plt.plot(r[:,0,0],FKE10,'g',label=r"$F_{KE,b^2/\rho<30}$",lw=2)
                #plt.plot(r[:,0,0],Fm+edtotvsr,'g:',label=r"$F_{EM}$",lw=2)
        if dofeavg and isinteractive and doplot: 
            plt.plot(r[:,0,0],FE-DFfloor1,'k',label=r"$F_E$",lw=2)
        if isinteractive and ax is None and doplot:
            plt.plot(r[:,0,0],(DFfloor1),'r:',lw=2)
    if ldtotvsr is not None and plotldtot and doplot:
        if isinteractive and ax is None:
            plt.plot(r[:,0,0],-ldtotvsr/dxdxp[3][3][:,0,0]/10.,'g--',label=r"$F_L/10$ (raw)",lw=2)
        if dotakeoutfloors and isinteractive and doplot:
            plt.plot(r[:,0,0],Fl/dxdxp[3][3][:,0,0]/10.,'g',label=r"$F_L/10$",lw=2)
    eta = ((Fm-Fe)/Fm)
    etap = (Fm-Fe)/Fe
    if isinteractive:
        print("Eff = %g, Eff' = %g" % ( eta[iofr(5)], etap[iofr(5)] ) )
        #plt.plot(r[:,0,0],DFfloor0,label=r"$dU^t$")
        #plt.plot(r[:,0,0],DFfloor*1e4,label=r"$dU^t\times10^4$")
        if doplot:
            if dolegend:
                plt.legend(loc='lower right',bbox_to_anchor=(0.97,0.39),
                           #borderpad = 1,
                           borderaxespad=0,frameon=True,labelspacing=0,
                           ncol=1)
            plt.xlim(rhor,19.99)
            plt.ylim(-10,18)
            plt.grid()
            plt.xlabel(r"$r\ [r_g]$",fontsize=20)
            if ax is None:
                plt.ylabel("Fluxes",fontsize=20,ha='center')
                plt.savefig("fig4.pdf",bbox_inches='tight',pad_inches=0.02)
                plt.savefig("fig4.eps",bbox_inches='tight',pad_inches=0.02)
                plt.savefig("fig4.png",bbox_inches='tight',pad_inches=0.02)
            #FIGURE: energy
            #plt.figure(2)
            #plt.plot(r[:,0,0],edtotvsr+DFfloor1,label=r"$\dot E+dU^1$")
            #plt.plot(r[:,0,0],DFfloor1,label=r"$dU^1$")
            #plt.legend()
            #plt.xlim(rhor,12)
            #plt.ylim(-3,20)
            #plt.grid()
    #
    corrfac = ((Fm-Fe)/(Fmuncorr-Feuncorr))[iofr(10)]
    corrabs =  ((Fm-Fe)-(Fmuncorr-Feuncorr))[iofr(10)]
    if writefile:
        #assume that eta is cross-correlated on shorter time scales than this
        dtavgmin = 500.
        dtavg = min((ftf-fti)/3.,dtavgmin)
        nmax = np.rint((ftf-fti)/dtavg)
        rx = 5
        rj = 100
        etamean = eta[iofr(rx)]
        sparmean = spar[iofr(rx)]
        ####################
        #
        #   phi
        #
        unitsfactor=(4*np.pi)**0.5*2*np.pi
        #phibh=fstot[:,ihor]/4/np.pi/FMavg**0.5*unitsfactor
        #where fstot = (gdetB1).sum(2).sum(1)*_dx2*_dx3 at horizon
        phimean = fstotsqfinavg/4/np.pi/Fm[iofr(rx)]**0.5*unitsfactor
        #
        #############
        etastd, sparstd, phistd, pjstd, pwstd = 0, 0, 0, 0, 0
        for nint in np.arange(2,nmax+1):
            etameann, etastdn, sparmeann, sparstdn, phimeann, phistdn, pjmeann, pjstdn, pwmeann, pwstdn = \
                computeeta(start_t=fti,end_t=ftf,numintervals=nint,doreload=0,qtymem=qtymem,rj=rj)
            etastd = max( etastd, etastdn )
            sparstd = max( sparstd, sparstdn )
            phistd = max( phistd, phistdn )
            pjstd = max( pjstd, pjstdn )
            pwstd = max( pwstd, pwstdn )
        pj_uncorr = pjke_mu2_avg[iofr(rj)]
        pw_uncorr = (pjke_mu1_avg-pjke_mu2_avg)[iofr(rj)]
        #Assume all correction goes into jet per
        pj = pj_uncorr + corrabs
        pw = pw_uncorr
        #
        # OUTPUT for plotting
        #
        foutpower = open( "siminfo_%s.txt" %  os.path.basename(os.getcwd()), "w" )
        #foutpower.write( "#Name a Mdot   Pjet    Etajet  Psitot Psisqtot**0.5 Psijet Psisqjet**0.5 rstag Pjtotmax Pjtot1rstag Pjtot2rstag Pjtot4rstag Pjtot8rstag\n"  )
        foutpower.write( "%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (
                                                           pn, os.path.basename(os.getcwd()), a, 
                                                           etamean, etastd, sparmean, sparstd, phimean, phistd,
                                                           Fm[iofr(rx)], Fe[iofr(rx)], Fl[iofr(rx)]/dxdxp[3][3][0,0,0],
                                                           FEM[iofr(rhor)], FEM[iofr(2)],
                                                           pj, pjstd, 
                                                           pw, pwstd,
                                                           fstotfinavg, fstotsqfinavg,
                                                           horavg[iofr(5)], horavg[iofr(10)], horavg[iofr(20)], 
                                                           horavg[iofr(25)], horavg[iofr(30)], horavg[iofr(100)]) )
        #flush to disk just in case to make sure all is written
        foutpower.flush()
        os.fsync(foutpower.fileno())
        foutpower.close()
        #
        # LATEX output for table
        #
        foutpower = open( "simtex_%s.txt" %  os.path.basename(os.getcwd()), "w" )
        if scaletofullwedge(1.) > 1.5:
            swedge = "\\pi"
        else:
            swedge = "2\\pi"
        #foutpower.write( "#Name a Mdot   Pjet    Etajet  Psitot Psisqtot**0.5 Psijet Psisqjet**0.5 rstag Pjtotmax Pjtot1rstag Pjtot2rstag Pjtot4rstag Pjtot8rstag\n"  )
        #                 Name Spin    Resolution          Phi-wedge  rbr    eta+-deta     etaj   etaw   flux      simname
        #pdb.set_trace()
        #corrfac corrects jet power for floor effects (assumes that all floor effects go into jet and not into wind)
        etajet = pj/Fm[iofr(rx)]
        etawind = pw/Fm[iofr(rx)]
        foutpower.write( "%15s & $%g$ &\t $%d\\pm%d$ &\t $%d$ &\t $%s$ &\t $%d\\times%d\\times%d$ &\t $%d$ &\t $%g$ &\t $%g$ &\t $%g$ &\t $%d$ &\t $(%d; %d)$ &\t $(%d; %d)$ \\\\ %% %s\n" 
                         % (pn, a, np.rint(etamean*100.), np.rint(2*etastd*100.), np.rint(betamin), swedge, nxf, nyf, nzf, np.rint(rin), rmax, Rin/rhor, Rout, rbr, np.rint(simti), np.rint(simtf), np.rint(fti), np.rint(ftf), os.path.basename(os.getcwd())) )
        #flush to disk just in case to make sure all is written
        foutpower.flush()
        os.fsync(foutpower.fileno())
        foutpower.close()
    if False:
        avgmem = get2davg(usedefault=1)
        assignavg2dvars(avgmem)
        edtot2davg = (gdet[:,:,0:1]*avg_Tud[1][0][:,:,0:1]*_dx2*_dx3*nz).sum(-1).sum(-1)
        rhouuudtot2davg = (gdet[:,:,0:1]*avg_rhouuud[1][0][:,:,0:1]*_dx2*_dx3*nz).sum(-1).sum(-1)
        uguuudtot2davg = (gdet[:,:,0:1]*avg_uguuud[1][0][:,:,0:1]*_dx2*_dx3*nz).sum(-1).sum(-1)
        avg_tudmass = (gdet[:,:,0:1]*(avg_rhouu[1])*(avg_ud[0])*_dx2*_dx3*nz).sum(-1).sum(-1)
        avg_tudug = (gdet[:,:,0:1]*(avg_uguu[1])*(avg_ud[0])*_dx2*_dx3*nz).sum(-1).sum(-1)
        avg_tudmassug = (gdet[:,:,0:1]*(avg_rhouu[1]+avg_uguu[1])*(avg_ud[0])*_dx2*_dx3*nz).sum(-1).sum(-1)
        gc.collect()
    #return efficiency at r = 5:
    if returndf:
        Fmraw = mdtotvsr[ihor]
        Fmraw30 = mdotfinavgvsr30[ihor]
        Feraw = -edtotvsr[ihor]
        Fmval = Fm[iofr(5)]
        Feval = Fe[iofr(5)]
        epsFm = Fmval/Fmraw
        epsFm30 = Fmval/Fmraw30
        epsFke = (Fmval-Feval)/(Fmraw-Feraw)
        epsetaj = corrfac
        return( (epsFm,epsFke,epsetaj,epsFm30) )
    if ldtotvsr is not None:
        return( (eta[iofr(5)], Fm[iofr(5)], Fe[iofr(5)], Fl[iofr(5)]) )
    else:
        return( (eta[iofr(5)], Fm[iofr(5)], Fe[iofr(5)], None) )
    #
    # plt.figure(2)
    # plt.plot(r[:,0,0],edtotvsr,label="tot")
    # plt.plot(r[:,0,0],-edtot2davg,label="tot2davg")
    # #plt.plot(r[:,0,0],-rhouuudtot2davg,label="rhouuud")
    # #plt.plot(r[:,0,0],-gam*uguuudtot2davg,label="gamuguuud")
    # myma=-(rhouuudtot2davg+gam*uguuudtot2davg)
    # plt.plot(r[:,0,0],-avg_tudmass,label="mymass")
    # plt.plot(r[:,0,0],-avg_tudmassug,label="mymassug")
    # plt.plot(r[:,0,0],-avg_tudug,label="myug")
    # plt.plot(r[:,0,0],edmavsr,label="ma")
    # plt.plot(r[:,0,0],edtotvsr-edmavsr,label="tot-ma")
    # #plt.plot(r[:,0,0],DFfloor[1])
    # plt.xlim(rh,20); plt.ylim(-20,20)
    # plt.legend()
    # if ldtotvsr is not None:
    #     plt.plot(r[:,0,0],ldtotvsr+DFfloor4,label=r"$Lwoutfloor$")
    #plt.xlim(rhor,12)
    #plt.ylim(-3,20)
    #xx
    # plt.grid()
    # #
    # plt.figure(3)
    # plt.plot(r[:,0,0],-edtot2davg,label="tot2davg")
    # gc.collect()

def computeeta(start_t=8000,end_t=1e5,numintervals=8,doreload=1,qtymem=None,rj=100):
    #getqtyvstime(ihor,horval=0.2,fmtver=2,dobob=0,whichi=None,whichn=None):
    grid3d("gdump.bin", use2d = True)
    if qtymem is None:
        qtymem = getqtyvstime( iofr(rhor) )
    start_of_sim_t = qtymem[0,0,0]
    end_t1 = qtymem[0,-1,0]
    if end_t>end_t1:
        end_t = end_t1
    a_t,t_step = np.linspace(start_t,end_t,numintervals,retstep=True,endpoint=False)
    print( "start_t = %g, end_t = %g, nint = %d, step_t = %g" % (start_t,end_t,numintervals,t_step) )
    a_eta = np.zeros_like(a_t)
    a_Fm = np.zeros_like(a_t)
    a_Fe = np.zeros_like(a_t)
    a_Fl = np.zeros_like(a_t)
    a_phi = np.zeros_like(a_t)
    a_pj = np.zeros_like(a_t)
    a_pw = np.zeros_like(a_t)
    for (i,t_i) in enumerate(a_t):
        if i == 0: 
            doreload_local = doreload
        else: 
            doreload_local = 0
        res = takeoutfloors(doreload=doreload_local,fti=t_i,ftf=t_i+t_step,isinteractive=0,writefile=False,qtymem=qtymem)
        a_eta[i],a_Fm[i],a_Fe[i],a_Fl[i] = res
        mdtotvsr, edtotvsr, edmavsr, ldtotvsr,\
                    mdotfinavgvsr5, mdotfinavgvsr10, mdotfinavgvsr20, mdotfinavgvsr30, mdotfinavgvsr40, \
                    pjemfinavgvsr5, pjemfinavgvsr10, pjemfinavgvsr20, pjemfinavgvsr30, pjemfinavgvsr40, \
                    pjmafinavgvsr5, pjmafinavgvsr10, pjmafinavgvsr20, pjmafinavgvsr30, pjmafinavgvsr40, \
                    fstotfinavg, fstotsqfinavg, \
                    pjke_mu2_avg, pjke_mu1_avg, \
                    gdetF10, gdetF11, gdetF12 \
                    = plotqtyvstime( qtymem, whichplot = -200, fti=t_i,ftf=t_i+t_step, aphi_j_val=0 )
        unitsfactor=(4*np.pi)**0.5*2*np.pi
        #phibh=fstot[:,ihor]/4/np.pi/FMavg**0.5*unitsfactor
        #where fstot = (gdetB1).sum(2).sum(1)*_dx2*_dx3 at horizon
        a_phi[i] = fstotsqfinavg/4/np.pi/a_Fm[i]**0.5*unitsfactor
        a_pj[i] = pjke_mu2_avg[iofr(rj)]/a_Fm[i]
        a_pw[i] = (pjke_mu1_avg-pjke_mu2_avg)[iofr(rj)]/a_Fm[i]
    a_spar = (a_Fl/dxdxp[3,3,0,0,0]-2*a*a_Fe)/a_Fm
    print("Efficiencies:")    
    print zip(a_eta,a_Fm,a_Fe,a_Fl)
    print( "Average efficiency = %g" % a_eta.mean() ) 
    print( "Stdev eta: %g; stdev <eta>: %g" % (a_eta.std(), a_eta.std()/np.sqrt(a_eta.shape[0])) )
    return( a_eta.mean(), a_eta.std()/np.sqrt(a_eta.shape[0]), a_spar.mean(), a_spar.std()/np.sqrt(a_spar.shape[0]), a_phi.mean(), a_phi.std()/np.sqrt(a_phi.shape[0]), a_pj.mean()*a_Fm.mean(), a_pj.std()/np.sqrt(a_pj.shape[0])*a_Fm.mean(), a_pw.mean()*a_Fm.mean(), a_pw.std()/np.sqrt(a_pw.shape[0])*a_Fm.mean() )
    

def plotj(ts,fs,md,jem,jtot):
    #rc('font', family='serif')
    #plt.figure( figsize=(12,9) )
    plt.clf()
    fig,plotlist=plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(12,12))
    plottitle = "a = %g: %s" % ( a, os.path.basename(os.getcwd()) )
    plt.suptitle( plottitle )
    plt.subplots_adjust(hspace=0.1) #increase vertical spacing to avoid crowding
    plotlist[0].plot(ts,fs,label=r'$\Phi_{\rm h}/\Phi_{\rm i}$')
    #plotlist[0].plot(ts,fs,'r+') #, label=r'$\Phi_{\rm h}/0.5\Phi_{\rm i}$: Data Points')
    plotlist[0].legend(loc='lower right')
    #plt.xlabel(r'$t\;(GM/c^3)$')
    plotlist[0].set_ylabel(r'$\Phi_{\rm h}$',fontsize=16)
    plt.setp( plotlist[0].get_xticklabels(), visible=False)
    plotlist[0].grid(True)
    #
    #plotlist[1].subplot(212,sharex=True)
    plotlist[1].plot(ts,md,label=r'$\dot M_{\rm h}$')
    #plotlist[1].plot(ts,md,'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
    plotlist[1].legend(loc='lower right')
    #plotlist[1].set_xlabel(r'$t\;(GM/c^3)$')
    plotlist[1].set_ylabel(r'$\dot M_{\rm h}$',fontsize=16)
    plt.setp( plotlist[1].get_xticklabels(), visible=False)
    
    #plotlist[2].subplot(212,sharex=True)
    plotlist[2].plot(ts,jem/md,label=r'$P_{\rm j,em}/\dot M$')
    #plotlist[2].plot(ts,jem/md,'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
    plotlist[2].plot(ts,jtot/md,label=r'$P_{\rm j,tot}/\dot M$')
    #plotlist[2].plot(ts,jtot/md,'r+') #, label=r'$\dot M_{\rm h}$: Data Points')
    plotlist[2].legend(loc='lower right')
    plotlist[2].set_xlabel(r'$t\;(GM/c^3)$')
    plotlist[2].set_ylabel(r'$P_{\rm j}/\dot M_{\rm h}$',fontsize=16)

    #title("\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
    #      fontsize=16, color='r')
    plotlist[0].grid(True)
    plotlist[1].grid(True)
    plotlist[2].grid(True)
    fig.savefig('pjetf_%s.pdf' % os.path.basename(os.getcwd()) )


def test():
    t=np.arange(10)
    f=np.arange(10)**2
    plt.plot(t,f,label='$\Phi$')
    plt.title(r"This is a title")
    plt.legend(loc='upper right')
    plt.xlabel(r'$t (GM/c^3)$')
    plt.ylabel(r'$\Phi_{\rm h}$',fontsize=16)
    #plt.legend()

def gen_vpot(whichloop=None,phase=0.0,whichfield=None,fieldhor=0.194,rin=10):
    #whichfield = 0 -- single loop follows density contours
    #whichfield = None -- alternating loops
    global rho_av, rho_max, var, uq, uqc, uqcomax, aB, B, res,ud,etad, etau, gamma, vu, vd, bu, bd, bsq, phi
    #res=np.abs(bu[2]/np.sqrt(rho)/((uu[3]+1e-15)/uu[0])/_dx2)
    #plco(res,cb=True)
    #plt.plot(ti[:,ny/2,0],res[:,ny/2,0])
    #Old settings (now incorporated as defaults in this function call):
    #fieldhor = 0.194
    #rin = 10
    startfield = 1.1 * rin
    rho_av = np.copy(rho[:,:,0:1])
    #average to corners
    rho_av[1:nx,1:ny,0:1] = 0.25*(rho[0:nx-1,0:ny-1,0:1]+rho[1:nx,0:ny-1,0:1]+rho[0:nx-1,1:ny,0:1]+rho[1:nx,1:ny,0:1])
    rho_max=np.max(rho_av)
    #define aphi
    if( whichfield == None ):
        var = (r[:,:,0:1]**2*rho_av/rho_max) #**gam #*r[:,:,0:1]**1
        varc = (r[:,:,0:1]**2*rho/rho_max) #**gam #*r[:,:,0:1]**1
    else:
        var = (rho_av/rho_max) #**gam #*r[:,:,0:1]**1
        varc = (rho/rho_max) #**gam #*r[:,:,0:1]**1
    #note r should be shifted, too (not done yet):
    maxvar=np.max(var)
    maxvarc=np.max(varc)
    uq = (var-0.0*maxvar) #*r[:,:,0:1]**0.75 #/(0.1**2+(h-np.pi/2)**2)
    uqc = (varc-0.0*maxvarc) #*r[:,:,0:1]**0.75 #/(0.1**2+(h-np.pi/2)**2)
    uqcomax = varc/maxvarc #rho/rho_max #varc/maxvarc
    phi = np.log(r[:,:,0:1]/startfield)/fieldhor
    arg = phi-phase*np.pi
    #aaphi = uq**2 * (r-startfield)**1.1
    if( whichfield == None ):
        aaphi = uq**2 #* np.sin( arg )**1
    elif( whichfield == 0 ):
        aaphi = uq**2
    #aaphi = uq**2 * (1+0.2*np.sin( arg )**1)
    aaphi[uq<0] = 0
    if whichloop != None:
        notuse1 = arg > np.pi*(whichloop+1)
        notuse2 = arg < np.pi*whichloop
        aaphi[notuse1] = 0.0
        aaphi[notuse2] = 0.0
    #aaphi = uq**2 #* np.log(r[:,:,0:1]/startfield)
    #aaphi = uq**(2)
    aphi2B(aaphi)
    #reset field components outside torus to zero
    #B[1,uqc<0] = 0
    #B[2,uqc<0] = 0
    return(aaphi)

def aphi2B(aaphi):
    #aB -- face-centered
    #B -- cell-centered
    global B, aB, gdetB
    aB = np.zeros_like(B)
    gdetB = np.zeros_like(B)
    gdetB[1,1:nx,0:ny-1] = (aaphi[1:nx,1:ny]-aaphi[1:nx,0:ny-1])/_dx2
    gdetB[2,0:nx-1,1:ny] = (aaphi[1:nx,1:ny]-aaphi[0:nx-1,1:ny])/_dx1
    aB[1,1:nx,0:ny-1] = gdetB[1,1:nx,0:ny-1] / (0.5*(gdet[0:nx-1,0:ny-1]+gdet[1:nx,0:ny-1]))
    aB[2,0:nx-1,1:ny] = gdetB[2,0:nx-1,1:ny] / (0.5*(gdet[0:nx-1,0:ny-1]+gdet[0:nx-1,1:ny]))
    #ab[3] is zeroes
    #
    B=np.zeros_like(aB)
    #properly center the field
    B[1,0:nx-1,0:ny,:] = (aB[1,0:nx-1,0:ny,:] + aB[1,1:nx,0:ny,:])/2
    B[2,0:nx,0:ny-1,:] = (aB[2,0:nx,1:ny,:] + aB[2,0:nx,0:ny-1,:])/2

def pl(x,y,j=None):
    global ny
    if j == None: j = ny/2
    plt.plot(x[:,j,0],y[:,j,0])

def fac(ph):
    return(1+0.5*((ph/np.pi-1.5)/0.5)**2)

def avg2ctof(q):
    qavg2 = np.empty_like(q)
    qavg2[0:nx,1:ny,:] = (q[0:nx,1:ny,:] + q[0:nx,0:ny-1,:])/2
    return(qavg2)

def avg1ctof(q):
    qavg1 = np.empty_like(q)
    qavg1[1:nx,0:ny,:] = (q[0:nx-1,0:ny,:] + q[1:nx,0:ny,:])/2
    return(qavg1)

def avg0ctof(q):
    resavg0 = np.empty_like(q)
    resavg0[1:nx,1:ny,:] = 0.25*(q[0:nx-1,0:ny-1,:]+q[1:nx,0:ny-1,:]+q[0:nx-1,1:ny,:]+q[1:nx,1:ny,:])
    return(resavg0)

def normalize_field(targbsqoug):
    global B, gdetB, bsq, ug
    maxbsqoug = np.max(bsq/(ug+1e-5))
    rat = np.sqrt(targbsqoug/maxbsqoug)
    #rescale all field components
    B *= rat
    gdetB *= rat
    #recompute derived quantities
    cvel()
def plotbs(dy=0):
    plt.clf();
    plot(ti[:,ny/2,0],B[1,:,ny/2+dy,0])
    plot(ti[:,ny/2,0],B[2,:,ny/2+dy,0])
    plot(ti[:,ny/2,0],(bsq/ug)[:,ny/2+dy,0]/100)
    plot(ti[:,ny/2,0],(gdetB[1]/gdet)[:,ny/2+dy,0])
def plotaphi(dy=0):
    aphi=fieldcalc()
    plot(r[:,ny/2,0],aphi[:,ny/2,0])
    xlim(xmin=10,xmax=100)

def face2centdonor():
    global bcent
    bcent=np.zeros_like(B)
    bcent[1][0:nx-1,:,:]=0.5*(gdetB[1][0:nx-1,:,:]+gdetB[2][1:nx,:,:])/gdet[0:nx-1,:,:]
    bcent[2][:,0:ny-1,:]=0.5*(gdetB[2][:,0:ny-1,:]+gdetB[2][:,1:ny,:])/gdet[:,0:ny-1,:]
    bcent[3][:,:,0:nz-1]=0.5*(gdetB[2][:,:,0:nz-1]+gdetB[2][:,:,1:nz])/gdet[:,:,0:nz-1]

def pf(dir=2):
    global bcent
    grid3d("gdump.bin")
    #rfd("fieldline0001.bin")
    #rrdump("rdump--0000.bin")
    rd("dump0000.bin")
    face2centdonor(); 
    plt.clf(); 
    myi = 20
    myk = 0
    plt.plot(tj[myi,:,myk],bcent[dir,myi,:,myk]);
    plt.plot(tj[myi,:,myk],B[dir,myi,:,myk]);
    plt.plot(tj[myi,0:ny-1,myk]+0.5,gdetB[dir,myi,1:ny,myk]/(0.5*(gdet[myi,0:ny-1,myk]+gdet[myi,1:ny,myk])))


def chophi(var,maxvar):
    newvar = np.copy(var)
    newvar[var>maxvar]=0*var[var>maxvar]+maxvar
    return(newvar)

def choplo(var,minvar):
    newvar = np.copy(var)
    newvar[var<minvar]=0*var[var<minvar]+minvar
    return(newvar)

def Risco(a):
    Z1 = 1 + (1. - a**2)**(1./3.) * ((1. + a)**(1./3.) + (1. - a)**(1./3.))
    Z2 = (3*a**2 + Z1**2)**(1./2.)
    risco = 3 + Z2 - np.sign(a)* ( (3 - Z1)*(3 + Z1 + 2*Z2) )**0.5
    return(risco)

def Ebind(r,a):
    Eb = 1 - (r**2-2*r+a*r**0.5)/(r*(r**2-3*r+2*a*r**0.5)**0.5)
    return( Eb )

def Ebindisco(a):
    Eb = Ebind( Risco(a), a)
    return( Eb )

def ek(a,r):
    ek = (r**2-2*r+a*r**0.5)/(r*(r**2-3*r+2*a*r**0.5)**0.5)
    return(ek)

def lk(a,r):
    udphi = r**0.5*(r**2-2*a*r**0.5+a**2)/(r*(r**2-3*r+2*a*r**0.5)**0.5)
    return( udphi )

def sparthin(a):
    risco=Risco(a)
    l = lk(a,risco)
    e = ek(a,risco)
    s = l-2*a*e 
    return(s)

def getetaavg(fname,simnamelist):
    gd1 = np.loadtxt( fname, unpack = True, usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] )
    #gd=gd1.view().reshape((-1,nx,ny,nz), order='F')
    alist, etalist, etastdlist, sparlist, sparstdlist, philist, phistdlist, \
        Fmlist, Felist, Fllist, FEMrhorlist, FEM2list, \
        powjetlist, powjetstd, powwindlist, powwindstd, \
        ftotlist, ftotsqlist, \
        hor5, hor10, hor20, hor25, hor30, hor100 = gd1
    fsqtotlist = ftotsqlist
    mdotlist = Fmlist
    rhorlist = 1+(1-alist**2)**0.5
    omhlist = alist / 2 / rhorlist
    etaEMlist = -FEM2list/Fmlist
    etajetlist=powjetlist/Fmlist
    etawindlist = powwindlist/Fmlist
    #
    gin = open( fname, "rt" )
    emptyline = gin.readline()
    simname=[]
    indexsim=dict()
    etasim=dict()
    etastdsim=dict()
    simpath=[]
    for i in np.arange(alist.shape[0]):
        stringsplit=gin.readline().split()
        simname.append(stringsplit[0])
        simpath.append(stringsplit[1])
        indexsim[stringsplit[0]]=i
        etasim[stringsplit[0]]=etalist[i]
        etastdsim[stringsplit[0]]=etastdlist[i]
    gin.close()
    etas=np.zeros(len(simnamelist),dtype=np.float64)
    etas_std=np.zeros(len(simnamelist))
    for i,simname in enumerate(simnamelist):
        etas[i] = etasim[simname]
        etas_std[i] = etastdsim[simname]
    eta9avg,eta9err,eta9std=wmom(etas,etas_std**(-2),calcerr=True,sdev=True)
    return eta9avg*100.,2*eta9err*100. #,2*eta9std

def computeavgs():
    print( "a = 0.9:")
    getetaavg('siminfo.txt',('A0.9f','A0.9','A0.9$l_r$','A0.9$h_r$','A0.9$l_\\theta$','A0.9$h_\\theta$','A0.9$h_{\\theta}h_{\\varphi}$','A0.9$h_\\varphi$','A0.9$h^2_\\varphi$'))
    print( "a = -0.9:")
    getetaavg('siminfo.txt',('A-0.9f','A-0.9','A-0.9$l_r$','A-0.9$l_\\theta$','A-0.9$h_\\theta$','A-0.9$h_\\varphi$',))

def plotpowers(fname,hor=0,format=2,usegaussianunits=True,nmin=-20,plotetas=False,nsigma=1,eps=1e-5,dofill=False,doanalytic=False,fntsize=25):
    if usegaussianunits == True:
        unitsfactor = (4*np.pi)**0.5*2*np.pi
    else:
        unitsfactor = 1.
    if format == 0: #old format
        gd1 = np.loadtxt( fname, unpack = True, usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] )
        #gd=gd1.view().reshape((-1,nx,ny,nz), order='F')
        alist = gd1[0]
        rhorlist = 1+(1-alist**2)**0.5
        omhlist = alist / 2 / rhorlist
        mdotlist = gd1[1]
        #etalist = gd1[3]
        powlist=gd1[12] #pow(2*rstag)
        psitotsqlist = gd1[5]
        psi30sqlist = gd1[7]
        etalist = powEMKElist/mdotlist
        etaEMlist = eoutEMtotlist/mdotlist
        etawindlist = powwindEMKElist/mdotlist
    elif format == 1: #new avg2d format
        gd1 = np.loadtxt( fname, unpack = True, usecols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27] )
        #gd=gd1.view().reshape((-1,nx,ny,nz), order='F')
        alist = gd1[0]
        rhorlist = 1+(1-alist**2)**0.5
        omhlist = alist / 2 / rhorlist
        mdotlist = gd1[5]
        ftotlist=gd1[6]
        fsqtotlist=gd1[7]
        f30list=gd1[8]
        f30sqlist=gd1[9]
        pjetemtotlist=gd1[10]
        eoutEMtotlist=gd1[11]
        #etalist = gd1[3]
        i = 12
        #1
        powEMKElist=gd1[i]; i+=1
        powwindEMKElist=gd1[i]; i+=1
        powlist=gd1[i]; i+=1
        powwindlist=gd1[i]; i+=1
        rlist=gd1[i]; i+=1
        #2
        powEMKElist2=gd1[i]; i+=1
        powwindlEMKEist2=gd1[i]; i+=1
        powlist2=gd1[i]; i+=1
        powwindlist2=gd1[i]; i+=1
        rlist2=gd1[i]; i+=1
        #3
        powEMKElist3=gd1[i]; i+=1
        powwindEMKElist3=gd1[i]; i+=1
        powlist3=gd1[i]; i+=1
        powwindlist3=gd1[i]; i+=1
        rlist3=gd1[i]; i+=1
        etalist = powEMKElist/mdotlist
        etaEMlist = eoutEMtotlist/mdotlist
        etajetlist = (powwindlist-powlist)/mdotlist
        etawindlist = powwindEMKElist/mdotlist
    elif format == 2:
        # foutpower.write( "%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (
        #                                                    pn, os.path.basename(os.getcwd()), a, 
        #                                                    etamean, etastd, sparmean, sparstd, phimean, phistd,
        #                                                    Fm[iofr(rx)], Fe[iofr(rx)], Fl[iofr(rx)]/dxdxp[3][3][0,0,0],
        #                                                    FEM[iofr(rhor)], FEM[iofr(2)],
        #                                                    pj, 
        #                                                    pw,
        #                                                    fstotfinavg, fstotsqfinavg,
        #                                                    horavg[iofr(5)], horavg[iofr(10)], horavg[iofr(20)], 
        #                                                    horavg[iofr(25)], horavg[iofr(30)], horavg[iofr(100)]) )
        gd1 = np.loadtxt( fname, unpack = True, usecols = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] )
        #gd=gd1.view().reshape((-1,nx,ny,nz), order='F')
        alist, etalist, etastdlist, sparlist, sparstdlist, philist, phistdlist, \
            Fmlist, Felist, Fllist, FEMrhorlist, FEM2list, \
            powjetlist, powjetstd, powwindlist, powwindstd, \
            ftotlist, ftotsqlist, \
            hor5, hor10, hor20, hor25, hor30, hor100 = gd1
        fsqtotlist = ftotsqlist
        mdotlist = Fmlist
        rhorlist = 1+(1-alist**2)**0.5
        omhlist = alist / 2 / rhorlist
        etaEMlist = -FEM2list/Fmlist
        #divide by Fm to get efficiency
        etajetlist=powjetlist/Fmlist
        etawindlist = powwindlist/Fmlist
        etajetstdlist = powjetstd/Fmlist
        etawindstdlist = powwindstd/Fmlist
    gin = open( fname, "rt" )
    emptyline = gin.readline()
    simname=[]
    simpath=[]
    print( "##: %20s: %9s %9s %9s %9s %9s %9s %9s %9s\n" % ("Name", "Mdot", "FEMrh", "Ftotsq", "etaEM", "eta", "etajw", "etaj", "etaw") )
    for i in np.arange(alist.shape[0]):
        stringsplit=gin.readline().split()
        simname.append(stringsplit[0])
        simpath.append(stringsplit[1])
        print( "%2d: %20.20s: %9.5g %9.5g %9.5g %9.5g %9.5g %9.5g %9.5g %9.5g" % ( i, simname[i], Fmlist[i], FEMrhorlist[i], ftotsqlist[i], etaEMlist[i]*100, etalist[i]*100, etajetlist[i]*100+etawindlist[i]*100, etajetlist[i]*100, etawindlist[i]*100 ) ) 
    gin.close()
    if plotetas:
        #plt.figure(1)
        plt.xlim(0,200)
        plt.ylim(-2,alist.shape[0])
        ilist = np.arange(alist.shape[0])
        pylab.errorbar(etalist*100, ilist, xerr=nsigma*etastdlist*100,marker='o',ls='None')
        plt.xlabel(r"$\eta$")
        ax=plt.gca()
        ax.get_yaxis().set_visible(False)
        for i in ilist:
            t = ax.text((etalist[i]+etastdlist[i]*2)*100+5, i, r"$\mathrm{%s}$" % simname[i], withdash=False, va='center' )
        etam9avg,etam9err,etam9std=wmom(etalist[alist<0],etastdlist[alist<0]**(-2),calcerr=True,sdev=True)
        eta9avg,eta9err,eta9std=wmom(etalist[alist>0],etastdlist[alist>0]**(-2),calcerr=True,sdev=True)
        print etam9avg, etam9std
        print eta9avg, eta9std
        pylab.errorbar(np.array([etam9avg,eta9avg])*100, np.array([-1.,-1.],dtype=np.float64),marker='o', xerr=nsigma*np.array([etam9std,eta9std])*100,ls='None')
        #def wmom(arrin, weights_in, inputmean=None, calcerr=False, sdev=False):
        plt.savefig( "fig1.eps",bbox_inches='tight',pad_inches=0.02  )
        return
    mya=np.arange(-1,1,0.001)
    rhor = 1+(1-mya**2)**0.5
    myomh = mya / 2/ rhor
    #fitting function
    ff=1.08
    f0 = 2.9*ff
    f0n = 0.9*f0*0.98/ff
    f1 = -0.8/ff
    f1n = 1.2
    f2 = 0.
    #f = f0 * (1 + (f1*(1+np.sign(myomh))/2. + f1n*(1-np.sign(myomh))/2.) * myomh + f2 * myomh**2)
    fneg = f0 * (1 + f1*myomh + f2 * myomh**2)
    fpos = f0n* (1 + f1n*myomh)
    f = amin(fneg,fpos)
    #gammie way -- too large amplitude
    #gammieparamopi = (8./3.*(3.-mya**2+3.*(1-mya**2)**0.5))**0.5
    #f = f0 * 0.25 * gammieparamopi
    #mypwr = 5*myomh**2
    psi = 1.0
    mypwr = 2.0000 * 1.*1.0472*myomh**2 * 1.5*(psi**2-psi**3/3) #prefactor = 2pi/3, consistent with (A7) from TMN10a
    horx=0.09333
    #myr = Risco(mya) #does not work at all: a < 0 power is much greater than a > 0
    myeta = mypwr * (mya**2+3*rhor**2)/3 / (2*np.pi*horx)
    #plt.plot( mya, myeta )
    rhor6 = 1+(1-mspina6[mhor6==hor]**2)**0.5
    #Tried to equate pressures -- works but mistake in calculaton -- wrong power
    #plt.plot(mspina6[mhor6==hor],mpow6[mhor6==hor]* ((mspina6[mhor6==hor]**2+3*rhor6**2)/3/(2*np.pi*horx)) )
    #Simple multiplication by rhor -- works!  \Phi^2/Mdot * rhor ~ const
    fac = 0.838783 #conversion to parabolic power from monopolar power = 0.044/(1./(6*np.pi))
    #plt.plot(mya,mya**2)
    #plt.plot(alist,etawindlist,'go',label=r'$\eta_{\rm jet}+\eta_{\rm wind}$')
    # plt.plot(mspina6[mhor6==hor],fac*6.94*mpow6[mhor6==hor],'r--',label=r'$P_{\rm BZ,6}$')
    # plt.plot(mspina6[mhor6==hor],fac*3.75*mpow6[mhor6==hor]*rhor6,'r',label=r'$P_{\rm BZ,6}\times\, r_h$' )
    if True:
        myomh6=np.concatenate((-momh6[mhor6==hor][nmin::-1],momh6[mhor6==hor]))
        myspina6=np.concatenate((-mspina6[mhor6==hor][nmin::-1],mspina6[mhor6==hor]))
        mypow6 = np.concatenate((mpow6[mhor6==hor][nmin::-1],mpow6[mhor6==hor]))
    else:
        myomh6=momh6[mhor6==hor]
        myspina6=mspina6[mhor6==hor]
        mypow6 = mpow6[mhor6==hor]
    #mypsiosqrtmdot = f0*(1.+(f1*(1+np.sign(myomh6))/2. + f1n*(1-np.sign(myomh6))/2.)*myomh6)
    fneg6 = f0 * (1 + f1*myomh6 + f2 * myomh6**2)
    fpos6 = f0n* (1 + f1n*myomh6)
    mypsiosqrtmdot = amin(fneg6,fpos6)
    myeta6 = (mypsiosqrtmdot)**2*mypow6
    # plt.figure(5)
    # plt.clf()
    # plt.plot(mspina6[mhor6==hor],mpow6[mhor6==hor] )
    # plt.plot(mspina2[mhor2==hor],mpow2a[mhor2==hor] )
    # mpow2ahere = 2.0000 * 1.*1.0472*momh2**2 * 1.5*(psi**2-psi**3/3)
    # plt.plot(mspina2[mhor2==hor], mpow2ahere[mhor2==hor])
    #plt.plot(mya,mya**2)
    #plt.plot(mya,mypwr)
    #
    if format==0:
        plt.figure(2)
        plt.clf()
        #plt.plot( mya, myeta )
        #plt.plot(mya,mya**2)
        #y = (psi30sqlist)**2/(2*mdotlist)
        y = (psi30sqlist)**2/(2*mdotlist)
        plt.plot(alist,y,'bo')
        plt.plot(mya,(250+0*mya)*rhor) 
        plt.plot(mya,250./((3./(mya**2 + 3*rhor**2))**2*2*rhor**2)) 
        #plt.plot(mya,((mya**2+3*rhor**2)/3)**2/(2/rhor)) 
        plt.ylim(ymin=0)
        #plt.plot(alist,2*mdotlist/(psitotsqlist)**2,'o')
        #plt.plot(mspina6[mhor6==hor],5*mpow6[mhor6==hor])
        #plt.plot(mspina2[mhor2==hor],5*mpow2a[mhor2==hor])
        # plt.figure(4)
        # plt.clf()
        # y = (psi30sqlist)**2/(2*mdotlist*rhorlist**2)
        # plt.plot(alist,1/(y/np.max(y)),'o')
        # plt.plot(mya,Risco(mya)**(1./2.)*0.9) 
        # plt.ylim(ymin=0)
    else:
        plt.figure(2)
        plt.clf()
        #plt.plot( mya, myeta )
        #plt.plot(mya,mya**2)
        #y = (psi30sqlist)**2/(2*mdotlist)
        #y = (f30sqlist/2./(2*np.pi))/(mdotlist)**0.5
        y1= (fsqtotlist/2./(2*np.pi))/(mdotlist)**0.5
        #plt.plot(alist,y,'bo')
        plt.plot(omegah_compute(alist)/omegah_compute(1),y1,'rx')
        plt.plot(omegah_compute(mya)/omegah_compute(1),f,'g')
        # plt.plot(mya,(250+0*mya)*rhor) 
        # plt.plot(mya,250./((3./(mya**2 + 3*rhor**2))**2*2*rhor**2)) 
        #plt.plot(mya,((mya**2+3*rhor**2)/3)**2/(2/rhor)) 
        plt.ylim(ymin=0)
        #plt.plot(alist,2*mdotlist/(psitotsqlist)**2,'o')
        #plt.plot(mspina6[mhor6==hor],5*mpow6[mhor6==hor])
        #plt.plot(mspina2[mhor2==hor],5*mpow2a[mhor2==hor])
        # plt.figure(4)
        # plt.clf()
        # y = (psi30sqlist)**2/(2*mdotlist*rhorlist**2)
        # plt.plot(alist,1/(y/np.max(y)),'o')
        # plt.plot(mya,Risco(mya)**(1./2.)*0.9) 
        # plt.ylim(ymin=0)
    if format == 2:
        plt.figure(0,figsize=(8,6),dpi=100)
        plt.clf()
        cond=sparlist<-1e10
        plt.plot(alist,sparlist,'o')
        plt.plot(alist[:9],sparlist[:9],'b-',lw=2)
        plt.ylim(-10,10)
        plt.grid()
        plt.ylabel(r"$s = (\dot L - 2 a \dot E)/\dot M_0$", fontsize=20)
        plt.xlabel(r"$a$",fontsize=20)
        print zip(alist,sparlist)
    #
    # Compute averages and std's if there are several sims at one value
    #
    u_alist = np.unique(alist)
    u_philist = np.zeros_like(u_alist)
    u_phistdlist = np.zeros_like(u_alist)
    u_etalist = np.zeros_like(u_alist)
    u_etastdlist = np.zeros_like(u_alist)
    u_etajetlist = np.zeros_like(u_alist)
    u_etajetstdlist = np.zeros_like(u_alist)
    u_etawindlist = np.zeros_like(u_alist)
    u_etawindstdlist = np.zeros_like(u_alist)
    u_sparlist = np.zeros_like(u_alist)
    u_sparstdlist = np.zeros_like(u_alist)
    u_numsims =  np.zeros_like(u_alist)
    for i,aval in enumerate(u_alist):
        #make comparison in omegah space where closely spaced values of a are 
        #not spread out form each other
        cond = (np.abs(omegah_compute(alist)-omegah_compute(aval))<eps)
        u_philist[i],u_phistdlist[i] = getavgstd( philist[cond], phistdlist[cond])
        u_etalist[i],u_etastdlist[i] = getavgstd( etalist[cond], etastdlist[cond])
        u_etajetlist[i],u_etajetstdlist[i] = getavgstd( etajetlist[cond], etajetstdlist[cond])
        u_etawindlist[i],u_etawindstdlist[i] = getavgstd( etawindlist[cond], etawindstdlist[cond])
        u_sparlist[i],u_sparstdlist[i] = getavgstd( sparlist[cond], sparstdlist[cond])
        u_numsims[i] = cond.sum()
        print( "%2d: %9.5g %9.5g %9.5g %9.5g %9.5g %3g" % (
                i, aval, u_philist[i], 2*u_phistdlist[i], 100*u_etalist[i], 2*100*u_etastdlist[i], u_numsims[i]) )
    #
    plt.figure(2)
    plt.errorbar(omegah_compute(u_alist)/omegah_compute(1),u_philist/unitsfactor,yerr=2*u_phistdlist/unitsfactor,label=r'$\langle\phi^2\!\rangle^{1/2}$',mfc='b',ecolor='b',lw=2,fmt='+',elinewidth=1,mew=1)

    #
    plt.figure(1, figsize=(8,4),dpi=100)
    plt.clf()
    gs = GridSpec(2, 2)
    gs.update(left=0.09, right=0.94, top=0.95, bottom=0.1, wspace=0.25, hspace=0.04)
    #############
    #
    # phi
    #
    #############
    ax1 = plt.subplot(gs[0,0])
    newy1 = 1.05*f*unitsfactor
    newy2 = 0.95*f*unitsfactor
    col= ( 0.52941176,  0.80784314,  0.98039216, 0.5) #(0.5,0.5,1,0.75) #(0.8,1,0.8,1)
    if dofill:
        ax1.fill_between(mya,newy1,newy2,where=newy1>newy2,facecolor=col,edgecolor=col)
    ax1.plot(mya,f*unitsfactor,'k-',label=r'$\phi_{\rm fit}$',lw=2) #=2.9(1-0.6 \Omega_{\rm H})
    #ax1.plot(alist,y1*unitsfactor,'o',label=r'$\langle\phi^2\!\rangle^{1/2}$',mfc='r')
    ax1.errorbar(u_alist,u_philist,yerr=2*u_phistdlist,label=r'$\langle\phi^2\!\rangle^{1/2}$',mfc='r',ecolor='r',fmt='o',lw=2,elinewidth=1,mew=1)
    # plt.plot(mya,(250+0*mya)*rhor) 
    # plt.plot(mya,250./((3./(mya**2 + 3*rhor**2))**2*2*rhor**2)) 
    #plt.plot(mya,((mya**2+3*rhor**2)/3)**2/(2/rhor)) 
    plt.ylim(ymin=0.0001)
    plt.ylabel(r"$\phi_{\rm BH}$",fontsize='x-large',ha='center',labelpad=16)
    plt.grid()
    plt.setp( ax1.get_xticklabels(), visible=False )
    plt.legend(ncol=1,loc='lower center',frameon=True,labelspacing=0.0,borderpad=0.2) #,scatterpoints=1,numpoints=1)
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
    plt.text(-0.85, 0.9*plt.ylim()[1], r"$(\mathrm{a})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    #second y-axis
    if False:
        ax1r = ax1.twinx()
        ax1r.set_xlim(-1,1)
        ax1r.set_ylim(ax1.get_ylim())
    #
    #############
    #
    # eta
    #
    #############
    ax2 = plt.subplot(gs[1,0])
    newy1 = 1.1*100*fac*myeta6
    newy2 = 0.9*100*fac*myeta6
    if dofill:
        ax2.fill_between(myspina6,newy1,newy2,where=newy1>newy2,facecolor=col,edgecolor=col,lw=2)
    #plt.plot(alist,100*(etawindlist-etalist),'gv',label=r'$\eta_{\rm wind}$')
    #plt.plot(myspina6,0.9*100*fac*myeta6,'k',label=r'$0.9\eta_{\rm BZ6}(\phi_{\rm fit})$' )
    plt.plot(myspina6,100*fac*myeta6,'k-',label=r'$\eta_{\rm BZ6}(\phi_{\rm fit})$',lw=2)
    #plt.plot(myspina6,(100-4.4305)*fac*myeta6+4.4305,'k:',label=r'$\eta_{\rm BZ6}(\phi_{\rm fit})$',lw=2)
    #u_etalist[0]*=0.8
    #z=np.polyfit(omegah_compute(u_alist),100*u_etalist,6)
    z=np.polyfit(u_alist,100*u_etalist,3)#,w=1/(2*100*u_etastdlist)**2)
    z[-1]=4.4
    z[-2]=0
    print z
    eta_func=np.poly1d(z)    
    # eta_func2=poly1dt(z)    
    plt.plot(mya,eta_func(mya),'k:',lw=2)
    # plt.plot(myspina6,4.4305+20*(myomh6/omegah_compute(0.9))**1+100*(myomh6/omegah_compute(0.9))**2+10*(myomh6/omegah_compute(0.9))**3-30*(myomh6/omegah_compute(0.9))**4,'k--',label=r'$\eta_{\rm BZ6}(\phi_{\rm fit})$',lw=2)
    # plt.plot(myspina6,4.4305+130*(myomh6/omegah_compute(0.9))**2-30*(myomh6/omegah_compute(0.9))**4,'k--',label=r'$\eta_{\rm BZ6}(\phi_{\rm fit})$',lw=2)
    # plt.plot(myspina6,95*(np.abs(omegah_compute(myspina6))/omegah_compute(0.9))**2+5,'k:',label=r'$100(a/0.9)^2$',lw=2)
    #plt.plot(u_alist,100*u_etalist,'o',label=r'$\eta$',mfc='r',lw=2)
    ax2.errorbar(u_alist,100*u_etalist,yerr=2*100*u_etastdlist,label=r'$\eta$',mfc='r',ecolor='r',fmt='o',lw=2,elinewidth=1,mew=1)
    plt.ylim(0.0001,160-1e-5)
    plt.grid()
    # plt.setp( ax2.get_xticklabels(), visible=False )
    plt.ylabel(r"$\eta\  [\%]$",fontsize='x-large',ha='center',labelpad=12)
    plt.xlabel(r"$a$",fontsize='x-large')
    plt.text(-0.85,  0.9*plt.ylim()[1], r"$(\mathrm{b})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    plt.legend(ncol=1,loc='upper center',frameon=True,labelspacing=0.0,borderpad=0.2)
    #second y-axis
    # ax2r = ax2.twinx()
    # ax2r.set_xlim(-1,1)
    # ax2r.set_ylim(0.0001,160)
    #
    #############
    #
    # eta_j, eta_w
    #
    #############
    ax3 = plt.subplot(gs[1,1])
    newy1 = 1.1*0.85*100*fac*myeta6
    newy2 = 0.9*0.85*100*fac*myeta6
    #col=(0.93333333,  0.60980392,  0.93333333,0.75) #(0.8,0.52,0.25,0.4)
    if dofill:
        ax3.fill_between(myspina6,newy1,newy2,where=newy1>newy2,facecolor=col,edgecolor=col)  #(0.8,1,0.8,1)
    l,=plt.plot(myspina6,0.85*100*fac*myeta6,'k--',lw=2,label=r'$0.85\eta_{\rm BZ6}(\phi_{\rm fit})$' )
    l.set_dashes([10,5])
    #plt.plot(myspina6,myeta6,'r:',label=r'$\eta_{\rm BZ,6}$')
    #plt.plot(alist,100*etajetlist,'gs',label=r'$\eta_{\rm jet}$',lw=2)
    ax3.errorbar(u_alist,100*u_etajetlist,yerr=2*100*u_etajetstdlist,label=r'$\eta_{\rm jet}$',mfc='g',ecolor='g',fmt='^',lw=2,elinewidth=1,mew=1)
    etajet_polycoef=np.polyfit(u_alist,100*u_etajetlist,3)#,w=1/(2*100*u_etastdlist)**2)
    print etajet_polycoef
    etajet_polycoef[-1]=0
    etajet_poly=np.poly1d(etajet_polycoef)
    print etajet_poly(0)
    plt.plot(mya,etajet_poly(mya),"k:")
    #plt.plot(alist,100*etaEMlist,'rx',label=r'$\eta_{\rm jet}$')
    #plt.plot(alist,100*etawindlist,'bv',label=r'$\eta_{\rm wind}$')
    ax3.errorbar(u_alist,100*u_etawindlist,yerr=2*100*u_etawindstdlist,label=r'$\eta_{\rm wind}$',mfc='b',ecolor='b',fmt='v',lw=2,elinewidth=1,mew=1)
    #plt.plot(myspina6,100*fac*myeta6,'k-',lw=2) #,label=r'$\eta_{\rm BZ6}(\phi_{\rm fit})$' )
    plt.ylim(0.0001,160-1e-5)
    plt.grid()
    plt.legend(ncol=1,loc='upper center',frameon=True,labelspacing=0.0,borderpad=0.2)
    plt.xlabel(r"$a$",fontsize='x-large')
    plt.ylabel(r"$\eta_{\rm jet},\ \eta_{\rm wind}\  [\%]$",fontsize='x-large',ha='center',labelpad=12)
    plt.text(-0.85,  0.9*plt.ylim()[1], r"$(\mathrm{c})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    #second y-axis
    ax3r = ax3.twinx()
    ax3r.set_xlim(-1,1)
    ax3r.set_ylim(ax3.get_ylim())
    #
    #############
    #
    # s (spin-up parameter)
    #
    #############
    ax4 = plt.subplot(gs[0,1])
    x=(0.07,0.07)
    y=(-10,10)
    plt.plot(x,y,color='red',lw=4,alpha=0.3)
    l,=plt.plot(mya,sparthin(mya),'g-.',lw=2,label=r"$s_{\rm NT}$")
    l.set_dashes([10,3,2,3])
    if doanalytic:
        #to show "analytic" rought approximation of Ramesh
        plt.plot(mya,sparthin(0)*(1-mya),'k-',lw=1)
    #plt.plot(alist,sparlist,'ro',mec='r')
    newa=np.concatenate((u_alist[0:-4],u_alist[-3:]))
    news=np.concatenate((u_sparlist[0:-4],u_sparlist[-3:]))
    spar_polyfit=np.polyfit(newa,news,4)#,w=1/(2*100*u_etastdlist)**2)
    spar_func=np.poly1d(spar_polyfit)
    # spar_func2=poly1dt(spar_polyfit)
    plt.plot(mya,spar_func(mya),'k:',lw=2)
    print spar_polyfit
    ax4.errorbar(u_alist,u_sparlist,yerr=2*u_sparstdlist,label=r"$s_{\rm MAD}$",mfc='r',ecolor='r',fmt='o',color='r',lw=2,elinewidth=1,mew=1)
    if doanalytic:
        #to show "analytic" rought approximation of Ramesh
        plt.plot(mya,-8*mya,'k-',lw=1)
    #plt.plot(alist[:9],sparlist[:9],'ro-',lw=2,label=r"$s_{\rm MAD}$")
    plt.text(x[0]+0.02,7,r"$a_{\rm eq}^{\rm Sim}\!\approx0.07$",va="center",ha="left",fontsize=16,color="red",alpha=1)
    plt.ylim(-10,10)
    plt.grid()
    plt.ylabel(r"$s$", fontsize='x-large',ha='center',labelpad=9) # = (\dot F_L/M - 2 a \dot F_E)/\dot F_M
    #plt.xlabel(r"$a$",fontsize='x-large')
    plt.legend(ncol=1,loc='lower left',frameon=True,labelspacing=0.0,borderpad=0.2) #,scatterpoints=1,numpoints=1)
    plt.setp( ax4.get_xticklabels(), visible=False )
    plt.text(-0.85,  0.9*(plt.ylim()[1]-plt.ylim()[0])+plt.ylim()[0], r"$(\mathrm{d})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular' #,bbox=bbox_props
             )
    ax4r = ax4.twinx()
    ax4r.set_ylim(ax4.get_ylim())
    #plt.savefig("jetwindeta.pdf",bbox_inches='tight',pad_inches=0)
    #plt.savefig("jetwindeta.eps",bbox_inches='tight',pad_inches=0)
    plt.savefig("jetwindeta.pdf",bbox_inches='tight',pad_inches=0.02)
    # plt.savefig("jetwindeta.eps",bbox_inches='tight',pad_inches=0.02)
    # plt.savefig("jetwindeta.png",bbox_inches='tight',pad_inches=0.02)
    #plt.plot(mspina2[mhor2==hor],5*mpow2a[mhor2==hor])
    #
    #
    plt.figure(3)
    plt.clf()
    #plt.plot( mya, myeta )
    #plt.plot(mya,mya**2)
    psi=1.0
    pow6func = interp1d(momh6[mhor6==hor], mpow6[mhor6==hor])
    #divide flux by 2 (converts flux to single hemisphere) and by dxdxp33 (accounts for A_3 -> A_\phi):
    # pwrlist = (psi30sqlist/2/(2*np.pi))**2*2.0000 * 1.*1.0472*omhlist**2 * 1.5*(psi**2-psi**3/3)
    # plt.plot(alist,pwrlist6/mdotlist,'o')
    if format == 0:
        pwrlist6 = (psi30sqlist/2/(2*np.pi))**2*pow6func(np.abs(omhlist))
        plt.plot(alist,pwrlist6/mdotlist,'o')
    plt.ylim(ymin=0,ymax=3)
    #plt.plot(alist,2*mdotlist/(psitotsqlist)**2,'o')
    plt.plot( mya, myeta )
    plt.plot(mspina6[mhor6==hor],5*mpow6[mhor6==hor])
    plt.plot(mspina2[mhor2==hor],5*mpow2a[mhor2==hor])
    # psi = 1
    # mp = 2.0000 * 1.*1.0472*myomh**2 * 1.5*(psi**2-psi**3/3)
    # plt.plot(mya,1.5*mp)
    # plt.figure(2)
    # plt.clf()
    # plt.plot(mya,myomh)
    # plt.plot(mspina2[mhor2==hor],momh2[mhor2==hor])
    #
    plt.figure(4,figsize=(8,8))
    plt.clf()
    a0 = -1
    plot_spindown(a0,spar_func=spar_func,eta_func=eta_func,fntsize=fntsize)
    plt.savefig("retrospindown.pdf")
    #
    plt.figure(5,figsize=(8,8))
    plt.clf()
    a0 = 1
    plot_spindown(a0,spar_func=spar_func,eta_func=eta_func,fntsize=fntsize)
    plt.savefig("prospindown.pdf")

def plot_spindown(a0,spar_func=None,eta_func=None,fntsize=20):
    if spar_func is None or eta_func is None:
        print( "Need spar_func() and eta_func() for plotting")
        return
    t=np.linspace(0,1,num=10000)
    #plotting
    gs = GridSpec(3, 3)
    gs.update(left=0.12, right=0.98, top=0.95, bottom=0.1, wspace=0.25, hspace=0.04)
    ax1=plt.subplot(gs[0,:])
    ax2=plt.subplot(gs[1,:])
    ax3=plt.subplot(gs[2,:])
    aeq = brentq(spar_func,-1,1)
    print("Equilibrium spin: %g" % aeq)
    #initial value
    if a0 > 0:
        ax1.set_title(r"Spin-down of PROGRADE black holes",fontsize=0.8*fntsize)
    else:
        ax1.set_title(r"Spin-down of RETROGRADE black holes",fontsize=0.8*fntsize)
    a_of_t = odeint(lambda a,t: spar_func(a),a0,t)[:,0]
    a_of_t_func=interp1d(t,a_of_t,bounds_error=False)
    ax1.plot(t,a_of_t_func(t),"k-",lw=2)
    ax1.plot(t,t*0+aeq,"k:",lw=2)
    if a0 > 0:
        ax1.set_ylabel(r"$a$",ha="right",labelpad=3,fontsize=fntsize)
        ax1.set_ylim(-0.2,1.+1e-5)
    else:
        ax1.set_ylabel(r"$a$",ha="right",labelpad=3,fontsize=fntsize)
        ax1.set_ylim(-1.,0.2+1e-5)
    ax1.text(0.07, 0.5*aeq, r"$a=a_{\rm eq}$", size=fntsize, rotation=0.,
         ha="center", va="top",
         color='k',weight='regular'
         )
    #initial value
    lnM0 = 0
    lnM_of_t = odeint(lambda lnM,t: 1-0.01*eta_func(a_of_t_func(t)),lnM0,t)[:,0]
    M_of_t=exp(lnM_of_t)
    Mirr_of_t = M_of_t*(0.5*rhor_compute(a_of_t))**0.5
    ax2.plot(t,M_of_t,"k-",lw=2,label=r"$\mathrm{BH\ mass},\ M$")
    l1,=ax2.plot(t,Mirr_of_t,"b--",lw=2,label=r"$\mathrm{Irreducible\ BH\ mass},\ M_{\rm ir}$")
    l1.set_dashes([10,5])
    leg=ax2.legend(loc="lower right")
    for txts in leg.get_texts():
        txts.set_fontsize(0.7*fntsize)    # the legend text fontsize
    ax2.set_ylabel(r"$M,\ M_{\rm ir}$",ha="right",fontsize=fntsize)
    ax2.set_ylim(0.5,1.7)
    ax3.plot(t,eta_func(a_of_t_func(t)),"k-",lw=2)
    ax3.set_ylabel(r"$\eta$",ha="right",fontsize=fntsize)
    ax3.set_xlabel(r"$t/\tau$",fontsize=fntsize)
    if a0 > 0:
        ax3.set_yticks(np.arange(50,200,50))
        ax3.set_ylim(1e-5,150)
    else:
        ax3.set_yticks(np.arange(10,60,10))
        ax3.set_ylim(1e-5,50)
    #plt.plot(t,rhor_compute(a_of_t))
    ax1.grid(visible=True)
    ax2.grid(visible=True)
    ax3.grid(visible=True)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0,0.5-1e-5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(0.6*fntsize)
    for ax in [ax1, ax2]:
        plt.setp( ax.get_xticklabels(), visible=False)
    # tck=(0.5,1)
    # ax34.set_yticks(tck)
    # ax34.set_yticklabels(('','1'))


def poly1dt(poly_coef):
    return lambda x,t: sum(poly_coef[::-1]*x**np.arange(len(poly_coef)))

def readmytests1():
    global momh2, mhor2, mpsi2, mpow2, mBr2, mtheta2, mspina2, mpow2a, mpow2abz
    global momh4, mhor4, mpsi4, mpow4, mBr4, mtheta4, mspina4
    global momh6, mhor6, mpsi6, mpow6, mBr6, mtheta6, mspina6
    #
    gd2 = np.loadtxt( "mytest2", unpack = True )
    momh2, mhor2, mpsi2, mpow2, mBr2 = gd2[0:5]
    mtheta2 = np.pi/2-mhor2
    mspina2 = 4*momh2/(1+4*momh2**2)
    psi = (1-np.cos(np.pi/2-mhor2))
    mpow2a = 2.0000 * 1.*1.0472*momh2**2 * 1.5*(psi**2-psi**3/3)  #for two jets? 1.0472=pi/3
    momh2bz = mspina2/4.
    mpow2abz = 2.0000 * 1.*1.0472*momh2bz**2 * 1.5*(psi**2-psi**3/3)  #for two jets? 1.0472=pi/3
    #
    gd4 = np.loadtxt( "mytest4", unpack = True )
    momh4, mhor4, mpsi4, mpow4, mBr4 = gd4[0:5]
    mtheta4 = np.pi/2-mhor4
    mspina4 = 4*momh4/(1+4*momh4**2)
    #
    gd6 = np.loadtxt( "mytest6", unpack = True )
    momh6, mhor6, mpsi6, mpow6, mBr6 = gd6[0:5]
    mtheta6 = np.pi/2-mhor6
    mspina6 = 4*momh6/(1+4*momh6**2)

def getavgstd( vals, vals_std ):
    res = wmom(vals,vals_std**(-2),calcerr=True,sdev=True)
    return res[0], res[1]
 
def wmom(arrin, weights_in, inputmean=None, calcerr=False, sdev=False):
    """
    NAME:
      wmom()
      
    PURPOSE:
      Calculate the weighted mean, error, and optionally standard deviation of
      an input array.  By default error is calculated assuming the weights are
      1/err^2, but if you send calcerr=True this assumption is dropped and the
      error is determined from the weighted scatter.

    CALLING SEQUENCE:
     wmean,werr = wmom(arr, weights, inputmean=None, calcerr=False, sdev=False)
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean: 
          An input mean value, around which them mean is calculated.
      calcerr=False: 
          Calculate the weighted error.  By default the error is calculated as
          1/sqrt( weights.sum() ).  If calcerr=True it is calculated as sqrt(
          (w**2 * (arr-mean)**2).sum() )/weights.sum()
      sdev=False: 
          If True, also return the weighted standard deviation as a third
          element in the tuple.

    OUTPUTS:
      wmean, werr: A tuple of the weighted mean and error. If sdev=True the
         tuple will also contain sdev: wmean,werr,wsdev

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU

   """
    
    # no copy made if they are already arrays
    arr = np.array(arrin, ndmin=1, copy=False)
    
    # Weights is forced to be type double. All resulting calculations
    # will also be double
    weights = np.array(weights_in, ndmin=1, dtype='f8', copy=False)
  
    wtot = weights.sum()
    # user has input a mean value
    if inputmean is None:
        wmean = ( weights*arr ).sum()/wtot
    else:
        wmean=float(inputmean)

    # how should error be calculated?
    if calcerr:
        werr2 = ( weights**2 * (arr-wmean)**2 ).sum()
        werr = np.sqrt( werr2 )/wtot
        wstaterr = 1.0/np.sqrt(wtot)
        wtoterr = np.sqrt(werr**2+wstaterr**2)
    else:
        werr = 1.0/np.sqrt(wtot)
        wtoterr=werr

    # should output include the weighted standard deviation?
    if sdev:
        wvar = ( weights*(arr-wmean)**2 ).sum()/wtot * (weights.shape[0]-1.)/weights.shape[0]
        wsdev = np.sqrt(wvar)
        return wmean,wtoterr,wsdev
    else:
        return wmean,werr

def plotomegaf2hor():
    #plot omegaf2/omegah on the horizon
    plt.clf(); 
    rhor = 1 + (1-a**2)**0.5
    omh = a / (2*rhor)
    ihor = iofr(rhor)
    rhoavg=(rho[ihor].sum(1)/nz)
    bsqorhoavg=(bsq/rho)[ihor].sum(1)/nz
    TudEM10avg = (-gdet*TudEM[1,0])[ihor].sum(1)/nz
    Etot = (-gdet*TudEM[1,0])[ihor].sum()*_dx2*_dx3*2
    plt.plot(tj[ihor,:,0],omegaf2[ihor].sum(1)/nz*dxdxp[3][3][0,0,0]/omh); 
    plt.plot(tj[ihor,:,0],0.5*rhoavg/np.max(rhoavg)); 
    plt.plot(tj[ihor,:,0],bsqorhoavg/100); 
    plt.plot(tj[ihor,:,0],0.5*TudEM10avg/np.max(TudEM10avg)); 
    print( "Etot = %g" % Etot )
    plt.ylim(0,0.5)

def plotakshay():
    #grid3d("gdump.bin")
    #rfd("fieldline4670.bin")
    #cvel()
    #rhor = 1+(1-a**2)**0.5
    #ihor = iofr(rhor)
    iref = 0
    rref = r[iref,0,0]
    #far view
    plt.figure(1)
    plt.clf()
    #Density
    plt.suptitle(r"Angular density profile ($a = %g$ at $i = %d$, $r = %g$, $t = %g$)" % (a, iref, rref,t))
    plt.plot(h[iref,:,0],rho[iref,:,0])
    plt.plot(h[iref,:,0],rho[iref,:,0],'o')
    plt.xlim(0,3.5)
    plt.xlabel(r"$\theta-\pi/2$")
    plt.ylabel(r"$\rho$")
    plt.savefig("rho-i0.pdf")
    #close view
    plt.figure(2)
    plt.clf()
    #Density
    plt.suptitle(r"[blowup] Angular density profile ($a = %g$ at $i = %d$, $r = %g$, $t = %g)$" % (a, iref, rref,t))
    plt.plot(h[iref,:,0]-np.pi/2,rho[iref,:,0])
    plt.plot(h[iref,:,0]-np.pi/2,rho[iref,:,0],'o')
    plt.xlim(-0.2,0.2)
    plt.xlabel(r"$\theta-\pi/2$")
    plt.ylabel(r"$\rho$")
    plt.savefig("rho-i0_blowup.pdf")
    #Time step
    #dt2 = _dx2 / amax(np.abs(v2m),np.abs(v2p)) 
    #plt.plot(h[iref,:,0]-np.pi/2,dt2*1000)
 
def plotrfisco():
    plco(v1p,levels=(0,),xcoord=r*np.sin(h),ycoord=r*np.cos(h),colors='c'); plt.xlim(0,10); plt.ylim(-5,5)
    plc(np.log10(rho),xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True)
    rh=1+(1-a**2)**0.5
    plc(r-rh,levels=(0,),colors='k',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
    risco = Risco(a)
    print( "Risco(a=%g) = %g" % (a, risco) )
    plc(r*np.sin(h)-risco,levels=(0,),colors='g',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
    plt.grid()

def plotaphivsr():
    plt.clf();plt.semilogx(r[:,0,0],aphi[:,ny/2,0]/10,label=r'$\Phi/10$')
    plt.semilogx(r[:,0,0],15+lrho[:,ny/2,0],label=r'$\log_{10}\rho+15$')
    plt.semilogx(r[:,0,0],20+np.log10(ug[:,ny/2,0]),label=r'$\log_{10}u_g\!+20$')
    plt.semilogx(r[:,0,0],(r[:,0,0]-34)**(2./3.)/2.1,label=r'$\propto (r-r_{\rm max})^{2/3}\!$')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$r$')
    plt.ylim(0,20)

def div( vec ):
    res = np.zeros_like(vec[1])
    res[0:-1]  = (vec[1][1:]-vec[1][:-1])
    return( res )

def omegah_compute(a):
    rh = rhor_compute(a)
    omegah = 0.5 * a / rh
    return( omegah )

def rhor_compute(a):
    return( 1+(1-a**2)**0.5 )

def plotdiv():    
    global madded, eadded
    rh = 1+(1-a**2)**0.5
    omegah=a/2/rh
    etad = np.zeros_like(uu)
    mdotden = (gdet[:,:,0:1]*avg_rhouu)[1]
    mdotval = scaletofullwedge(nz*(mdotden*_dx2*_dx3).sum(axis=-1).sum(-1))
    edotval = -scaletofullwedge(nz*(gdet*avg_Tud[1,0]*_dx2*_dx3).sum(axis=-1).sum(-1))
    ldotval = scaletofullwedge(nz*(gdet*avg_Tud[1,3]/dxdxp[3][3][:,:,0:1]*_dx2*_dx3).sum(axis=-1).sum(-1))
    etad[0] = -1/(-gn3[0,0])**0.5
    gdetdivrhouu=div(gdet[:,:,0:1]*avg_rhouu)
    etot = scaletofullwedge(nz*(-gdet*avg_Tud[1,0]*_dx2*_dx3).sum(axis=-1).sum(-1))
    madded = scaletofullwedge(nz*(gdetdivrhouu*_dx2*_dx3).sum(axis=-1)).sum(-1).cumsum(0)
    madded -= madded[iofr(10)]
    eadded = scaletofullwedge(nz*(etad[0]*gdetdivrhouu*_dx2*_dx3).sum(axis=-1)).sum(-1).cumsum(0)
    eadded -= eadded[iofr(10)]
    eadded/=nz
    plt.figure(10)
    #xxx
    #plt.plot(r[:,0,0], mdotval-mdotval[iofr(10)],label="mdotval")
    #$plt.plot(r[:,0,0], madded,label="madded")
    #plt.plot(r[:,0,0], eadded,label="eadded")
    plt.plot(r[:,0,0], edotval,label="Etot")
    plt.plot(r[:,0,0], ldotval/10.,label="Ltot/10")
    #plt.plot(r[:,0,0], ldotval*omegah,label="Omegah * Ltot")
    plt.plot(r[:,0,0], -mdotval,label="Mtot")
    plt.ylim(0,10)
    plt.legend(loc='lower right')
    rh = 1+(1-a**2)**0.5
    plt.xlim(rh,30)
    #plt.ylim(-20,20)
    #res[:,1:-1] += (vec[2][:,2:]-vec[2][:,:-2])  

def ploteta():
    #FIGURE 1 LOTSOPANELS
    #Figure 1
    #To make plot, run 
    #run ~/py/mread/__init__.py 1 1
    #To re-make plot without reloading the fiels, run
    #run ~/py/mread/__init__.py 1 -1
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
    #AT: plt.legend( loc = 'upper left', bbox_to_anchor = (0.5, 0.5) ) #0.5, 0.5 = center of plot
    #To generate movies for all sub-folders of a folder:
    #cd ~/Research/runart; for f in *; do cd ~/Research/runart/$f; (python  ~/py/mread/__init__.py &> python.out &); done
    grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]), use2d=True )
    #rd( "dump0000.bin" )
    #rfd("fieldline0000.bin")  #to definea
    #grid3dlight("gdump")
    qtymem=None #clear to free mem
    rhor=1+(1+a**2)**0.5
    ihor = np.floor(iofr(rhor)+0.5);
    qtymem=getqtyvstime(ihor,0.2)
    fig=plt.figure(0, figsize=(12,6), dpi=100)
    plt.clf()
    #
    #pjet/<mdot>
    #
    ax34 = plt.gca()
    plotqtyvstime(qtymem,ax=ax34,whichplot=4,prefactor=1)
    ymax=ax34.get_ylim()[1]
    if 1 < ymax and ymax < 2: 
        #ymax = 2
        tck=(1,)
        ax34.set_yticks(tck)
        #ax34.set_yticklabels(('','1','2'))
    elif ymax < 1: 
        ymax = 1
        tck=(0.5,1)
        ax34.set_yticks(tck)
        ax34.set_yticklabels(('','1'))
    else:
        ymax=np.floor(ymax)+1
        tck=np.arange(1,ymax)
        ax34.set_yticks(tck)
    #reset lower limit to 0
    ax34.set_ylim((0,ax34.get_ylim()[1]))
    ax34.grid(True)
    ax34.set_ylabel(r"$\eta$")
    # plt.text(ax34.get_xlim()[1]/40., 0.8*ax34.get_ylim()[1], r"$(\mathrm{g})$", size=16, rotation=0.,
    #          ha="center", va="center",
    #          color='k',weight='regular',bbox=bbox_props
    #          )
    ax34r = ax34.twinx()
    ax34r.set_ylim(ax34.get_ylim())
    ax34r.set_yticks(tck)
    gc.collect()

def mkmovie(framesize=50, domakeavi=False,**kwargs):
    #Rz and xy planes side by side
    plotlenf=10
    plotleni=framesize
    plotlenti=1e6 #so high that never gets used
    plotlentf=2e6
    #To generate movies for all sub-folders of a folder:
    #cd ~/Research/runart; for f in *; do cd ~/Research/runart/$f; (python  ~/py/mread/__init__.py &> python.out &); done
    if len(sys.argv[1:])==2 and sys.argv[1].isdigit() and (sys.argv[2].isdigit() or sys.argv[2][0]=="-") :
        whichi = int(sys.argv[1])
        whichn = int(sys.argv[2])
        print( "Doing every %d slice of total %d slices" % (whichi, whichn) )
        sys.stdout.flush()
    else:
        whichi = None
        whichn = None
    if whichn < 0 and whichn is not None:
        whichn = -whichn
        dontloadfiles = True
    else:
        dontloadfiles = False
        grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]), use2d=True )
        #rd( "dump0000.bin" )
        #rfd("fieldline0000.bin")  #to definea
        #grid3dlight("gdump")
        qtymem=None #clear to free mem
        rhor=1+(1+a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        qtymem=getqtyvstime(ihor,0.2)
        flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9].bin") ) )

    for findex, fname in enumerate(flist):
        if findex % whichn != whichi:
            continue
        if dontloadfiles == False and os.path.isfile("lrho%04d_Rzxym1.png" % (findex)):
            print( "Skipping " + fname + " as lrho%04d_Rzxym1.png exists" % (findex) );
        else:
            print( "Processing " + fname + " ..." )
            sys.stdout.flush()
            kwargs['plotleni']=plotleni
            kwargs['plotlenf']=plotlenf
            kwargs['plotlenti']=plotlenti
            kwargs['plotlentf']=plotlentf
            kwargs['qtymem']=qtymem
            mkmovieframe( findex, fname, **kwargs )
    print( "Done!" )
    sys.stdout.flush()
    if domakeavi:
        #print( "Now you can make a movie by running:" )
        #print( "ffmpeg -fflags +genpts -r 10 -i lrho%04d.png -vcodec mpeg4 -qmax 5 mov.avi" )
        os.system("mv mov_%s_Rzxym1.avi mov_%s_Rzxym1.bak.avi" % ( os.path.basename(os.getcwd()), os.path.basename(os.getcwd())) )
        #os.system("ffmpeg -fflags +genpts -r 20 -i lrho%%04d_Rzxym1.png -vcodec mpeg4 -qmax 5 mov_%s_Rzxym1.avi" % (os.path.basename(os.getcwd())) )
        os.system("ffmpeg -fflags +genpts -r 20 -i lrho%%04d_Rzxym1.png -vcodec mpeg4 -qmax 5 -b 10000k -pass 1 mov_%s_Rzxym1p1.avi" % (os.path.basename(os.getcwd())) )
        os.system("ffmpeg -fflags +genpts -r 20 -i lrho%%04d_Rzxym1.png -vcodec mpeg4 -qmax 5 -b 10000k -pass 2 mov_%s_Rzxym1.avi" % (os.path.basename(os.getcwd())) )
        #os.system("scp mov.avi 128.112.70.76:Research/movies/mov_`basename \`pwd\``.avi")

def mkmovieframe( findex, fname, **kwargs ):
    dostreamlines = kwargs.pop('dostreamlines',True)
    frametype = kwargs.get('frametype','5panels')
    prefactor = kwargs.get('prefactor',1.)
    sigma = kwargs.get('sigma',None)
    usegaussianunits = kwargs.get('usegaussianunits',False)
    domakeframes = kwargs.get('domakeframes',True)
    epsFm = kwargs.get('epsFm',None)
    epsFm30 = kwargs.get('epsFm30',None)
    epsFke = kwargs.get('epsFke',None)
    epsetaj = kwargs.get('epsetaj',None)
    fti = kwargs.get('fti',None)
    ftf = kwargs.get('ftf',None)
    plotleni = kwargs.get('plotleni',50)
    plotlenf = kwargs.get('plotlenf',50)
    plotlenti = kwargs.get('plotlenti',1e6)
    plotlentf = kwargs.get('plotlentf',2e6)
    qtymem = kwargs.get('qtymem',2e6)
    # oldnz=nz
    rfd("../"+fname)
    # if oldnz < nz:
    #     #resolution changed on the fly, get correct-size arrays for r, h, ph
    #     rd("dump0147.bin")
    #     #reread the fieldline dump
    #     rfd("../"+fname)
    cvel() #for calculating bsq
    plotlen = plotleni+(plotlenf-plotleni)*(t-plotlenti)/(plotlentf-plotlenti)
    plotlen = min(plotlen,plotleni)
    plotlen = max(plotlen,plotlenf)
    plt.figure(0, figsize=(12,9), dpi=100)
    plt.clf()
    if frametype=='5panels':
        #SWITCH OFF SUPTITLE
        #plt.suptitle(r'$\log_{10}\rho$ at t = %4.0f' % t)
        #mdot,pjet,pjet/mdot plots
        gs3 = GridSpec(3, 3)
        #gs3.update(left=0.055, right=0.97, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
        #gs3.update(left=0.055, right=0.95, top=0.42, bottom=0.03, wspace=0.01, hspace=0.04)
        gs3.update(left=0.055, right=0.97, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
        #mdot
        ax31 = plt.subplot(gs3[-3,:])
        plotqtyvstime(qtymem,ax=ax31,whichplot=1,findex=findex,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,usegaussianunits=True)
        ymax=ax31.get_ylim()[1]
        ymax=2*(np.floor(np.floor(ymax+1.5)/2))
        ax31.set_yticks((ymax/2,ymax))
        ax31.grid(True)
        bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
        placeletter(ax31,"$(\mathrm{c})$",fx=0.02,bbox=bbox_props)
        ax31r = ax31.twinx()
        ax31r.set_ylim(ax31.get_ylim())
        ax31r.set_yticks((ymax/2,ymax))
        #pjet
        # ax32 = plt.subplot(gs3[-2,:])
        # plotqtyvstime(qtymem,ax=ax32,whichplot=2)
        # ymax=ax32.get_ylim()[1]
        # ax32.set_yticks((ymax/2,ymax))
        # ax32.grid(True)
        #pjet/mdot
        # ax33 = plt.subplot(gs3[-1,:])
        # plotqtyvstime(qtymem,ax=ax33,whichplot=3)
        # ymax=ax33.get_ylim()[1]
        # ax33.set_yticks((ymax/2,ymax))
        # ax33.grid(True)
        #
        #\phi
        #
        ax35 = plt.subplot(gs3[-2,:])
        plotqtyvstime(qtymem,ax=ax35,whichplot=5,findex=findex,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,usegaussianunits=True)
        ymax=ax35.get_ylim()[1]
        if 1 < ymax and ymax < 2: 
            #ymax = 2
            tck=(1,)
            ax35.set_yticks(tck)
            #ax35.set_yticklabels(('','1','2'))
        elif ymax < 1: 
            ymax = 1
            tck=(0.5,1)
            ax35.set_yticks(tck)
            ax35.set_yticklabels(('','1'))
        else:
            ymax=np.floor(ymax)+1
            if ymax >= 60:
                tck=np.arange(1,ymax/30.)*30.
            elif ymax >= 20:
                tck=np.arange(1,ymax/10.)*10.
            elif ymax >= 10:
                tck=np.arange(1,ymax/5.)*5.
            else:
                tck=np.arange(1,ymax)
            ax35.set_yticks(tck)
        ax35.grid(True)
        placeletter(ax35,"$(\mathrm{d})$",fx=0.02,bbox=bbox_props)
        if ymax >= 10:
            ax35.set_ylabel(r"$\phi_{\rm BH}$",size=16,ha='left',labelpad=25)
        ax35.grid(True)
        ax35r = ax35.twinx()
        ax35r.set_ylim(ax35.get_ylim())
        ax35r.set_yticks(tck)
        #
        #pjet/<mdot>
        #
        ax34 = plt.subplot(gs3[-1,:])
        plotqtyvstime(qtymem,ax=ax34,whichplot=4,findex=findex,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,usegaussianunits=True)
        #OVERRIDE
        #ax34.set_ylim((-.5*prefactor,1.99*prefactor))
        ax34.set_ylim((0,3.8*prefactor))
        placeletter(ax34,"$(\mathrm{e})$",fx=0.02,bbox=bbox_props)
        ymax=ax34.get_ylim()[1]
        ymin=ax34.get_ylim()[0]
        if ymin < -.25 * prefactor:
            ymin = -.25 * prefactor
            ax34.set_ylim((ymin,ymax))
        if prefactor < ymax and ymax < 1.5*prefactor: 
            #ymax = 2
            tck=(0.5*prefactor,prefactor,)
            if ymin < 0:
                tck=(0,0.5*prefactor,prefactor,)
            ax34.set_yticks(tck)
            #ax34.set_yticklabels(('','100','200'))
        elif ymax <= prefactor: 
            ymax=np.floor(ymax)+1
            if ymin < 0:
                minval = 0
            else:
                minval = 1
            if ymax >= 50:
                tck=np.arange(minval,ymax/50.)*50.
            elif ymax >= 20:
                tck=np.arange(minval,ymax/10.)*10.
            elif ymax >= 10:
                tck=np.arange(minval,ymax/5.)*5.
            else:
                tck=np.arange(minval,ymax)
            ax34.set_yticks(tck)
            if False:
                ymax = prefactor
                tck=(0.5*prefactor,prefactor)
                if ymin < 0:
                    tck=(0,0.5*prefactor,prefactor)
                ax34.set_yticks(tck)
                if ymin >= 0:
                    ax34.set_yticklabels(('','%d' % prefactor))
        else:
            ymax=np.floor(ymax/prefactor)+1
            ymax*=prefactor
            tck=np.arange(1,ymax/prefactor)*prefactor
            if ymin < 0:
                tck=np.arange(0,ymax/prefactor)*prefactor
            ax34.set_yticks(tck)
        #reset lower limit to 0
        #ax34.set_ylim((0,ax34.get_ylim()[1]))
        ax34.grid(True)
        ax34r = ax34.twinx()
        ax34r.set_ylim(ax34.get_ylim())
        ax34r.set_yticks(tck)
        #Rz xy
        gs1 = GridSpec(1, 1)
        gs1.update(left=0.04, right=0.45, top=0.995, bottom=0.48, wspace=0.05)
        #gs1.update(left=0.05, right=0.45, top=0.99, bottom=0.45, wspace=0.05)
        ax1 = plt.subplot(gs1[:, -1])
        if domakeframes:
            mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False)
        ax1.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        placeletter(ax1,"$(\mathrm{a})$",va="center",bbox=bbox_props)
        gs2 = GridSpec(1, 1)
        gs2.update(left=0.5, right=1, top=0.995, bottom=0.48, wspace=0.05)
        ax2 = plt.subplot(gs2[:, -1])
        if domakeframes:
            mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=True,pt=False,dostreamlines=True)
        ax2.set_ylabel(r'$y\ [r_g]$',fontsize=16,ha='center')
        ax2.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        placeletter(ax2,"$(\mathrm{b})$",va="center",bbox=bbox_props)
    elif frametype=='Rzpanel':
        #Rz xy
        gs1 = GridSpec(1, 1)
        gs1.update(left=0.04, right=0.45, top=0.995, bottom=0.48, wspace=0.05)
        #gs1.update(left=0.05, right=0.45, top=0.99, bottom=0.45, wspace=0.05)
        ax1 = plt.subplot(gs1[:, -1])
        if domakeframes:
            mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=dostreamlines,ncont=50)
        ax1.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        # gs2 = GridSpec(1, 1)
        # gs2.update(left=0.5, right=1, top=0.995, bottom=0.48, wspace=0.05)
        # ax2 = plt.subplot(gs2[:, -1])
        # ax2.set_ylabel(r'$y\ [r_g]$',fontsize=16,ha='center')
        # ax2.set_xlabel(r'$x\ [r_g]$',fontsize=16)
    elif frametype=='Rzzypanels':
        #Rz xy
        gs1 = GridSpec(1, 1)
        gs1.update(left=0.04, right=0.45, top=0.995, bottom=0.48, wspace=0.05)
        #gs1.update(left=0.05, right=0.45, top=0.99, bottom=0.45, wspace=0.05)
        ax1 = plt.subplot(gs1[:, -1])
        if domakeframes:
            mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False)
        ax1.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        gs2 = GridSpec(1, 1)
        gs2.update(left=0.5, right=1, top=0.995, bottom=0.48, wspace=0.05)
        ax2 = plt.subplot(gs2[:, -1])
        ax2.set_ylabel(r'$y\ [r_g]$',fontsize=16,ha='center')
        ax2.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        if domakeframes:
            mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=True,pt=False,dostreamlines=True)
    #print xxx
    plt.savefig( "lrho%04d_Rzxym1.png" % (findex),bbox_inches='tight',pad_inches=0.02  )
    plt.savefig( "lrho%04d_Rzxym1.eps" % (findex),bbox_inches='tight',pad_inches=0.02  )
    plt.savefig( "lrho%04d_Rzxym1.pdf" % (findex),bbox_inches='tight',pad_inches=0.02  )
    #print xxx

def mk2davg():
    if len(sys.argv[1:])!=0:
        grid3d("gdump.bin",use2d=True)
        #rd("dump0000.bin")
        rfd("fieldline0000.bin")
    if len(sys.argv[1:])==2 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        whichgroup = int(sys.argv[1])
        step = int(sys.argv[2])
        itemspergroup = 20
        for whichgroup in np.arange(whichgroup,1000,step):
            avgmem = get2davg(whichgroup=whichgroup,itemspergroup=itemspergroup)
        #plot2davg(avgmem)
    elif len(sys.argv[1:])==3 and sys.argv[1].isdigit() and sys.argv[2].isdigit() and sys.argv[3].isdigit():
        whichgroups = int(sys.argv[1])
        whichgroupe = int(sys.argv[2])
        step = int(sys.argv[3])
        itemspergroup = 20
        if step == 0:
            avgmem = get2davg(usedefault=1)
        elif step == 1:
            avgmem = get2davg(whichgroups=whichgroups,whichgroupe=whichgroupe,itemspergroup=itemspergroup)
        else:
            for whichgroup in np.arange(whichgroups,whichgroupe,step):
                avgmem = get2davg(whichgroup=whichgroup,itemspergroup=itemspergroup)
        print( "Assigning averages..." )
        assignavg2dvars(avgmem)
    else:
        return
    plot2davg(whichplot=1)
    gc.collect()

def getFe2davg(aphi_j_val=0):
    #sum(axis=2) is summation in phi (since 1 element in phi anyway, this just removes the phi-index)
    eout1den   = scaletofullwedge(nz*(-gdet*avg_Tud[1,0]*_dx2*_dx3).sum(axis=2))
    eout1denEM = scaletofullwedge(nz*(-gdet*avg_TudEM[1,0]*_dx2*_dx3).sum(axis=2))
    eout1denMA = scaletofullwedge(nz*(-gdet*avg_TudMA[1,0]*_dx2*_dx3).sum(axis=2))
    #subtract off rest-energy flux
    eout1denKE = scaletofullwedge(nz*(gdet*(-avg_TudMA[1,0]-avg_rhouu[1])*_dx2*_dx3).sum(axis=2))
    #
    #need to shift these so 0 at axis
    eoutEM1 = eout1denEM.cumsum(axis=1)
    eoutMA1 = eout1denMA.cumsum(axis=1)
    eoutKE1 = eout1denKE.cumsum(axis=1)
    #shift
    eout1 = eoutEM1+eoutMA1
    eouttot = eout1den.sum(axis=-1)
    #
    #MA
    #
    powMAtot = cutout_along_aphi(eoutMA1,aphi_j_val=0)
    plt.figure(5)
    plt.plot(powMAtot)
    plt.plot(eoutMA1[:,ny-1])
    plt.figure(6)
    plt.plot(eoutMA1[:,ny-1]-powMAtot)

###
### Fix near-BH solution for aphi_j_val = 1 or 2
###
def cutout_along_aphi(ecum,aphi_j_val=0):
    #
    ndim = ecum.ndim
    if aphi_j_val > 0:
        if ndim == 3:
            return( ecum[:,:,ny-1-aphi_j_val] - ecum[:,:,aphi_j_val-1] )
        else:
            return( ecum[:,ny-1-aphi_j_val] - ecum[:,aphi_j_val-1] )
    elif aphi_j_val == 0:
        if ndim == 3:
            return( ecum[:,:,ny-1] )
        else:
            return( ecum[:,ny-1] )
    else:
        if ndim == 3:
            return( ecum[:,:,ny-2] - ecum[:,:,0] )
        else:
            return( ecum[:,ny-2] - ecum[:,0] )
    #get avg_aphi
    avgmem = get2davg(usedefault=1)
    assignavg2dvars(avgmem)
    #face-centered aphi
    avg_aphi_stag = fieldcalcface(gdetB1=avg_gdetB[0])
    avg_aphi = np.zeros_like(avg_aphi_stag)
    #shift by one cell to be consistent in x2-location with eout1's
    avg_aphi[:,0:ny-1] = avg_aphi_stag[:,1:ny]
    #make it cell centered in radius
    avg_aphi[0:nx-1] = 0.5*(avg_aphi[0:nx-1]+avg_aphi[1:nx])
    #
    aphi_cut_val = avg_aphi[iofr(rhor),aphi_j_val,0]
    ecumleft  = findroot2d( avg_aphi[:,:,0]-aphi_cut_val, ecum, isleft=True, fallback = 1, fallbackval = 0 )
    ecumright = findroot2d( avg_aphi[:,:,0]-aphi_cut_val, ecum, isleft=False, fallback = 1, fallbackval = np.pi )
    ecumtot = ecumright-ecumleft
    return( ecumtot )
    

def mkonestreamline(u, x0, y0, mylen=30):
    """Despite scary-looking contents, this extracts and returns a single streamline starting at (x0, y0); mylen is the size of the square within which a field line is to be traced"""
    B[1:] = u[1:]
    traj = mkframe("myframe",len=mylen,ax=None,density=24,downsample=1,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=False,dodiskfield=False,minlenbhfield=0.2,minlendiskfield=0.1,dsval=0.005,color='k',doarrows=False,dorandomcolor=False,lw=1,skipblankint=True,detectLoops=False,ncell=800,minindent=5,minlengthdefault=0.2,startatmidplane=False,startxabs=x0,startyabs=y0)
    return( traj )

def mkonestreamlinex1x2(ux, uy, xi, yi, x0, y0):
    """ compute a streamline in x1-x2 (internal) code coordinates
        Example inputs:
            ux = B[1,:,:,0]
            uy = B[2,:,:,0]
            xi = x1[:,:,0]
            yi = x2[:,:,0]
            starting point:
            x0 = value of x1
            y0 = value of x2
        Output:
            Tuple of (x1[:], x2[:])
        Example:
        #trace a field line starting at x1 = 1, x2 = 0
        traj = mkonestreamlinex1x2( B[1,:,:,0], B[2,:,:,0], x1[:,0,0], x2[0,:,0], 1, 0 )
        #alternatively, you could trace energy streamline
        traj = mkonestreamlinex1x2( Tud[1,0,:,:,0], Tud[2,0,:,:,0], x1[:,0,0], x2[0,:,0], 1, 0 )
        xtraj, ytraj = traj
        #plot it
        plt.figure()
        plt.plot(xtraj,ytraj)
        #evaluate streamline coordinates at radial grid nodes, x1[0:nx]:
        x1traj = x1[:,0,0]
        x2traj = interp1d(xtraj, ytraj, kind='linear',bounds_error=False)(x1traj)
        #Evaluate, e.g., density along streamline:
        #  Note: if trajectory has multiple values of x2 for a single value of x1, then 
        #        isleft = True picks the solution for rho at the smallest value of x2 and 
        #        isleft = False picks the solution for rho at the largest value of x2
        #this evaluates rho where the 1st argument vanishes, i.e., x2[:,:,0]-ytrajvsti[:,None] == 0
        rhotraj = findroot2d(x2[:,:,0]-x2traj[:,None], rho[:,:,0], axis = 0, isleft = True )
        #plot it
        plt.figure()
        plt.plot(np.log10(r[:,0,0]),np.log10(rhotraj))
    """
    if ux.ndim==3:
        ux=ux[:,:,0]
    if uy.ndim==3:
        uy=uy[:,:,0]
    if xi.ndim==3:
        xi=xi[:,0,0]
    if yi.ndim==3:
        yi=yi[0,:,0]
    if xi.ndim==2:
        xi=xi[:,0]
    if yi.ndim==2:
        yi=yi[0,:]
    traj = fstreamplot(yi,xi,uy,ux,ua=None,va=None,ax=None,density=24,downsample=1,dobhfield=False,dodiskfield=False,minlenbhfield=0.2,minlendiskfield=0.1,dsval=0.005,color='k',doarrows=False,dorandomcolor=False,skipblankint=True,detectLoops=False,minindent=5,minlengthdefault=0.2,startatmidplane=False,startxabs=y0,startyabs=x0)
    if traj is not None:
        if traj[1][0] > x0:
            #order such that x1 increases along streamline
            return( traj[1][::-1], traj[0][::-1] )
        return( traj[1], traj[0] )
    else:
        return( None )

def mkmanystreamlinesx1x2(): 
    startxabs=2
    a_startyabs=np.linspace(0.583,0.583,num=1)
    #DRAW FIGURE
    #fig=plt.figure(1,figsize=(12,9),dpi=300)
    fig=plt.figure(1)
    ax = fig.add_subplot(111)
    #fig=plt.figure(1)
    fntsize=24
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.set_aspect('equal')   
    #TRACE ENERGY STREAMLINE
    mylen = 30
    grid3d("gdump.bin",use2d=True)
    rfd("fieldline0000.bin")
    avgmem = get2davg(usedefault=1)
    assignavg2dvars(avgmem)
    avg_aphi = scaletofullwedge(nz*_dx3*fieldcalc(gdetB1=avg_gdetB[0]))
    plco(avg_aphi,xcoord=x1[:,:,0],ycoord=x2[:,:,0])
    #energy
    #B[1:] = avg_Tud[1:,0]
    for startyabs in a_startyabs:
        print( "x0 = %g, y0 = %g" % (startxabs, startyabs) )
        traj = mkonestreamlinex1x2( avg_B[0,:,:,0], avg_B[1,:,:,0], x1[:,0,0], x2[0,:,0], startxabs, startyabs )
        if traj is not None:
            xtraj, ytraj = traj
            #DRAW STREAMLINE
            plt.plot(xtraj,ytraj,'g-')
            yfunc = interp1d(xtraj, ytraj, kind='linear',bounds_error=False)
            #pdb.set_trace()
            #check how well the field line is extracted
            xtrajvsti = x1[:,0,0]
            ytrajvsti = yfunc(xtrajvsti)
            my_aphi = findroot2d(x2[:,:,0]-ytrajvsti[:,None], avg_aphi, axis = 0 )
            plt.figure()
            plt.plot(x1[:,0,0],my_aphi)
            # findroot2d(, vals, axis = 0 )
            # ax2
            plt.draw()

def finddiskjetbnds(r0=5,upperx2=True,maxiter=100,eps=1e-14,doplot=True):
    """upperx2 = False means look for low-x2 boundary; dir = True means look for high-x2 boundary"""
    #starting point
    x0 = x1[iofr(r0),0,0]
    y0 = 0.5*(x2.min() + x2.max())
    ymax = max(abs(x2.min()),abs(x2.max()))
    if upperx2:
        ypole = x2.max()
        ydisk = y0
    else:
        ypole = x2.min()
        ydisk = y0
    trajold = None
    traj = None
    xh = x1[iofr(rhor),ny/2,0]
    trajdisk = None
    trajpole = None
    for i in xrange(maxiter):
        y = 0.5*(ydisk+ypole)
        #save old traj
        trajold = traj
        traj = mkvelsline(x0,y)
        if traj[0].min() < xh:
            ydisk = y
            trajdisk = traj
        else:
            ypole = y
            trajpole = traj
        if trajdisk is None:
            trajdisk = mkvelsline(x0,ydisk)
        if trajpole is None:
            trajpole = mkvelsline(x0,ypole)
        if doplot:
            #plot current trajectory
            plt.plot(traj[0],traj[1],'g')
            plt.draw()
        #break if reached accuracy
        if np.abs(ypole-ydisk)<eps*ymax:
            print( "Reached given accuracy, %g" % eps )
            break
        if i == maxiter-2:
            print( "Reached max iter number, %d" % maxiter )
    return( trajdisk, trajpole)
        

def mkvelsline(x0,y0):
    """Makes one streamline passing through (x0,y0), returns the min value of x
       along the streamline
    """
    global avg_uu, x1, x2
    traj = mkonestreamlinex1x2( avg_uu[1,:,:,0], avg_uu[2,:,:,0], x1[:,0,0], x2[0,:,0], x0, y0 )
    return( traj )
    
    
def mkmanystreamlinesxy():
    startxabs=2
    a_startyabs=np.linspace(-6,6,num=2)
    #DRAW FIGURE
    #fig=plt.figure(1,figsize=(12,9),dpi=300)
    fig=plt.figure(2)
    ax2 = fig.add_subplot(111)
    fig=plt.figure(1)
    fntsize=24
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_aspect('equal')   
    #TRACE ENERGY STREAMLINE
    mylen = 30
    grid3d("gdump.bin",use2d=True)
    rfd("fieldline0000.bin")
    avgmem = get2davg(usedefault=1)
    assignavg2dvars(avgmem)
    #energy
    #B[1:] = avg_Tud[1:,0]
    for startyabs in a_startyabs:
        print( "x0 = %g, y0 = %g" % (startxabs, startyabs) )
        traj = mkonestreamline( -avg_Tud[:,0], startxabs, startyabs, mylen = mylen )
        if traj is not None:
            xtraj, ytraj = traj
            #DRAW STREAMLINE
            ax.plot(xtraj,ytraj,'g-')
            # findroot2d(, vals, axis = 0 )
            # ax2
            plt.draw()
        else:
            print("Got Null trajectory: (%f,%f)" % (startxabs, startyabs))
            continue
    rhor=1+(1-a**2)**0.5
    el = Ellipse((0,0), 2*rhor, 2*rhor, facecolor='k', alpha=1)
    art=ax.add_artist(el)
    art.set_zorder(20)
    mylenshow = 25./30.*mylen
    plt.xlim(-mylenshow,mylenshow)
    plt.ylim(-mylenshow,mylenshow)
    plt.xlabel(r"$x\ [r_g]$",fontsize=fntsize,ha='center')
    plt.ylabel(r"$z\ [r_g]$",ha='left',labelpad=15,fontsize=fntsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fntsize)
    # plt.savefig("fig2.pdf",bbox_inches='tight',pad_inches=0.02)
    # plt.savefig("fig2.eps",bbox_inches='tight',pad_inches=0.02)
    plt.savefig("fig2oneline.png",bbox_inches='tight',pad_inches=0.02,dpi=300)

def removefloorsavg2d(usestaggeredfluxes=False,DFfloor=None):
    """ Removes floors and returns a tuple, (Fm, FmMinusFe1, FmMinusFe2), from which floors were removed.
        Does not multiply the result by _dx2*_dx3 -- you should do so yourself if you wish.
        When removing the floors assumes that the flow is aligned with the radial grid lines (x2=const).
        In reality, this is not exactly correct, however, most of the floor addition happens 
        close to the BH, where the grid is very much radial, so this approximation works quite well.
    """
    if DFfloor is None:
        DFfloor=takeoutfloors(ax=None,doreload=1,dotakeoutfloors=True,dofeavg=0,isinteractive=0,writefile=False,doplot=False,aphi_j_val=0, ndim=2, is_output_cell_center = False)
    if 'avg_gdetF' in globals() and not avg_gdetF[0,0].any() or \
            usestaggeredfluxes == False:
        is_output_cell_center = True
        print( "Using gdet*avg_rhouu[1]" )
        enden1=(-gdet*avg_Tud[1,0]-gdet*avg_rhouu[1])*nz
        enden2=(-gdet*avg_Tud[2,0]-gdet*avg_rhouu[2])*nz
        mdden=(-gdet*avg_rhouu[1])*nz
    else:
        print( "Using avg_gdetF" )
        is_output_cell_center = False
        #x1-fluxes of:
        #0,0 mass   
        #0,1 energy 
        #0,2 ang.m. 
        #x2-fluxes of:
        #0,0 mass   
        #0,1 energy 
        #0,2 ang.m. 
        enden1=(-avg_gdetF[0,1]*nz)
        enden2=(-avg_gdetF[1,1]*nz)
        mdden =(-avg_gdetF[0,0]*nz)
    dotakeoutfloors=True
    if dotakeoutfloors:
        #subtract rest-mass from total energy flux and flip the sign to get correct direction
        DFen = DFfloor[1]+DFfloor[0] 
        #pdb.set_trace()
        enden1 += DFen[:,:,None]/(_dx2*_dx3) 
        mdden += DFfloor[0][:,:,None]/(_dx2*_dx3)
    if is_output_cell_center == False:
        #en[:-1]=0.5*(en[:-1]+en[1:])
        enden1[:-1]=0.5*(enden1[1:]+enden1[:-1])
        enden2[:,:-1]=0.5*(enden2[:,1:]+enden2[:,:-1])
        mdden[:-1]=0.5*(mdden[:-1]+mdden[1:])
    Fm_floorremoved = mdden
    FmMinusFe_floorremoved1 = enden1
    FmMinusFe_floorremoved2 = enden2
    return( Fm_floorremoved, FmMinusFe_floorremoved1, FmMinusFe_floorremoved2 )


def get_fluxes(usestaggeredfluxes=False):
    if 'avg_gdetF' in globals() and not avg_gdetF[0,0].any() or \
            usestaggeredfluxes == False:
        is_output_cell_center = True
        print( "Using gdet*avg_rhouu[1]" )
        enden1=(-gdet*avg_Tud[1,0]-gdet*avg_rhouu[1])*nz
        enden2=(-gdet*avg_Tud[2,0]-gdet*avg_rhouu[2])*nz
        mdden=(-gdet*avg_rhouu[1])*nz
    else:
        print( "Using avg_gdetF" )
        is_output_cell_center = False
        #x1-fluxes of:
        #0,0 mass   
        #0,1 energy 
        #0,2 ang.m. 
        #x2-fluxes of:
        #0,0 mass   
        #0,1 energy 
        #0,2 ang.m. 
        enden1=(-avg_gdetF[0,1]*nz)
        enden2=(-avg_gdetF[1,1]*nz)
        mdden =(-avg_gdetF[0,0]*nz)
        enden1[:-1]=0.5*(enden1[1:]+enden1[:-1])
        enden2[:,:-1]=0.5*(enden2[:,1:]+enden2[:,:-1])
        mdden[:-1]=0.5*(mdden[:-1]+mdden[1:])
    #factor to rescale all fluxes to full 2pi wedge
    wedge_scale_factor = scaletofullwedge(1.)
    #scale flux densities to full wedge
    enden1 *= wedge_scale_factor
    enden2 *= wedge_scale_factor
    mdden *= wedge_scale_factor
    #Fm, (Fm-Fe)x1, (Fm-Fe)x2
    return mdden, enden1, enden2

def extract_along_x2vsi(var,x2vsi,isleft=True,fallback=1,fallbackval = np.pi/2,shiftx2=None):
    if shiftx2 is None:
        shiftx2 = 0.5*_dx2
    #shiftx2 is used to account for center to face difference in locations (half a cell = 0.5*_dx2)
    res = findroot2d( x2[:,:,0] - x2vsi[:,None] + shiftx2, var, isleft=isleft, fallback = fallback, fallbackval = fallbackval )
    return(res)


def removefloorsavg2djetwind(usestaggeredfluxes=False,DFfloor=None, jet1x2=None, jet2x2=None, dotakeoutfloors = True):
    """ Removes floors and returns a tuple, (Fm, FmMinusFe1, FmMinusFe2), from which floors were removed.
        Does not multiply the result by _dx2*_dx3 -- you should do so yourself if you wish.
        When removing the floors assumes that the flow is aligned with the radial grid lines (x2=const).
        In reality, this is not exactly correct, however, most of the floor addition happens 
        close to the BH, where the grid is very much radial, so this approximation works quite well.
    """
    avg_enth=1+avg_ug*gam/avg_rho
    avg_unb1=avg_enth*avg_ud[0]
    avg_isunbound=(-avg_unb1>1.0)
    #
    RR=0
    TH=1
    PH=2
    #x2 = -1 at h = 0
    #x2 =  1 at h = pi
    #get fluxes
    mdden, enden1, enden2 = get_fluxes(usestaggeredfluxes=usestaggeredfluxes)
    ######################
    #
    #  Fluxes
    #
    ######################
    if DFfloor is None:
        DFfloor=takeoutfloors(ax=None,doreload=1,dotakeoutfloors=True,dofeavg=0,isinteractive=0,writefile=False,doplot=False,aphi_j_val=0, ndim=-2, is_output_cell_center = False)
    #process floors
    DUin, DUout = DFfloor
    #subtract back (by adding) the rest-mass floor from energy, so get floor on Fm-Fe
    DUin[1] += DUin[0]
    DUout[1] += DUout[0]
    DUinden=DUin
    DUoutden=DUout
    #integrate DUin/DUout in theta
    DUin = DUin.cumsum(1+TH)
    DUout = DUout.cumsum(1+TH)
    ######################
    #
    #  Fluxes
    #
    ######################
    Fmcum = mdden.cumsum(TH).sum(PH)*_dx2*_dx3
    Fxcum = enden1.cumsum(TH).sum(PH)*_dx2*_dx3  #Fx = FmMinusFe
    #Convert to tuples
    DUin  = DUin[0],  DUin[1]
    DUout = DUout[0], DUout[1]
    Fcum = Fmcum, Fxcum
    #INSERT ANGLE FILTERING HERE
    #
    #JET1
    #
    DUin_jet1=np.array(extract_along_x2vsi(DUin,jet1x2,fallbackval=0))
    DUout_jet1=np.array(extract_along_x2vsi(DUout,jet1x2,fallbackval=0))
    F_jet1=np.array(extract_along_x2vsi(Fcum,jet1x2,fallbackval=0))
    #pdb.set_trace()
    #
    #JET2
    #
    DUin_jet2cum=np.array(extract_along_x2vsi(DUin,jet2x2,isleft=False,fallbackval=0))
    DUout_jet2cum=np.array(extract_along_x2vsi(DUout,jet2x2,isleft=False,fallbackval=0))
    F_jet2cum=np.array(extract_along_x2vsi(Fcum,jet2x2,isleft=False,fallbackval=0))
    #convert to arrays
    DUin = np.array(DUin)
    DUout = np.array(DUout)
    Fcum = np.array(Fcum)
    #subtract from other axis
    DUin_jet2=DUin[:,:,ny-1]-DUin_jet2cum
    DUout_jet2=DUout[:,:,ny-1]-DUout_jet2cum
    F_jet2 = Fcum[:,:,ny-1]-F_jet2cum
    #
    #WIND
    #
    DUin_wind=DUin_jet2cum-DUin_jet1
    DUout_wind=DUout_jet2cum-DUout_jet1
    F_wind=F_jet2cum-F_jet1
    #
    #now combine into 1D floor corrections for jet1, wind, and jet2
    #pick out something that's not too far
    fnx = iofr(100)
    #
    #JET1
    #
    DFin_jet1  = DUin_jet1.cumsum(1+RR)
    DFout_jet1 = DUout_jet1.cumsum(1+RR)
    DF_jet1 = (DFin_jet1-DFin_jet1[:,fnx-1:fnx]) + DFout_jet1
    #
    #JET2
    #
    DFin_jet2  = DUin_jet2.cumsum(1+RR)
    DFout_jet2 = DUout_jet2.cumsum(1+RR)
    DF_jet2 = (DFin_jet2-DFin_jet2[:,fnx-1:fnx]) + DFout_jet2
    #
    #WIND
    #
    DFin_wind  = DUin_wind.cumsum(1+RR)
    DFout_wind = DUout_wind.cumsum(1+RR)
    DF_wind = (DFin_wind-DFin_wind[:,fnx-1:fnx]) + DFout_wind

    if dotakeoutfloors:
        #subtract rest-mass from total energy flux and flip the sign to get correct direction
        #DFen = DF[1]+DF[0] 
        F_jet1 += DF_jet1[0:2]
        F_jet2 += DF_jet2[0:2]
        F_wind += DF_wind[0:2]

    return( F_jet1, F_jet2, F_wind )

def mkstreamlinefigure(length=25,doenergy=False,frac=0.75,frameon=True,dpi=300,showticks=True,usedefault=2,fc='white',mc='white',dotakeoutfloors=0,showtitle=False):
    #fc='#D8D8D8'
    global bsq, ug, mu, B, DF, qtymem
    mylen = length/frac
    arrowsize=4
    grid3d("gdump.bin",use2d=True)
    rfd("fieldline0000.bin")
    avgmem = get2davg(usedefault=usedefault)
    assignavg2dvars(avgmem)
    fig=plt.figure(1,figsize=(4,3),frameon=frameon)
    fig.patch.set_facecolor(fc)
    fig.patch.set_alpha(1.0)
    fntsize=8
    ax = fig.add_subplot(111, aspect='equal', frameon=frameon)
    if dotakeoutfloors:
        #Do this first, which uses a lot of memory and do the rest after this memory is freed up
        DFfloor=takeoutfloors(ax=None,doreload=1,dotakeoutfloors=dotakeoutfloors,dofeavg=0,isinteractive=0,writefile=False,doplot=False,aphi_j_val=0, ndim=2, is_output_cell_center = False)
        #
        #Find accretion rate
        #
        # res = takeoutfloors(doreload=True,isinteractive=0,writefile=False)
        # a_eta,a_Fm,a_Fe,a_Fl = res
        avgmem = get2davg(usedefault=1)
        assignavg2dvars(avgmem)
        rho = avg_rho
        bsq = avg_bsq
    if doenergy==False and True:
        #velocity
        qty=avg_uu
        #
        #mass flow
        #qty=avg_rhouu
        #
        #angular momentum flow
        #qty=avg_Tud[:,3]
        if True:
            qty[2,:,-1]*=0
            qty[2,:,-2]*=0
            qty[2,:,0]*=0
            qty[2,:,1]*=0
        #avg_uu[1,:,-3:-1] = np.abs(avg_uu[1,:,-3:-1])
        # avg_uu[1,:,-1]=avg_uu[1,:,-4]
        # avg_uu[1,:,-2]=avg_uu[1,:,-4]
        # avg_uu[1,:,-3]=avg_uu[1,:,-4]
        #avg_uu[1,:,0:3] = np.abs(avg_uu[1,:,0:3])
        # avg_uu[1,:,0]=avg_uu[1,:,3]
        # avg_uu[1,:,1]=avg_uu[1,:,3]
        # avg_uu[1,:,2]=avg_uu[1,:,3]
        #B[1:] = avg_uu[1:]
        B[1:] = qty[1:]
        bsq = avg_bsq
        mkframe("myframe",len=mylen,ax=ax,density=24,downsample=1,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=False,dodiskfield=False,minlenbhfield=0.2,minlendiskfield=0.5,dsval=0.0025,color='k',doarrows=False,dorandomcolor=True,lw=1,skipblankint=True,detectLoops=False,ncell=800,minindent=5,minlengthdefault=0.2,startatmidplane=False)
    if doenergy==True and False:
        #energy
        B[1:] = avg_Tud[1:,0]
        bsq = avg_bsq
        mkframe("myframe",len=mylen,ax=ax,density=24,downsample=1,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=False,dodiskfield=False,minlenbhfield=0.2,minlendiskfield=0.5,dsval=0.005,color='k',doarrows=False,dorandomcolor=True,lw=1,skipblankint=True,detectLoops=False,ncell=800,minindent=5,minlengthdefault=0.2,startatmidplane=False)
    if False:
        #energy vectors
        B[1:] = -avg_Tud[1:,0]
        bsq = avg_bsq
        mkframe("myframe",len=mylen,ax=ax,density=4,downsample=4,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=12,dodiskfield=True,minlenbhfield=0.2,minlendiskfield=0.5,dsval=0.005,color='r',lw=2,startatmidplane=True,showjet=False,arrowsize=arrowsize)
    if False:
        #KE+EM without floors
        B[1:] = -avg_Tud[1:,0]-avg_rhouu[1:]
        bsq = avg_bsq
        gdetB[1:] = avg_gdetB[0:]
        mu = avg_mu
        # mkframe("myframe",len=mylen,ax=ax,density=4,downsample=1,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=False,dodiskfield=False,minlenbhfield=0.2,minlendiskfield=0.5,dsval=0.0025,color='k',doarrows=False,dorandomcolor=True,lw=1,skipblankint=True,detectLoops=False,ncell=800,minindent=5,minlengthdefault=0.2,startatmidplane=False)
        mkframe("myframe",len=mylen,ax=ax,density=1,downsample=4,cb=False,pt=False,dorho=False,dovarylw=False,vmin=-6,vmax=0.5,dobhfield=28,dodiskfield=0,minlenbhfield=0.1,minlendiskfield=0.1,dsval=0.001,color='r',lw=0.5,startatmidplane=True,showjet=False,arrowsize=arrowsize,skipblankint=True,populatestreamlines=False,useblankdiskfield=False,dnarrow=4,whichr=15)
    if True:
        #KE+EM without floors with contourf
        #energy flow (no rest-mass) vs. radius and theta
        #FLR: here replace this with actual flux of energy: gdetF12 (if non-zero, which means if defined; account for face-location!)
        #     provide interface through takeoutfloors to return "floor-corrected" 2D arrays of mass and energy flows
        #     maybe just make a call to takeoutfloors() (or similar function) that would return energy and mass fluxes
        if os.path.basename(os.getcwd()) == "rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2":
            #^^^ hack ^^^ to avoid using this for all models except a = -0.9 model (for now)
            #this is because some of the models were restarted half-way with this diagnostic added,
            #and so their averages will not be correct (since the missing data is filled with zeros)
            #---> saved face-centered fluxes exist
            usestaggeredfluxes = True
            is_output_cell_center = False
        else:
            is_output_cell_center = True
            usestaggeredfluxes = False
        #######################
        #
        # REMOVE FLOORS
        #
        #######################
        Fm_floorremoved, FmMinusFe_floorremoved1, FmMinusFe_floorremoved2  = removefloorsavg2d(usestaggeredfluxes=usestaggeredfluxes,DFfloor=DFfloor)
        enden1 = FmMinusFe_floorremoved1
        enden = enden1
        enden2 = FmMinusFe_floorremoved2
        mdden = Fm_floorremoved
        en=(enden.cumsum(1)-0.5*enden)*_dx2*_dx3 #subtract half of current cell's density to get cell-centered quantity
        md=(mdden).sum(2).sum(1)*_dx2*_dx3
        #pdb.set_trace()
        #mdot vs. radius
        #pick out a scalar value at r = 5M
        md=md[iofr(5)]
        a_Fm=md
        a_FmminusFe=enden.sum(1)[iofr(5),0]*_dx2*_dx3
        a_Fe1=md-a_FmminusFe
        a_eta = a_FmminusFe/md
        #equatorial trajectory: starts at r = rh, theta = pi/2
        rhor=1+(1-a**2)**0.5
        radval=10.
        if False and is_output_cell_center == True:
            #internal fluxes not available
            traj = mkonestreamlinex1x2( -avg_Tud[1,0,:,:,0]-avg_rhouu[1,:,:,0],
                                    -avg_Tud[2,0,:,:,0]-avg_rhouu[2,:,:,0],
                                    x1[:,0,0],x2[0,:,0],
                                    x1[iofr(radval),ny/2,0],0.)
        else:
            #internal fluxes available: use them (they have already been reinterpolated to cell centers)
            traj = mkonestreamlinex1x2( enden1[:,:,0],
                                    enden2[:,:,0],
                                    x1[:,0,0],x2[0,:,0],
                                    x1[iofr(radval),ny/2,0],0.)
        xtraj,ytraj=traj
        x1traj=x1[:,0,0]
        x2traj=interp1d(xtraj, ytraj, kind='linear',bounds_error=False)(x1traj)
        entraj=findroot2d(x2[:,:,0]-x2traj[:,None], en[:,:,0], axis = 0, isleft = True )
        #change zero to be at the equatorial field line
        en=en-entraj[:,None,None]
        r2=np.concatenate((r[:,::-1],r),axis=1)
        h2=np.concatenate((-h[:,::-1],h),axis=1)
        en2=np.concatenate((en[:,::-1],en),axis=1)
        #adjust the last cell positions to ensure last contour is closed
        h2[:,0]=h2[:,0]*0-np.pi*1.
        h2[:,-1]=h2[:,-1]*0+np.pi*1.
        #plc(en2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=True,nc=20,isfilled=True)
        if False: 
            #normalization by max energy flux so energy flux goes from 0 to 100%
            z = np.abs(en2)/np.nanmax(np.abs(en2))
            zhalf = np.abs(en)/np.nanmax(np.abs(en))
        else:
            #normalization by mass accretion rate; 2 to account for each hemisphere is half
            z = 2*np.abs(en2)/a_Fm
            zhalf =  2*np.abs(en)/a_Fm
            #print md, a_Fm, a_Fe1, a_Fe
        # lev_exp = np.arange(np.floor(np.log10(np.nanmin(z))-1),
        #                      np.ceil(np.log10(np.nanmax(z))+1))
        #lev_exp=np.linspace(lminval,lmaxval,nc)
        # levs=[1e-3,1e-2,2e-2]
        if np.abs(a-0.9)<1e-2:
            #a=0.9
            levs_label = np.array([0.125,0.25,0.5,1.])*0.16 #*Ebindisco(a)
            levs = np.arange(0.125,1.125,0.125)*0.16 #*Ebindisco(a)
        elif np.abs(a-0.99)<1e-2:
            #a=0.9
            levs_label = np.array([0.125,0.25,0.5,1.])*0.18 #*Ebindisco(a)
            levs = np.arange(0.125,1.125,0.125)*0.18 #*Ebindisco(a)
        else:
            #a=-0.9
            levs_label = np.array([0.125,0.25,0.5,1.])*0.07 #Ebindisco(a)
            levs = np.arange(0.125,1.125,0.125)*0.07 #Ebindisco(a)
        hofz1=interp1d( zhalf[iofr(rhor),ny/2-5:0:-1,0], h[iofr(rhor),ny/2-5:0:-1,0], kind='linear' )
        hofz2=interp1d( zhalf[iofr(rhor),ny/2+5:,0], h[iofr(rhor),ny/2+5:,0], kind='linear' )
        print( "theta1 = %g, theta2 = %g\n" % (hofz1(levs[-1]), hofz2(levs[-1])) )
        theta1 = 1.34995
        theta2 = 1.82129
        zofh=interp1d( h[iofr(rhor),:,0],zhalf[iofr(rhor),:,0], kind='linear' )
        print( "z(h=%g) = %g, z(h=%g) = %g\n" % (theta1, zofh(theta1), theta2, zofh(theta2)) )
        minval=levs[0]
        maxval=levs[-1]
        lminval=np.log10(minval)
        lmaxval=np.log10(maxval)
        lev_exp = np.log10(levs)
        cutval=0.1*minval
        z[z<cutval]=z[z<cutval]*0+cutval
        #plco(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=True,nc=20,levels=levs,isfilled=True,norm=colors.LogNorm())
        #palette =  cm.autumn_r #mpl.colors.ListedColormap(['r', 'g', 'b'])
        palette =  mpl.colors.ListedColormap([ 'purple', 'blue', 'green', 'green', 'yellow', 'yellow', 'yellow', 'yellow','red'])
        #palette.set_over('red')
        #palette.set_under('blue')
        alpha=0.25
        ctsf=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=levs,isfilled=True,alpha=alpha,zorder=2,cmap=palette,extend='both') 
        #cts=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=ctsf.levels[0::1],isfilled=False,alpha=alpha,zorder=2,linestyles='solid',linewidths=0.5,colors='r')
        # Make a colorbar for the ContourSet returned by the contourf call.
        #ctsf.cmap.set_under('green',-2)
        #ctsf.cmap.set_under('red',1.0)
        cbar = plt.colorbar(ctsf)
        cbar.ax.set_ylabel('Enclosed energy outflow efficiency',fontsize=fntsize)
        cbar.set_alpha(1.0)
        # Add the contour line levels to the colorbar
        #cbar.add_lines(cts)
        print lev_exp
        tcks=[x for x in levs_label]
        labs=['%d%%'%(x*100+0.5) for x in levs_label]
        for (i,x) in enumerate(levs_label):
            if x==np.rint(x):
                labs[i] = '%d%%' % (np.rint(x*100.))
            else:
                #elif 10*x==np.rint(10*x):
                labs[i] = '%g%%' % (np.rint(x*1000.)/10.)
        #else:
        #    labs[i] = '%g%%' % (np.rint(x*10000.)/100.)
        cbar.set_ticks(tcks)
        cbar.set_ticklabels(labs)
        cbar.update_ticks()
        #set font size of colorbar tick labels
        cl = plt.getp(cbar.ax, 'ymajorticklabels')
        plt.setp(cl, fontsize=fntsize)
        #pdb.set_trace()
        #plt.xlim(-30,30); plt.ylim(-30,30)
        mylenshow = frac*mylen
        plt.xlim(-mylenshow,mylenshow)
        plt.ylim(-mylenshow,mylenshow)
        #pdb.set_trace()
    if False:
        #Angular momentum flow plot
        if not avg_gdetF[0,0].any():
            #saved face-centered fluxes exist
            is_output_cell_center = True
            enden1=(gdet*avg_Tud[1,3])*nz
            enden2=(gdet*avg_Tud[2,3])*nz
            enden=enden1
            mdden=(-gdet*avg_rhouu[1])*nz
        else:
            is_output_cell_center = False
            #x1-fluxes of:
            #0,0 mass   
            #0,1 energy 
            #0,2 ang.m. 
            #x2-fluxes of:
            #1,0 mass  
            #1,1 energy 
            #1,2 ang.m. 
            enden1=(avg_gdetF[0,2]*nz)
            enden2=(avg_gdetF[1,2]*nz)
            enden=enden1
            mdden =(-avg_gdetF[0,0]*nz)
        if False and dotakeoutfloors:
            DFfloor=takeoutfloors(ax=None,doreload=1,dotakeoutfloors=dotakeoutfloors,dofeavg=0,isinteractive=0,writefile=False,doplot=False,aphi_j_val=0, ndim=2, is_output_cell_center = False)
            #subtract rest-mass from total energy flux and flip the sign to get correct direction
            DFen = DFfloor[4]
            #pdb.set_trace()
            enden += DFen[:,:,None]/(_dx2*_dx3)
            mdden += DFfloor[0][:,:,None]/(_dx2*_dx3)
        en=(enden.cumsum(1)-0.5*enden)*_dx2*_dx3 #subtract half of current cell's density to get cell-centered quantity
        md=(mdden).sum(2).sum(1)*_dx2*_dx3
        #pdb.set_trace()
        if is_output_cell_center == False:
            en[:-1]=0.5*(en[:-1]+en[1:])
            enden1[:-1]=0.5*(enden1[1:]+enden1[:-1])
            enden2[:,:-1]=0.5*(enden2[:,1:]+enden2[:,:-1])
            md[:-1]=0.5*(md[:-1]+md[1:])
        #mdot vs. radius
        #pick out a scalar value at r = 5M
        md=md[iofr(5)]
        #equatorial trajectory: starts at r = rh, theta = pi/2
        rhor=1+(1-a**2)**0.5
        radval=10.
        # traj = mkonestreamlinex1x2( enden1[:,:,0],
        #                         enden2[:,:,0],
        #                         x1[:,0,0],x2[0,:,0],
        #                         x1[iofr(radval),ny/2,0],0.)
        # xtraj,ytraj=traj
        # x1traj=x1[:,0,0]
        # x2traj=interp1d(xtraj, ytraj, kind='linear',bounds_error=False)(x1traj)
        # entraj=findroot2d(x2[:,:,0]-x2traj[:,None], en[:,:,0], axis = 0, isleft = True )
        # #change zero to be at the equatorial field line
        # en=en-entraj[:,None,None]
        #change zero to be at the equator, r = 10
        en=en-0.5*(en[iofr(10),ny/2,0]+en[iofr(10),ny/2-1,0])
        r2=np.concatenate((r[:,::-1],r),axis=1)
        h2=np.concatenate((-h[:,::-1],h),axis=1)
        en2=np.concatenate((en[:,::-1],en),axis=1)
        #adjust the last cell positions to ensure last contour is closed
        h2[:,0]=h2[:,0]*0-np.pi*1.
        h2[:,-1]=h2[:,-1]*0+np.pi*1.
        #plc(en2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=True,nc=20,isfilled=True)
        cond=(r2[:,:,0]<mylen*2**0.5)
        z = en2/np.nanmax(np.abs(en2[cond]))
        # minval=1e-3
        # cutval=1*minval
        # z[z<cutval]=z[z<cutval]*0+cutval
        # # lev_exp = np.arange(np.floor(np.log10(np.nanmin(z))-1),
        # #                      np.ceil(np.log10(np.nanmax(z))+1))
        # lev_exp=np.linspace(np.log10(minval),0,10)
        # levs = np.power(10, lev_exp)
        levs=np.linspace(-1,1,100)
        #plco(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=True,nc=20,levels=levs,isfilled=True,norm=colors.LogNorm())
        if False:
            #log
            ctsf=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,nc=20,isfilled=True,locator=ticker.LogLocator(),alpha=0.25,zorder=2,cmap=cm.jet)
            cts=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=ctsf.levels[1::1],isfilled=False,locator=ticker.LogLocator(),alpha=0.25,zorder=2,linestyles='solid',linewidths=0.5,colors='r')
        else:
            ctsf=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=levs,isfilled=True,alpha=0.25,zorder=2,cmap=cm.jet)
            cts =plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=ctsf.levels[1::1],isfilled=False,alpha=0.25,zorder=2,linestyles='solid',linewidths=0.5,colors='r')
        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = plt.colorbar(ctsf)
        #cbar.ax.set_ylabel('verbosity coefficient')
        # Add the contour line levels to the colorbar
        cbar.add_lines(cts)
        #pdb.set_trace()
        #plt.xlim(-30,30); plt.ylim(-30,30)
        mylenshow = frac*mylen
        plt.xlim(-mylenshow,mylenshow)
        plt.ylim(-mylenshow,mylenshow)
        #pdb.set_trace()
    if False:
        #u_\phi plot
        en=avg_ud[3]/dxdxp[3,3,:,:,0:1] #subtract half of current cell's density to get cell-centered quantity
        r2=np.concatenate((r[:,ny-1:ny],r[:,::-1],r,r[:,ny-1:ny]),axis=1)
        h2=np.concatenate((-h[:,ny-1:ny],-h[:,::-1],h,h[:,ny-1:ny]),axis=1)
        en2=np.concatenate((en[:,ny-1:ny],en[:,::-1],en,en[:,ny-1:ny]),axis=1)
        #adjust the last cell positions to ensure last contour is closed
        h2[:,0]=h2[:,0]*0-np.pi*1.
        h2[:,-1]=h2[:,-1]*0+np.pi*1.
        z=en2
        cond=(r2[:,:,0]<mylen*2**0.5)
        maxudphi=np.nanmax(np.abs(en2[cond]))
        maxudphi=np.floor(maxudphi)
        # # lev_exp = np.arange(np.floor(np.log10(np.nanmin(z))-1),
        # #                      np.ceil(np.log10(np.nanmax(z))+1))
        # lev_exp=np.linspace(np.log10(minval),0,10)
        # levs = np.power(10, lev_exp)
        #levs=np.linspace(0,maxudphi,17)
        levs=np.arange(-maxudphi,maxudphi+1,1)
        #plco(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=True,nc=20,levels=levs,isfilled=True,norm=colors.LogNorm())
        if False:
            #log
            ctsf=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,nc=20,isfilled=True,locator=ticker.LogLocator(),alpha=0.25,zorder=2,cmap=cm.jet)
            cts=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=ctsf.levels[1::1],isfilled=False,locator=ticker.LogLocator(),alpha=0.25,zorder=2,linestyles='solid',linewidths=0.5,colors='r')
        else:
            ctsf=plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=levs,isfilled=True,alpha=0.25,zorder=2,cmap=cm.RdYlGn_r) #cmap=cm.hot_r)
            clrs=[]
            for i,lev in enumerate(ctsf.levels):
                if(lev<0):
                    clrs.append('green')
                elif(lev>0):
                    clrs.append('red')
                else:
                    clrs.append((0.4,0.26,0.13))  #dark brown
            cts =plc(z,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),cb=False,levels=ctsf.levels,isfilled=False,alpha=0.25,zorder=2,linestyles='solid',linewidths=0.5,colors=clrs)
        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = plt.colorbar(ctsf)
        #cbar.ax.set_ylabel('verbosity coefficient')
        # Add the contour line levels to the colorbar
        #pdb.set_trace()
        cbar.add_lines(cts)
        #pdb.set_trace()
        #plt.xlim(-30,30); plt.ylim(-30,30)
        mylenshow = frac*mylen
        plt.xlim(-mylenshow,mylenshow)
        plt.ylim(-mylenshow,mylenshow)
        #pdb.set_trace()
    if True:
        istag, jstag, hstag, rstag = getstagparams(doplot=0,usedefault=usedefault)
        linewidth=1
        myRmax=4
        #z>0
        rs=rstag[(rstag*np.sin(hstag)<myRmax)*np.cos(hstag)>0]
        hs=hstag[(rstag*np.sin(hstag)<myRmax)*np.cos(hstag)>0]
        hs2=np.concatenate((-hs[::-1],hs),axis=1)
        rs2=np.concatenate((rs[::-1],rs),axis=1)
        ax.plot(rs2*np.sin(hs2),rs2*np.cos(hs2),'g',lw=linewidth,zorder=21)
        #z<0
        rs=rstag[(rstag*np.sin(hstag)<myRmax)*np.cos(hstag)<0]
        hs=hstag[(rstag*np.sin(hstag)<myRmax)*np.cos(hstag)<0]
        hs2=np.concatenate((hs,-hs[::-1]),axis=1)
        rs2=np.concatenate((rs,rs[::-1]),axis=1)
        ax.plot(rs2*np.sin(hs2),rs2*np.cos(hs2),'g',lw=linewidth,zorder=21)
    if True:
        avg_aphi = fieldcalc(gdetB1=avg_gdetB[0])
        if dotakeoutfloors:
            #normalize avg_phi by Mdot to get dimensionless flux
            phibh = (4*np.pi)**0.5*avg_aphi/a_Fm**0.5
            avg_aphi = phibh
            step=10
            levs=np.arange(step,100*step,step)
            #for a = 0.99 increase the nuumber of contours by 2x so see more detail
            if np.abs(a-0.99)<0.01:
                levs = levs/2.
        else:
            levs=None
        r2=np.concatenate((r[:,::-1],r),axis=1)
        h2=np.concatenate((-h[:,::-1],h),axis=1)
        avg_aphi2=np.concatenate((avg_aphi[:,::-1],avg_aphi),axis=1)
        if levs is not None:
            plc(avg_aphi2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),levels=levs,colors='black',linewidths=0.5)
            cnt=plc(avg_aphi2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),levels=levs,colors='white',linewidths=0.5,linestyles='dashed')
        else:
            plc(avg_aphi2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),nc=30,colors='black',linewidths=0.5)
            cnt=plc(avg_aphi2,xcoord=r2*np.sin(h2),ycoord=r2*np.cos(h2),nc=30,colors='white',linewidths=0.5,linestyles='dashed')
        #cnt.set_dashes([fntsize,fntsize])
    if False:
        #field
        B[1] = avg_B[0]
        B[2] = avg_B[1]
        B[3] = avg_B[2]
        bsq = avg_bsq
        plt.figure(1)
        gdetB[1:] = avg_gdetB[0:]
        mu = avg_mu
        mkframe("myframe",len=mylen,ax=ax,density=1,downsample=4,cb=False,pt=False,dorho=False,dovarylw=True,vmin=-6,vmax=0.5,dobhfield=12,dodiskfield=8,minlenbhfield=0.1,minlendiskfield=0.1,dsval=0.001,color='b',lw=1.5,startatmidplane=True,showjet=False,arrowsize=arrowsize,skipblankint=True,populatestreamlines=False,useblankdiskfield=False,dnarrow=1)
    if False:
        x = (r*np.sin(h))[:,:,0]
        z = (r*np.cos(h))[:,:,0]
        x = np.concatenate(-x,x)
        z = np.concatenate(y,y)
        mu = np.concatenate(avg_mu[:,:,0],avg_mu[:,:,0])
        plt.contourf( x, z, mu )
    ax.set_aspect('equal')   
    rhor=1+(1-a**2)**0.5
    el = Ellipse((0,0), 2*rhor, 2*rhor, facecolor='k', alpha=1)
    art=ax.add_artist(el)
    art.set_zorder(20)
    mylenshow = frac*mylen
    plt.xlim(-mylenshow,mylenshow)
    plt.ylim(-mylenshow,mylenshow)
    if showticks == True:
        plt.xlabel(r"$x\ [r_g]$",fontsize=fntsize,ha='center')
        plt.ylabel(r"$z\ [r_g]$",ha='left',fontsize=fntsize) #labelpad=15,
    else:
        plt.setp( ax.get_xticklabels(), visible=False)
        plt.setp( ax.get_yticklabels(), visible=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fntsize)
    # run through all lines drawn for xticks and yticks
    if showticks == False:
        for i, line in enumerate(ax.get_xticklines() + ax.get_yticklines()):
            #if i%2 == 1:   # odd indices
            line.set_visible(False)     
    # plt.savefig("fig2.pdf",bbox_inches='tight',pad_inches=0.02)
    # plt.savefig("fig2.eps",bbox_inches='tight',pad_inches=0.02)
    bbox = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.5)
    if showtitle:
        if a < 0:
            plt.title(r"${\rm Retrograde\ BH,\ a = %g,\ \eta = %d\%%\ (model\ A-0.9f})$" % (a, np.rint(a_eta*100.)), fontsize=fntsize )
            placeletter( plt.gca(),"$\mathrm{(a)}$",bbox=bbox,size=fntsize*1.5,ha='center',va='center')
        else:
            plt.title(r"${\rm Prograde\ BH,\ a = %g,\ \eta = %d\%%\ (model\ A0.9f})$" % (a, np.rint(a_eta*100.)), fontsize=fntsize )
            placeletter( plt.gca(),"$\mathrm{(b)}$",bbox=bbox,size=fntsize*1.5,ha='center',va='center')
    # haxes=pylab.axes()
    # haxes.yaxis.LABELPAD=0
    fig.patch.set_facecolor(mc)
    fig.patch.set_alpha(0.5)
    fig.patch.set_visible(True)
    ax.patch.set_facecolor(fc)
    ax.patch.set_alpha(0.5)
    ax.patch.set_visible(False)
    plt.savefig("fig2.png",bbox_inches='tight',pad_inches=0.02,dpi=dpi,facecolor=fig.get_facecolor(), edgecolor='none',transparent=False)
    #plt.savefig("fig2.png",bbox_inches='tight',pad_inches=0.02,dpi=dpi,facecolor=fig.get_facecolor(),transparent=True)
    #fig.get_facecolor()

def mklotsopanels(doreload=1,epsFm=None,epsFke=None,epsetaj=None,epsFm30=None,fti=None,ftf=None,domakeframes=True,prefactor=100,sigma=None,usegaussianunits=False,arrowsize=1):
    global qtymem
    #Figure 1
    #To make plot, run 
    #run ~/py/mread/__init__.py 1 1
    #To re-make plot without reloading the fiels, run
    #run ~/py/mread/__init__.py 1 -1
    doslines=True
    plotlenf=10
    plotleni=25
    plen=plotleni
    plotlenti=40000
    plotlentf=45000
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
    #AT: plt.legend( loc = 'upper left', bbox_to_anchor = (0.5, 0.5) ) #0.5, 0.5 = center of plot
    #To generate movies for all sub-folders of a folder:
    #cd ~/Research/runart; for f in *; do cd ~/Research/runart/$f; (python  ~/py/mread/__init__.py &> python.out &); done
    if len(sys.argv[1:])==2 and sys.argv[1].isdigit() and (sys.argv[2].isdigit() or sys.argv[2][0]=="-") :
        whichi = int(sys.argv[1])
        whichn = int(sys.argv[2])
        print( "Doing every %d slice of total %d slices" % (whichi, whichn) )
        sys.stdout.flush()
    else:
        whichi = None
        whichn = None
    if whichn < 0 and whichn is not None:
        whichn = -whichn
        dontloadfiles = True
    else:
        dontloadfiles = False
        grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]), use2d=True )
        #rd( "dump0000.bin" )
        rfd("fieldline0000.bin")  #to definea
        #grid3dlight("gdump")
        rhor=1+(1+a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        if doreload == 1:
            qtymem=getqtyvstime(ihor,0.2)
        flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline*.bin") ) )
    #make accretion rate plot, etc.
    sys.stdout.flush()
    plotlen = plotleni+(plotlenf-plotleni)*(t-plotlenti)/(plotlentf-plotlenti)
    plotlen = min(plotlen,plotleni)
    plotlen = max(plotlen,plotlenf)
    fig=plt.figure(0, figsize=(12,9), dpi=100)
    plt.clf()
    #findexlist=(0,600,1285,1459)
    #findexlist=(0,600,1225,1369)
    #findexlist=(0,600,1225,3297)
    #findexlist=(0,600,5403,5468)
    #findexlist=(0,600,3297,5403)
    findexlist=(0,1157,3297,5403)
    #SWITCH OFF SUPTITLE
    #plt.suptitle(r'$\log_{10}\rho$ at t = %4.0f' % t)
    #mdot,pjet,pjet/mdot plots
    findex = 0
    gs3 = GridSpec(3, 3)
    gs3.update(left=0.055, right=0.97, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
    #mdot
    ax31 = plt.subplot(gs3[-3,:])
    plotqtyvstime(qtymem,ax=ax31,whichplot=1,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma) #AT: need to specify index!
    ymax=ax31.get_ylim()[1]
    ymax=2*(np.floor(np.floor(ymax+1.5)/2))
    ax31.set_yticks((ymax/2,ymax))
    #ax31.set_xlabel(r"$t\ [r_g/c]")
    ax31.grid(True)
    plt.text(ax31.get_xlim()[1]/40., 0.8*ax31.get_ylim()[1], "$(\mathrm{e})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    ax31r = ax31.twinx()
    ax31r.set_ylim(ax31.get_ylim())
    ax31r.set_yticks((ymax/2,ymax))
    #pjet
    # ax32 = plt.subplot(gs3[-2,:])
    # plotqtyvstime(qtymem,ax=ax32,whichplot=2)
    # ymax=ax32.get_ylim()[1]
    # ax32.set_yticks((ymax/2,ymax))
    # ax32.grid(True)
    #pjet/mdot
    # ax33 = plt.subplot(gs3[-1,:])
    # plotqtyvstime(qtymem,ax=ax33,whichplot=3)
    # ymax=ax33.get_ylim()[1]
    # ax33.set_yticks((ymax/2,ymax))
    # ax33.grid(True)
    #
    #\phi
    #

    # plt.text(250, 0.9*ymax, "i", size=10, rotation=0.,
    #          ha="center", va="center",
    #          bbox = dict(boxstyle="square",
    #                      ec=(1., 0.5, 0.5),
    #                      fc=(1., 0.8, 0.8),
    #                      )
    #          )
    ax35 = plt.subplot(gs3[-2,:])
    plotqtyvstime(qtymem,ax=ax35,whichplot=5,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,usegaussianunits=True)
    ymax=ax35.get_ylim()[1]
    if 1 < ymax and ymax < 2: 
        #ymax = 2
        tck=(1,)
        ax35.set_yticks(tck)
        #ax35.set_yticklabels(('','1','2'))
    elif ymax < 1: 
        ymax = 1
        tck=(0.5,1)
        ax35.set_yticks(tck)
        ax35.set_yticklabels(('','1'))
    else:
        ymax=np.floor(ymax)+1
        if ymax >= 60:
            tck=np.arange(1,ymax/30.)*30.
        elif ymax >= 10:
            tck=np.arange(1,ymax/5.)*5.
        else:
            tck=np.arange(1,ymax)
        ax35.set_yticks(tck)
    ax35.grid(True)
    if ymax >= 10:
        ax35.set_ylabel(r"$\phi_{\rm BH}$",size=16,ha='left',labelpad=25)
    plt.text(ax35.get_xlim()[1]/40., 0.8*ax35.get_ylim()[1], r"$(\mathrm{f})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    ax35r = ax35.twinx()
    ax35r.set_ylim(ax35.get_ylim())
    ax35r.set_yticks(tck)
    #
    #pjet/<mdot>
    #
    ax34 = plt.subplot(gs3[-1,:])
    plotqtyvstime(qtymem,ax=ax34,whichplot=4,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma)
    ax34.set_ylim((0,3.8*prefactor))
    ymax=ax34.get_ylim()[1]
    if prefactor < ymax and ymax < 2*prefactor: 
        #ymax = 2
        tck=(prefactor,)
        ax34.set_yticks(tck)
        #ax34.set_yticklabels(('','100','200'))
    elif ymax < prefactor: 
        ymax = prefactor
        tck=(0.5*prefactor,prefactor)
        ax34.set_yticks(tck)
        ax34.set_yticklabels(('','%d' % prefactor))
    else:
        ymax=np.floor(ymax/prefactor)+1
        ymax*=prefactor
        tck=np.arange(1,ymax/prefactor)*prefactor
        ax34.set_yticks(tck)
    #reset lower limit to 0
    ax34.set_ylim((0,ax34.get_ylim()[1]))
    ax34.grid(True)
    plt.text(ax34.get_xlim()[1]/40., 0.8*ax34.get_ylim()[1], r"$(\mathrm{g})$", size=16, rotation=0.,
             ha="center", va="center",
             color='k',weight='regular',bbox=bbox_props
             )
    ax34r = ax34.twinx()
    ax34r.set_ylim(ax34.get_ylim())
    ax34r.set_yticks(tck)
    #
    if domakeframes:
        #
        # Make Frames
        #
        dogrid = False
        downsample=4
        density=2
        dodiskfield=True
        minlenbhfield=0.2
        minlendiskfield=0.2
        #
        # PLOT 1
        #
        fname = "fieldline%04d.bin" % findexlist[0]
        rfd(fname)
        cvel() #for calculating bsq
        #xz
        gs1 = GridSpec(4, 4)
        gs1.update(left=0.04, right=0.94, top=0.995, bottom=0.48, wspace=0.05)
        #
        ax1 = plt.subplot(gs1[2:4, 0])
        # plt.text(-0.75*plen, 0.75*plen, r"$(\mathrm{a})$", size=16, rotation=0.,
        #          ha="center", va="center",
        #          color='k',weight='regular',bbox=bbox_props
        #          )
        mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=doslines,downsample=downsample,density=density,dodiskfield=False,arrowsize=arrowsize)
        ax1.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        if dogrid: plt.grid()
        #xy
        ax2 = plt.subplot(gs1[0:2, 0])
        mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=False)
        plt.setp( ax2.get_xticklabels(), visible=False)
        plt.text(-0.75*plen, 0.8*plen, r"$(\mathrm{a})$", size=16, rotation=0.,
                 ha="center", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        plt.text(0.9*plen, 0.8*plen, r"$t=%g$" % np.floor(t), size=16, rotation=0.,
                 ha="right", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        ax2.set_ylabel(r'$y\ [r_g]$',fontsize=16,ha='center')
        if dogrid: plt.grid()
        plt.subplots_adjust(hspace=0.03) #increase vertical spacing to avoid crowding
        #
        # PLOT 2
        #
        fname = "fieldline%04d.bin" % findexlist[1]
        rfd(fname)
        cvel() #for calculating bsq
        #Rz
        #gs1 = GridSpec(4, 4)
        #
        ax1 = plt.subplot(gs1[2:4, 1])
        # plt.text(-0.75*plen, 0.75*plen, r"$(\mathrm{c})$", size=16, rotation=0.,
        #          ha="center", va="center",
        #          color='k',weight='regular',bbox=bbox_props
        #          )
        mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,
                ax=ax1,cb=False,pt=False,dostreamlines=doslines,downsample=downsample,
                density=density,dodiskfield=dodiskfield,minlendiskfield=minlendiskfield,minlenbhfield=minlenbhfield,arrowsize=arrowsize)
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        if dogrid: plt.grid()
        #xy
        ax2 = plt.subplot(gs1[0:2, 1])
        plt.text(-0.75*plen, 0.8*plen, r"$(\mathrm{b})$", size=16, rotation=0.,
                 ha="center", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=False)
        plt.setp( ax2.get_xticklabels(), visible=False)
        plt.text(0.9*plen, 0.8*plen, r"$t=%g$" % np.floor(t), size=16, rotation=0.,
                 ha="right", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        if dogrid: plt.grid()
        plt.subplots_adjust(hspace=0.03) #increase vertical spacing to avoid crowding
        #
        # PLOT 3
        #
        fname = "fieldline%04d.bin" % findexlist[2]
        rfd(fname)
        cvel() #for calculating bsq
        #Rz
        ax1 = plt.subplot(gs1[2:4, 2])
        # plt.text(-0.75*plen, 0.75*plen, r"$(\mathrm{e})$", size=16, rotation=0.,
        #          ha="center", va="center",
        #          color='k',weight='regular',bbox=bbox_props
        #          )
        mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=doslines,downsample=downsample,density=density,dodiskfield=dodiskfield,minlendiskfield=minlendiskfield,minlenbhfield=minlenbhfield,arrowsize=arrowsize)
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        if dogrid: plt.grid()
        #xy
        ax2 = plt.subplot(gs1[0:2, 2])
        plt.text(-0.75*plen, 0.8*plen, r"$(\mathrm{c})$", size=16, rotation=0.,
                 ha="center", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        plt.text(0.9*plen, 0.8*plen, r"$t=%g$" % np.floor(t), size=16, rotation=0.,
                 ha="right", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=False)
        plt.setp( ax2.get_xticklabels(), visible=False)
        if dogrid: plt.grid()
        plt.subplots_adjust(hspace=0.03) #increase vertical spacing to avoid crowding
        #
        # PLOT 4
        #
        fname = "fieldline%04d.bin" % findexlist[3]
        rfd(fname)
        cvel() #for calculating bsq
        #Rz
        ax1 = plt.subplot(gs1[2:4, 3])
        # plt.text(-0.75*plen, 0.75*plen, r"$(\mathrm{g})$", size=16, rotation=0.,
        #          ha="center", va="center",
        #          color='k',weight='regular',bbox=bbox_props
        #          )
        mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=doslines,downsample=downsample,density=density,dodiskfield=dodiskfield,minlendiskfield=minlendiskfield,minlenbhfield=minlenbhfield,arrowsize=arrowsize)
        ax1.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        if dogrid: plt.grid()
        #xy
        ax2 = plt.subplot(gs1[0:2, 3])
        plt.text(-0.75*plen, 0.8*plen, r"$(\mathrm{d})$", size=16, rotation=0.,
                 ha="center", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        plt.text(0.9*plen, 0.8*plen, r"$t=%g$" % np.floor(t), size=16, rotation=0.,
                 ha="right", va="center",
                 color='w',weight='regular' #,bbox=bbox_props
                 )
        mkframexy("lrho%04d_xy%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=False)
        plt.setp( ax2.get_xticklabels(), visible=False)
        if dogrid: plt.grid()
        #
        plt.subplots_adjust(hspace=0.03) #increase vertical spacing to avoid crowding
        #
        #(left=0.02, right=0.94, top=0.99, bottom=0.45, wspace=0.05)
        ax1 = fig.add_axes([0.94, 0.48, 0.02, 0.515])
        #
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=-6, vmax=0.5625)
        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                           norm=norm,
                                           orientation='vertical')
    #end if domakeframes
    #
    plt.savefig( "fig1.png" )
    plt.savefig( "fig1.eps" )
    #
    print( "Done!" )
    sys.stdout.flush()


def mkmdot(doreload=1,epsFm=None,epsFke=None,epsetaj=None,epsFm30=None,fti=None,ftf=None,prefactor=100,sigma=None,usegaussianunits=False,arrowsize=1,gs3=None,dotwinx=True,doylab=True,lab=None,title=None,plotFM30=False):
    global qtymem
    findexlist=None
    #Figure 1
    #To make plot, run 
    #run ~/py/mread/__init__.py 1 1
    #To re-make plot without reloading the fiels, run
    #run ~/py/mread/__init__.py 1 -1
    doslines=True
    plotlenf=10
    plotleni=25
    plen=plotleni
    plotlenti=40000
    plotlentf=45000
    bbox_props = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.9)
    #AT: plt.legend( loc = 'upper left', bbox_to_anchor = (0.5, 0.5) ) #0.5, 0.5 = center of plot
    #To generate movies for all sub-folders of a folder:
    #cd ~/Research/runart; for f in *; do cd ~/Research/runart/$f; (python  ~/py/mread/__init__.py &> python.out &); done
    dontloadfiles = False
    grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]), use2d=True )
    #rd( "dump0000.bin" )
    #rfd("fieldline0000.bin")  #to definea
    #grid3dlight("gdump")
    if doreload == 1:
        qtymem=getqtyvstime(None,0.2)
    #make accretion rate plot, etc.
    sys.stdout.flush()
    plotlen = plotleni+(plotlenf-plotleni)*(t-plotlenti)/(plotlentf-plotlenti)
    plotlen = min(plotlen,plotleni)
    plotlen = max(plotlen,plotlenf)
    if gs3 is None:
        gs3 = GridSpec(3, 3)
        gs3.update(left=0.055, right=0.97, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
    if lab is None:
        lab = ["a", "b", "c"]
    #mdot
    ax31 = plt.subplot(gs3[-3,:])
    if title is not None:
        plt.title(title)
    plotqtyvstime(qtymem,ax=ax31,whichplot=1,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,plotFM30=plotFM30) #AT: need to specify index!
    ymax=ax31.get_ylim()[1]
    ymax=2*(np.floor(np.floor(ymax+1.5)/2))
    #OVERRIDE
    ymax=40
    ax31.set_yticks((ymax/2,ymax))
    #ax31.set_xlabel(r"$t\ [r_g/c]")
    if doylab == False:
        ax31.set_ylabel("")
    ax31.grid(True)
    placeletter(ax31,"$(\mathrm{%s})$" % lab[0],bbox=bbox_props)
    if dotwinx:
        ax31r = ax31.twinx()
        ax31r.set_ylim(ax31.get_ylim())
        ax31r.set_yticks((ymax/2,ymax))
    #pjet
    # ax32 = plt.subplot(gs3[-2,:])
    # plotqtyvstime(qtymem,ax=ax32,whichplot=2)
    # ymax=ax32.get_ylim()[1]
    # ax32.set_yticks((ymax/2,ymax))
    # ax32.grid(True)
    #pjet/mdot
    # ax33 = plt.subplot(gs3[-1,:])
    # plotqtyvstime(qtymem,ax=ax33,whichplot=3)
    # ymax=ax33.get_ylim()[1]
    # ax33.set_yticks((ymax/2,ymax))
    # ax33.grid(True)
    #
    #\phi
    #

    # plt.text(250, 0.9*ymax, "i", size=10, rotation=0.,
    #          ha="center", va="center",
    #          bbox = dict(boxstyle="square",
    #                      ec=(1., 0.5, 0.5),
    #                      fc=(1., 0.8, 0.8),
    #                      )
    #          )
    ax35 = plt.subplot(gs3[-2,:])
    plotqtyvstime(qtymem,ax=ax35,whichplot=5,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma,usegaussianunits=True)
    #OVERRIDE
    ax35.set_ylim((0,70))
    ymax=ax35.get_ylim()[1]
    if 1 < ymax and ymax < 2: 
        #ymax = 2
        tck=(1,)
        ax35.set_yticks(tck)
        #ax35.set_yticklabels(('','1','2'))
    elif ymax < 1: 
        ymax = 1
        tck=(0.5,1)
        ax35.set_yticks(tck)
        ax35.set_yticklabels(('','1'))
    else:
        ymax=np.floor(ymax)+1
        if ymax >= 60:
            tck=np.arange(1,ymax/30.)*30.
        elif ymax >= 20:
            tck=np.arange(1,ymax/10.)*10.
        elif ymax >= 10:
            tck=np.arange(1,ymax/5.)*5.
        else:
            tck=np.arange(1,ymax)
        ax35.set_yticks(tck)
    ax35.grid(True)
    if ymax >= 10:
        ax35.set_ylabel(r"$\phi_{\rm BH}$",size=16,ha='left',labelpad=25)
    if doylab == False:
        ax35.set_ylabel("")
    placeletter(ax35,"$(\mathrm{%s})$" % lab[1],bbox=bbox_props)
    if dotwinx:
        ax35r = ax35.twinx()
        ax35r.set_ylim(ax35.get_ylim())
        ax35r.set_yticks(tck)
    #
    #pjet/<mdot>
    #
    ax34 = plt.subplot(gs3[-1,:])
    plotqtyvstime(qtymem,ax=ax34,whichplot=4,findex=findexlist,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=prefactor,sigma=sigma)
    #OVERRIDE
    ax34.set_ylim((-.5*prefactor,1.99*prefactor))
    ymax=ax34.get_ylim()[1]
    ymin=ax34.get_ylim()[0]
    if prefactor < ymax and ymax < 2*prefactor: 
        #ymax = 2
        if ymin < 0:
            tck=(0,prefactor)
        else:
            tck=(prefactor,)
        ax34.set_yticks(tck)
        #ax34.set_yticklabels(('','100','200'))
    elif ymax < prefactor: 
        ymax = prefactor
        tck=(0.5*prefactor,prefactor)
        ax34.set_yticks(tck)
        ax34.set_yticklabels(('','%d' % prefactor))
    else:
        ymax=np.floor(ymax/prefactor)+1
        ymax*=prefactor
        tck=np.arange(1,ymax/prefactor)*prefactor
        ax34.set_yticks(tck)
    #reset lower limit to 0
    #ax34.set_ylim((0,ax34.get_ylim()[1]))
    ax34.grid(True)
    placeletter(ax34,"$(\mathrm{%s})$" % lab[2],bbox=bbox_props)
    if doylab == False:
        ax34.set_ylabel("")
    if dotwinx:
        ax34r = ax34.twinx()
        ax34r.set_ylim(ax34.get_ylim())
        ax34r.set_yticks(tck)

def provsretro(dotakeoutfloors=False,doreload=True):
    global reslist
    grid3d("gdump.bin",use2d=True)
    #rfd("fieldline0000.bin")
    #flist = ["avg2d20_0000_0001.npy", "avg2d20_0000_0050.npy","avg2d20_0100_0150.npy","avg2d20_0150_0200.npy","avg2d20_0200_0250.npy"]
    #flist = ["avg2d20_00.npy", "avg2d20_0080_0100.npy", "avg2d20_0100_0120.npy", "avg2d20_0120_0140.npy", "avg2d20_0140_0156.npy","avg2d20_0080_0157.npy","avg2d20_0080_0157_nf.npy"]
    #flist = ["avg2d20_0080_0157.npy","avg2d20_0080_0157_nf.npy"]
    flist = ["avg2d.npy"
             #"avg2dnf.npy"
             ]
    #flist = ["avg0.npy", "avg2.npy"]
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()
    plt.figure(4)
    plt.clf()
    plt.figure(5)
    plt.clf()
    plt.figure(6)
    plt.clf()
    plt.figure(7)
    plt.clf()
    firsttime=True
    # if doreload:
    #     res = takeoutfloors(doreload=True,isinteractive=0,writefile=False,dotakeoutfloors=dotakeoutfloors)
    #     reslist=res
    #a_eta,a_Fm,a_Fe,a_Fl = res
    for (i,f) in enumerate(flist):
        print "%s\n" % f
        avgmem = get2davg(fname=f)
        assignavg2dvars(avgmem)
        lab = "(%g,%g)" % (avg_ts[0],avg_te1[0])
        #plot2davg(whichplot=2)
        #plot Mdot vs. r for region v^r < 0
        cond = (avg_rhouu[1]<0)
        mdot_den = gdet[:,:,0:1]*avg_rhouu[1]
        mdin = intangle( mdot_den*nz, which=cond )
        mdall = intangle( mdot_den*nz )
        #xxx
        #######################
        #
        #  FIGURE 1: Mdot
        #
        #######################
        plt.figure(1)
        if firsttime:
            ax1 = plt.gca()
        plt.plot( r[:,0,0], -mdin, ':' )
        ax1.plot( r[:,0,0], -mdall, '-', label=lab )
        ax1.set_xscale('log')
        ax1.set_xlim(rhor,100)
        ax1.set_ylim(0,20)
        ax1.set_xlabel(r"$r$",fontsize=16)
        ax1.set_ylabel(r"$\dot M$",fontsize=16)
        ax1.grid(b=True)
        #######################
        #
        #  FIGURE 2: u^r
        #
        #######################
        plt.figure(2)
        #plt.clf()
        if firsttime:
            ax2 = plt.gca()
        # #avg over all volume
        # up = (gdet[:,:,0:1]*avg_rhouu[1]*_dx2*_dx3).sum(-1).sum(-1)
        # dn = (gdet[:,:,0:1]*avg_rhouu[0]*_dx2*_dx3).sum(-1).sum(-1)/dxdxp[1,1,:,0,0]
        #avg over midplane
        up = (gdet[:,ny/2,0]*avg_rhouu[1][:,ny/2,0]*_dx2*_dx3)
        dn = (gdet[:,ny/2,0]*avg_rho[:,ny/2,0]*_dx2*_dx3)/dxdxp[1,1,:,0,0]
        uur1d = np.array(up/dn)
        uurmid = 0.5*(avg_uu[1,:,ny/2,0]+avg_uu[1,:,ny/2-1,0])*dxdxp[1,1,:,0,0]
        uutmid = 0.5*(avg_uu[0,:,ny/2,0]+avg_uu[0,:,ny/2-1,0])
        #vurmid = uurmid/uutmid
        #ax2.plot(r[:,0,0], -ur1d)
        #ax2.plot(r[:,0,0],-vur1d,'b:')
        #ax2.plot(r[:,0,0],-uurmid,'b--')
        ax2.plot(r[:,0,0],-uur1d,'m-', label=(r"$\langle\rho u^r\rangle/\langle\rho\rangle$ at " + lab))
        ax2.plot(r[:,0,0],-uurmid, label=(r"$\langle u^r\rangle$ at " + lab))
        #print ur1d.shape
        #ax2.plot(r[:,0,0],0.1*(r[:,0,0]/10)**(-1.2))
        ax2.set_ylim(1e-4,1.5)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(rhor,100)
        #ax2.ylim(0,20)
        ax2.set_xlabel(r"$r$",fontsize=20)
        ax2.set_ylabel(r"$-u^r$",fontsize=20)
        ax2.grid(b=True)
        #######################
        #
        #  FIGURE 3: \Sigma
        #
        #######################
        plt.figure(3)
        #plt.clf()
        ax = plt.gca()
        sigval = (gdet[:,:,0:1]*avg_rhouu[0]*_dx2*_dx3*nz).sum(-1).sum(-1)/dxdxp[1,1,:,0,0]*scaletofullwedge(1.)/(2*np.pi*r[:,ny/2,0])
        #sigvalfm = a_Fm / (-4*np.pi*r[:,ny/2,0]*avg_uu[1,:,ny/2,0]*dxdxp[1,1,:,0,0]/avg_uu[0,:,ny/2,0])
        plt.plot( r[:,0,0], sigval, 'k', label=lab )
        #plt.plot( r[:,0,0], sigvalfm, 'r', label=lab )
        #sigval=sigvalfm
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(rhor,100000)
        plt.xlabel(r"$r$",fontsize=16)
        plt.ylabel(r"$\Sigma$",fontsize=16)
        plt.grid(b=True)
        #######################
        #
        #  FIGURE 4: Ang. momentum
        #
        #######################
        plt.figure(4)
        rad=r[:,0,0]
        #From ST eq. (12.7.18)
        udphianal = lk( a, rad )
        udtanal = -ek( a, rad )
        risco=Risco(a)
        udphianal[rad<risco]=lk(a,risco)
        udtanal[rad<risco]=-ek(a,risco)
        ax = plt.gca()
        jin_nmr = (avg_rhouu+avg_uguu+avg_bsquu/2.)[1]*avg_ud[3]/dxdxp[3,3,0,0,0]
        jin_dnm = (avg_rhouu)[1]
        jinmid = (jin_nmr/jin_dnm)[:,ny/2,0]
        jtot_nmr = avg_Tud[1,3]/dxdxp[3,3,0,0,0]
        jtot_dnm = (avg_rhouu)[1]
        jtotmid = (jtot_nmr/jtot_dnm)[:,ny/2,0]
        jinfull = (gdet*jin_nmr).sum(1)/(gdet*jin_dnm).sum(1)
        isdisk=np.abs(h[:,:,0:1]-np.pi/2)<0.1
        jtotfull = (gdet[:,:,0:1]*jtot_nmr*isdisk).sum(-1).sum(-1)/(gdet[:,:,0:1]*jtot_dnm*isdisk).sum(-1).sum(-1)
        jinfull = (gdet[:,:,0:1]*jin_nmr*isdisk).sum(-1).sum(-1)/(gdet[:,:,0:1]*jin_dnm*isdisk).sum(-1).sum(-1)
        # plt.figure(6)
        # plt.plot( r[:,0,0], (gdet[:,:,0:1]*jtot_nmr*isdisk).sum(-1).sum(-1))
        # plt.figure(4)
        plt.plot( r[:,0,0], jinmid, label=(r"$l_{\rm in}$ for " + lab) )
        plt.plot( r[:,0,0], jtotmid, label=(r"$l_{\rm tot}$ for " + lab) )
        # plt.plot( r[:,0,0], jtotfull, label=(r"$l_{\rm tot,full}$ for " + lab) )
        plt.plot( r[:,0,0], avg_ud[3,:,ny/2,0]/dxdxp[3,3,:,0,0], label=(r"$u_\phi$ for " + lab) )
        plt.plot( r[:,0,0], udphianal, label=(r"$l_{\rm SS}$") )
        ax.set_xscale('log')
        #ax.set_yscale('log')plt.ylim(-40,0)
        plt.xlim(rhor,100)
        plt.xlabel(r"$r$",fontsize=20)
        plt.ylabel(r"$l$",fontsize=20)
        plt.grid(b=True)
        plt.ylim(-1,10)
        plt.figure(10)
        plt.clf()
        plt.plot( r[:,0,0], 1+(avg_ud)[0,:,ny/2,0], 'g',label=(r"$1-u_t$ for " + lab) )
        plt.plot( r[:,0,0], 1+udtanal, 'b', label=(r"$1-E_{SS}$") )
        plt.plot( r[iofr(risco),0,0],1+avg_ud[0,iofr(risco),ny/2,0],'og')
        plt.plot( r[iofr(risco),0,0],1+udtanal[iofr(risco)],'ob')
        plt.legend()
        ax.set_xscale('log')
        #ax.set_yscale('log')plt.ylim(-40,0)
        plt.xlim(rhor,10)
        plt.xlabel(r"$r$",fontsize=20)
        plt.ylabel(r"$e$",fontsize=20)
        plt.grid(b=True)
        plt.legend(loc='upper right')
        plt.ylim(0,0.6)

        #######################
        #
        #  FIGURE 5: Radial force balance
        #
        #######################
        plt.figure(5)
        if firsttime:
            plt.clf()
        #x1
        if True:
            pressureMA=(gam-1)*avg_ug
            inertiaMA=(avg_rho+gam*avg_ug)*avg_uu[1]*avg_ud[1]
            pressureEM=avg_bsq/2.-avg_bu[1]*avg_bd[1]
            inertiaEM=(avg_bsq)*avg_uu[1]*avg_ud[1]
            fpEM=-dfdx1(pressureEM)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fiEM=-dfdx1(inertiaEM)[:,ny/2,0]
            fpMA=-dfdx1(pressureMA)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fiMA=-dfdx1(inertiaMA)[:,ny/2,0]
            #x2
            pressure2MA=(gam-1)*avg_ug*0
            inertia2MA=(avg_rho+gam*avg_ug)*avg_uu[2]*avg_ud[1]
            pressure2EM=avg_bsq*0/2.-avg_bu[2]*avg_bd[1]
            inertia2EM=(avg_bsq)*avg_uu[2]*avg_ud[1]
            fp2EM=-dfdx2(pressure2EM)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fi2EM=-dfdx2(inertia2EM)[:,ny/2,0]
            fp2MA=-dfdx2(pressure2MA)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fi2MA=-dfdx2(inertia2MA)[:,ny/2,0]
        else:
            pressureMA=(gam-1)*avg_ug
            inertiaMA=(avg_rhouuud+gam*avg_uguuud)[1,1]
            pressureEM=0.5*avg_bsq-avg_bubd[1,1]
            inertiaEM=avg_bsquuud[1,1]
            fpEM=-dfdx1(pressureEM)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fiEM=-dfdx1(inertiaEM)[:,ny/2,0]
            fpMA=-dfdx1(pressureMA)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fiMA=-dfdx1(inertiaMA)[:,ny/2,0]
            #x2
            pressure2MA=(gam-1)*avg_ug*0
            inertia2MA=(avg_rhouuud+gam*avg_uguuud)[2,1]
            pressure2EM=avg_bsq*0/2.-avg_bubd[2,1]
            inertia2EM=avg_bsquuud[2,1]
            fp2EM=-dfdx2(pressure2EM)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fi2EM=-dfdx2(inertia2EM)[:,ny/2,0]
            fp2MA=-dfdx2(pressure2MA)[:,ny/2,0] #/dxdxp[1,1,0,0,0]
            fi2MA=-dfdx2(inertia2MA)[:,ny/2,0]
        #fg=-(avg_rho/r**2)[:,ny/2,0]
        connddd=mdot(gv3,conn)
        avg_Tuu=mdot(avg_Tud,gn3)
        avg_TuuMA=mdot(avg_TudMA,gn3)
        avg_TuuEM=mdot(avg_TudEM,gn3)
        fgMA=(avg_TuuMA[0,0]*connddd[0,1,0]+avg_TuuMA[0,1]*connddd[0,1,1]+avg_TuuMA[1,1]*connddd[1,1,1])[:,ny/2,0]
        fcMA=(avg_TuuMA[0,3]*connddd[0,1,3]+avg_TuuMA[2,2]*connddd[2,1,2]+avg_TuuMA[3,0]*connddd[3,1,0]+avg_TuuMA[3,1]*connddd[3,1,1]+avg_TuuMA[3,3]*connddd[3,1,3])[:,ny/2,0]
        fgEM=(avg_TuuEM[0,0]*connddd[0,1,0]+avg_TuuEM[0,1]*connddd[0,1,1]+avg_TuuEM[1,1]*connddd[1,1,1])[:,ny/2,0]
        fcEM=(avg_TuuEM[0,3]*connddd[0,1,3]+avg_TuuEM[2,2]*connddd[2,1,2]+avg_TuuEM[3,0]*connddd[3,1,0]
              +avg_TuuEM[3,1]*connddd[3,1,1]+(avg_TuuEM[3,3]*connddd[3,1,3]))[:,ny/2,0]
        norm=avg_rho[:,ny/2,0]
        plt.plot(rad,-fgMA/norm,'--',lw=2,label=r"$-F_{g,MA}$")
        plt.plot(rad,fcMA/norm,lw=2,label=r"$F_{c,MA}$")
        plt.plot(rad,-fgEM/norm,'--',lw=2,label=r"$-F_{g,EM}$",color='brown')
        plt.plot(rad,fcEM/norm,"-",lw=2,label=r"$F_{c,EM}$",color='orange')
        plt.plot(rad,(fiMA)/norm,lw=2,label=r"$F_{i,MA}$")
        plt.plot(rad,(-fiMA)/norm,'--',lw=2,label=r"$-F_{i,MA}$",color='b')
        plt.plot(rad,(fpMA)/norm,lw=2,label=r"$F_{p,MA}$")
        plt.plot(rad,(-fpMA)/norm,'--',lw=2,label=r"$-F_{p,MA}$",color='r')
        plt.plot(rad,(-fiEM)/norm,'--',lw=2,label=r"$-F_{i,EM}$")
        plt.plot(rad,(fpEM)/norm,lw=2,label=r"$F_{p,EM}$")
        plt.plot(rad,(-fpEM)/norm,'--',lw=2,label=r"$-F_{p,EM}$",color='m')
        #x2
        plt.plot(rad,(fi2MA)/norm,':',lw=2,label=r"$F_{i2,MA}$")
        plt.plot(rad,(fp2MA)/norm,':',lw=2,label=r"$F_{p2,MA}$")
        plt.plot(rad,(fi2EM)/norm,':',lw=2,label=r"$F_{i2,EM}$")
        plt.plot(rad,(fp2EM)/norm,':',lw=2,label=r"$F_{p2,EM}$")
        plt.plot(rad,3.5*(rad/rhor)**(-7./2.),'k-',lw=2)
        plt.plot(rad,(avg_bsq[:,ny/2,0]/(2*sigval)*r[:,ny/2,0]**3/1e2),'k-',lw=1,label=r"$r^3b^2/(2\Sigma)$")
        plt.plot(rad,(avg_bsq[:,ny/2,0]/(2*sigval)*r[:,ny/2,0]**2/1e2),'k--',lw=1,label=r"$r^2b^2/(2\Sigma)$")
        #plt.plot(rad,(avg_bsq*avg_rho)[:,ny/2,0]/sigval**2/1000.,'k-',lw=1,label=r"$b^2/rho$")
        plt.xlim(rhor,100)
        plt.ylim(ymin=0.5e-3,ymax=60)
        plt.xscale('log')
        plt.yscale('log')
        #######################
        #
        #  FIGURE 6: Flux
        #
        #######################
        plt.figure(6)
        #plt.clf()
        ax = plt.gca()
        aphi=fieldcalc(gdetB1=avg_gdetB[0])
        fluxval = aphi.max(axis=1)[:,0]
        if firsttime:
            fluxnorm = fluxval[iofr(rhor)]
        fluxval = fluxval/fluxnorm
        plt.plot( r[:,0,0], fluxval, label=lab )
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        plt.xlim(rhor,20)
        plt.ylim(1,4)
        plt.xlabel(r"$r$",fontsize=16)
        plt.ylabel(r"$\Phi$",fontsize=16)
        plt.grid(b=True)
        plt.legend()
        #######################
        #
        #  FIGURE 7: bsq/rho
        #
        #######################
        plt.figure(7)
        #plt.clf()
        ax = plt.gca()
        plt.plot( r[:,0,0], (avg_bsq/avg_rho)[:,ny/2,0], label=r"$b^2/\rho$ " + lab )
        plt.plot( r[:,0,0], (avg_bsq/avg_ug)[:,ny/2,0], label=r"$b^2/u_g$ " + lab )
        plt.plot( r[:,0,0], (0.5*avg_bsq/avg_ug/(gam-1))[:,ny/2,0], label=r"$p_m/p_g$ " + lab )
        plt.plot( r[:,0,0], (avg_ug/avg_rho)[:,ny/2,0], label=r"$u_g/\rho$ " + lab )
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(rhor,1000)
        plt.ylim(1e-5,4)
        plt.xlabel(r"$r$",fontsize=16)
        plt.ylabel(r"$b^2/rho$",fontsize=16)
        plt.grid(b=True)
        plt.legend()
        firsttime=False
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels,loc="upper left")
    plt.figure(2)
    plt.title(r"Radial velocity for $a=%g$" % a,fontsize=20)
    #free-fall radial 4-velocity
    plt.plot(r[:,0,0],(r[:,0,0]/1)**(-1.),'c--',label=r"$1/r$")
    # plt.plot(r[:,0,0],0.01*(r[:,0,0]/10)**(-2),'b:')
    # plt.plot(r[:,0,0],0.01*(r[:,0,0]/10)**(-1./2.),'b:')
    uurKSzamo = -gn3[0,1,:,ny/2,0]/(-gn3[0,0,:,ny/2,0])**0.5*dxdxp[1,1,:,0,0]
    vurKSzamo = gn3[0,1,:,ny/2,0]/gn3[0,0,:,ny/2,0]*dxdxp[1,1,:,0,0]
    vff_nonrel = (2/r[:,0,0])**0.5
    uurff = 2**0.5 * (a**2+r[:,0,0]**2)**0.5 / r[:,0,0]**1.5
    plt.plot(r[:,0,0],uurff,'r--', label=r"$u^r_{\rm ff}=[2(a^2+r^2)/r^3]^{1/2}$, $\mathrm{free-fall\ v}$")
    #plt.plot(r[:,0,0],vff_nonrel,'y-')
    #plt.plot(r[:,0,0],0.15*vff_nonrel,'y:')
    #plot instantaneous v^r
    if False:
        rfd("fieldline0000.bin")
        plt.plot(r[:,0,0],-uu[1,:,ny/2,0]*dxdxp[1,1,:,0,0],'r',lw=2)
    #plt.plot(r[:,0,0],-uurfreefall,'g-')
    #plt.plot(r[:,0,0],-uurfreefall,'g--')
    #plt.plot(r[:,0,0],-0.35*uurfreefall,'g--')
    plt.plot(r[:,0,0],-uurKSzamo,'g-', label=r"$u^r_{\rm ZAMO,KS}=2/[r(r+2)]^{1/2}$")
    plt.plot(r[:,0,0],-0.3*uurKSzamo,'g--', label=r"$0.3 u^r_{\rm ZAMO,KS}$")
    plt.legend(loc="lower left")
    plt.savefig("velocity%g.pdf" % a )
    plt.savefig("velocity%g.eps" % a )
    plt.figure(3)
    plt.legend(loc="lower center")
    plt.plot(r[:,0,0],1e2*(r[:,0,0]/100)**(1))
    plt.plot(r[:,0,0],1e2*(r[:,0,0]/100)**(2))
    plt.ylim(ymax=1e3)
    plt.figure(4)
    plt.title(r"Midplane angular momentum for $a=%g$" % a,fontsize=20)
    plt.legend(loc="upper left")
    plt.savefig("angmom%g.pdf" % a)
    plt.savefig("angmom%g.eps" % a)
    plt.figure(5)
    plt.legend(loc="upper right",ncol=3)
    plt.grid(b=True)
        

def dfdx1(f,dn=4):
    """returns gdet**(-1)*(gdet*f),x1"""
    #initialize dummy way
    dgf=f/_dx1
    gf=gdet*f
    dgf[dn:-dn]=(gf[2*dn:]-gf[:-2*dn])/(2.*dn*_dx1*gdet[dn:-dn])
    return(dgf)

def dfdx2(f,dn=4):
    """returns gdet**(-1)*(gdet*f),x2"""
    #initialize dummy way
    dgf=f/_dx2
    gf=gdet*f
    dgf[:,dn:-dn]=(gf[:,2*dn:]-gf[:,:-2*dn])/(2.*dn*_dx2*gdet[:,dn:-dn])
    return(dgf)


def generate_time_series(docompute=False):
        #cd ~/run; for f in rtf*; do cd ~/run/$f; (nice -n 10 python  ~/py/mread/__init__.py &> python.out); done
        grid3d("gdump.bin",use2d=True)
        #rd("dump0000.bin")
        #rfd("fieldline0000.bin")
        rhor=1+(1-a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        #diskflux=diskfluxcalc(ny/2)
        #qtymem=None #clear to free mem
        if len(sys.argv[1:])==2 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
            whichi = int(sys.argv[1])
            whichn = int(sys.argv[2])
            if whichi >= whichn:
                mergeqtyvstime(whichn)
            else:
                qtymem=getqtyvstime(ihor,0.2,whichi=whichi,whichn=whichn,docompute=docompute)
        else:
            qtymem=getqtyvstime(ihor,0.2,docompute=docompute)
            plotqtyvstime(qtymem)

def oldstuff():
    if False:
        #cd into the directory that contains the dumps/ directory
        #read in the grid file
        grid3d("gdump.bin")
        #plot log10(radius) vs. grid index, ti
        #plt.plot( ti[:,0,0], np.log10(r[:,0,0]) )
        plt.figure( 1, figsize=(6,12) )
        plt.clf()
        small=1e-5
        #theta grid lines through cell centers
        levs = np.linspace(0,nx-1-small,nx)
        plc(ti,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=levs,colors='#0fffff')
        levs = np.linspace(0,ny-1-small,ny)
        plc(tj,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=levs,colors='#0fffff')
        levs = np.linspace(0,nx-small,nx+1)
        plc(tif,xcoord=rf*np.sin(hf),ycoord=rf*np.cos(hf),levels=levs,colors='k')
        #radial grid lines through cell centers
        levs = np.linspace(0,ny-small,ny+1)
        plc(tjf,xcoord=rf*np.sin(hf),ycoord=rf*np.cos(hf),levels=levs,colors='k')
        plt.xlim(0,2)
        plt.ylim(-2,2)
        plt.savefig("grid.png")
        #
        #Read in the dump file and plot it
        #
        #read in the 1000th dump file
        plt.figure( 2, figsize=(6,12) )
        plt.clf()
        rfd("fieldline1000.bin")
        #plot contours of density
        plc(lrho,xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True,nc=50)
        plt.xlim(0,2)
        plt.ylim(-2,2)
        plt.savefig("logdensity.png")
    if False:
        grid3d("gdump.bin"); rfd("fieldline0000.bin")
        aphi = fieldcalcface()
        sig=intangle(gdet*rho)
        plt.clf()
        plt.plot(r[:,0,0],-gdetB[2,:,ny/2,0]/dxdxp[1,1,:,ny/2,0]/dxdxp[3,3,0,0,0]/sig) 
        plt.xlim(0,200)
        plt.ylim(0,2)
    if False:
        #grid3d("gdump")
        #rfd("fieldline0250.bin")
        #cvel()
        #plc(rho)
        grid3d("gdump.bin")
        rfd("fieldline0000.bin")
        diskflux=diskfluxcalc(ny/2)
        ts,fs,md=fhorvstime(11)
        plotit(ts,fs/(diskflux),md)
    if False:
        #cd ~/run; for f in rtf*; do cd ~/run/$f; (nice -n 10 python  ~/py/mread/__init__.py &> python.out); done
        grid3d("gdump.bin")
        rfd("fieldline0000.bin")
        diskflux=diskfluxcalc(ny/2)
        ts,fs,md,jem,jtot=mfjhorvstime(11)
        plotj(ts,fs/(diskflux),md,jem,jtot)
    if False:
        rfd("fieldline2344.bin")
        cvel()
        Tcalcud()
        xxx=-Tud[1,0]/(rho*uu[1])
        yyy=choplo(chophi(xxx.sum(2)/nz,50),-50)[:,:,None]
        plco(yyy,cb=True,nc=20)
        aphi=fieldcalcface()
        plc(aphi,nc=30)

    if False:
        #OLD FORMAT
        #Plot qtys vs. time
        #cd ~/run; for f in rtf*; do cd ~/run/$f; (nice -n 10 python  ~/py/mread/__init__.py &> python.out); done
        grid3d("gdump.bin")
        rfd("fieldline0000.bin")
        rhor=1+(1-a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        #diskflux=diskfluxcalc(ny/2)
        qtymem=None #clear to free mem, doesn't seem to work
        qtymem=getqtyvstime(ihor,0.2,fmtver=1)
        plotqtyvstime(qtymem)
    if False:
        rfd("fieldline0320.bin")
        plt.figure(1)
        aphi=fieldcalc()
        plc(aphi)
        plt.figure(2)
        aphi2=fieldcalc2()
        plc(aphi2)
        test()
    if False:
        grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]) )
        flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline*.bin") ))
        for findex, fname in enumerate(flist):
            print( "Reading " + fname + " ..." )
            rfd("../"+fname)
            plt.clf()
            mkframe("lrho%04d" % findex, vmin=-8,vmax=0.2)
        print( "Done!" )
    if False:
        grid3d("gdump.bin")
        rfd("fieldline0000.bin")
        rhor=1+(1+a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        hf=horfluxcalc(ihor)
        df=diskfluxcalc(ny/2)
        print "Initial (t=%-8g): BHflux = %g, Diskflux = %g" % (t, hf, df)
        rfd("fieldline1308.bin")
        hf=horfluxcalc(ihor)
        df=diskfluxcalc(ny/2,rmin=rhor)
        print "Final   (t=%-8g): BHflux = %g, Diskflux = %g" % (t, hf, df)
    if False:
        if len(sys.argv[1:])==2 and sys.argv[1].isdigit() and (sys.argv[2].isdigit() or sys.argv[2][0]=="-") :
            whichi = int(sys.argv[1])
            whichn = int(sys.argv[2])
            sys.stdout.flush()
        if whichn < 0 and whichn is not None:
            whichn = -whichn
            dontloadfiles = True
        else:
            dontloadfiles = False
            grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]), use2d=True )
            rd( "dump0000.bin" )
            qtymem=None #clear to free mem
            rhor=1+(1+a**2)**0.5
            ihor = np.floor(iofr(rhor)+0.5);
            qtymem=getqtyvstime(ihor,0.2)
            flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline*.bin") ) )
        fname=flist[whichi]
        print( "Processing " + fname + " ..." )
        sys.stdout.flush()
        rfd("../"+fname)
        cvel() #for calculating bsq
        plt.figure(0, figsize=(12,9), dpi=100)
        plt.clf()
        mkframe("lrho%04d_Rz%g" % (findex,plotlen), vmin=-6.,vmax=0.5625,len=plotlen,ax=ax1,cb=False,pt=False)
    if False:
        len=10
        #To generate movies for all sub-folders of a folder:
        #cd ~/Research/runart; for f in *; do cd ~/Research/runart/$f; (python  ~/py/mread/__init__.py &> python.out &); done
        grid3d( os.path.basename(glob.glob(os.path.join("dumps/", "gdump*"))[0]) )
        #rfd("fieldline0000.bin")  #to define _dx#
        #grid3dlight("gdump")
        flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline0000.bin") ) )
        for findex, fname in enumerate(flist):
            if os.path.isfile("lrho%04d_xy%g.png" % (findex,len)):
                print( "Skipping " + fname + " as lrho%04d_xy%g.png exists" % (findex,len) );
            else:
                print( "Processing " + fname + " ..." )
                rfd("../"+fname)
                plt.clf()
                mkframexy("lrho%04d_xy%g" % (findex,len), vmin=-8,vmax=1.5,len=len)
        print( "Done!" )
        #print( "Now you can make a movie by running:" )
        #print( "ffmpeg -fflags +genpts -r 10 -i lrho%04d.png -vcodec mpeg4 -qmax 5 mov.avi" )
        os.system("mv mov_%s_xy%g.avi mov_%s_xy%g.bak.avi" % ( os.path.basename(os.getcwd()), len, os.path.basename(os.getcwd()), len) )
        os.system("ffmpeg -fflags +genpts -r 10 -i lrho%%04d_xy%g.png -vcodec mpeg4 -qmax 5 mov_%s_xy%g.avi" % (len, os.path.basename(os.getcwd()), len) )
        #os.system("scp mov.avi 128.112.70.76:Research/movies/mov_`basename \`pwd\``.avi")

    #plt.clf(); rfd("fieldline0000.bin"); aphi=fieldcalc(); plc(ug/bsq) 
    #rfd("fieldline0002.bin")
    if False:
        grid3d( "gdump.bin" )
        rfd("fieldline0000.bin")
        plt.clf();
        mkframe("lrho%04d" % 0, vmin=-8,vmax=0.2)
    if False:
        grid3d("gdump"); rfd("fieldline0000.bin"); rrdump("rdump--0000"); plt.clf(); cvel(); plc(bsq,cb=True)
        plt.clf();plt.plot(x1[:,ny/2,0],(bsq/(2*(gam-1)*ug))[:,ny/2,0])
        plt.plot(x1[:,ny/2,0],(bsq/(2*(gam-1)*ug))[:,ny/2,0],'+')
        plt.plot(x1[:,ny/2,0],(0.01*rho)[:,ny/2,0])
    if False:
        plt.clf();plco(lrho,r*np.sin(h),r*np.cos(h),cb=True,levels=np.arange(-12,0,0.5)); plt.xlim(0,40); plt.ylim(-20,20)
    if False:
        rd( os.path.basename(glob.glob(os.path.join("dumps/", "dump0000*"))[0]) )
        #rrdump("rdump--0000")
        aphi = fieldcalc()
        plt.clf(); plt.plot(x1[:,ny/2,0],aphi[:,ny/2,0])
    if False:
        gen_vpot()
    if False:
        #Generates 6 co-aligned half-loops which combine into one big loop.
        #However, the field is noisy and probably the total flux in the big
        #loop is smaller than max possible given max ibeta.
        npow=4
        ap=np.zeros((6,rho.shape[0],rho.shape[1],rho.shape[2]))
        ap1=np.zeros((6,rho.shape[0],rho.shape[1],rho.shape[2]))
        #gives nearly uniform aphi at maxes
        #c=np.array([10,1.5,1,1,2.7,16])
        #gives uniform bsq/rho**gam at maxes
        c=np.array([3.5,2,1,2,3,10])
        phases=np.array([0,0.5,1,1.5,2,2.5])
        for i,phase in enumerate(phases):
            ap[i]=c[i]*gen_vpot(whichloop=0,phase=phase)
            ap1[i]=gen_vpot(whichloop=0,phase=phase)
        aaphi = np.sum(ap,axis=0)
        aaphi1 = np.sum(ap1,axis=0)
        aaphi2 = aaphi1 * fac(phi)
        plt.clf()
        if True:
            plt.plot(x1[:,ny/2,0],((ap[0]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((ap[1]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((ap[2]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((ap[3]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((ap[4]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((ap[5]))[:,ny/2,0])
            plt.plot(x1[:,ny/2,0],((aaphi))[:,ny/2,0])
        aphi2B(aaphi2)
        cvel()
        res=Qmri()
        #plt.plot(x1[:,ny/2,0],(res)[:,ny/2,0])
        #plt.clf();pl(x1,res)
        #plt.clf();pl(x1,aaphi)
        #plco(bsq/rho**gam,cb=True)
        #plco(res,cb=True)
        #pl(ti,10*fac(phi)); plt.ylim(0,ny-1)
        #pl(x1,res)
        #pl(x1,aaphi2)
        pl(x1,aaphi1)
    if False:
        rgfd("fieldline0000.bin")
        if False:
            #generate your favorite vector potential
            aaphi=gen_vpot(whichfield=None)
            #compute the field from that potential
            aphi2B(aaphi)
            B[2] = 1*B[2]
            cvel()
            #generate smoothing function
            profile = ((uqcomax-0.05)/0.1)
            profile[profile>1] = 1
            profile[profile<0] = 0
            #set target beta and desired bsqoug
            beta = 100.
            constbsqoug = 2*(gam-1)/beta
            #smooth bsqoug
            targbsqoug = constbsqoug*profile
            rat = ( targbsqoug/(bsq/ug+1e-15) )**0.5
            cvel()
        #rescale the field
        if False:
            B[1] *= rat
            B[2] *= rat
            #cvel()
            aphim=fieldcalcm()
            aphip=fieldcalcp()
            #aphi0 = avg0c2f(aphim)
            aphi2B(aphim)
            cvel()
        if False:
            rat2 = avg2ctof( rat )
            rat1 = avg1ctof( rat )
            gdetB[1] *= rat1
            gdetB[2] *= rat2
        if False:
            rat2 = avg2ctof( rat )
            rat1 = avg1ctof( rat )
            gdetB[1] *= rat1
            gdetB[2] *= rat2
            if False:
            #unsuccessful try to chop off the field spike in the middle of the loop
                minB1 = np.min(gdetB[1]/gdet)/1.5
                maxB1 = np.max(gdetB[1]/gdet)/1.5
                #gdetB1old=np.copy(gdetB[1])
                (gdetB[1])[gdetB[1]<gdet*minB1] = (minB1*gdet)[gdetB[1]<gdet*minB1]
                (gdetB[1])[gdetB[1]>gdet*maxB1] = (maxB1*gdet)[gdetB[1]>gdet*maxB1]
            #gdetB1new=np.copy(gdetB[1])
        if False:
            #at this point divb!=0, i.e. there are monopoles
            #to remove monopoles, compute vector potential
            aphi = fieldcalcface()
            #and compute the field from the potential
            #(this leaves B[1] the same and resets B[2]
            aphi2B(aphi)
        if False:
            cvel()
            normalize_field(constbsqoug)
            cvel()
        print("Disk flux = %g (@r<20: %g)" % (diskfluxcalc(ny/2), diskfluxcalc(ny/2,rmax=20)) )
    if False:
        #pf()
        grid3d("gdump.bin")
        rd("dump0000.bin")
        cvel()
        plco(np.log10(rho))
        plc(bsq/rho**gam)
        print("Disk flux = %g (@r<20: %g)" % (diskfluxcalc(ny/2,rmax=Rout), diskfluxcalc(ny/2,rmax=20)) )
        rh = 1+(1-a**2)**0.5
        print "r[5]/rh = %g\n" % (r[5,0,0]/rh) + "r[10]/rh = %g\n" % (r[10,0,0]/rh)
        res = Qmri()
        res[res>20] = 20+0*res[res>20]
        plc(res,cb=True)
        #plt.plot(x1[:,ny/2,0],(res)[:,ny/2,0])
        #plt.clf();pl(x1,res)
        #plt.clf();pl(x1,aaphi)
        #plco(bsq/rho**gam,cb=True)
        #plco(res,cb=True)
    if False:
        rin=15;
        R=r*np.sin(h);z=r*np.cos(h);
        alpha=1.5;t=0.9;aphi=(R/rin)**2/(1+(np.abs(z)/t/rin/(1+np.log10(1+r/rin)**2))**alpha)**(2/alpha); aphi[aphi>1]=0*aphi[aphi>1]+1; plco(np.log10(rho));plc(aphi)
    if False:
        grid3d("gdump.bin")
        rd("dump0040.bin")
        aphi=fieldcalcface()
        plco(np.log10(rho))
        plc(aphi,nc=50)
    if False:
        grid3d("gdump.bin")
        rfd("fieldline0222.bin")
        cvel()
        Tcalcud()
        jetpowcalc(whichbsqorho=0)[11]
        mdotcalc(11)
    if False:
        cvel()
        #entk=((gam-1)*ug/rho**gam);entk[entk>1]=0*entk[entk>1]+1;
        #plt.clf();
        #plt.figure();
        #pl(r,np.log10(entk));plt.xlim(1,20);plt.ylim(-3,-0.5)
    if False:
        #VRPLOT
        #grid3d("gdump.bin",use2d=True)
        #rd("dump0000.bin")
        #rfd("fieldline0000.bin")
        rhor=1+(1-a**2)**0.5
        ihor = np.floor(iofr(rhor)+0.5);
        #diskflux=diskfluxcalc(ny/2)
        #qtymem=None #clear to free mem
        #qtymem=getqtyvstime(ihor,0.2)
        plt.figure(1)
        plotqtyvstime(qtymem,whichplot=-3)
        #plt.figure(2)
        #plotqtyvstime(qtymem,whichplot=-4)

def placeletter(ax1,lab,size=16,fx=0.07,fy=0.07,ha="center",va="top",color='k',bbox=None):
    plt.text(
        ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])*fx,
        ax1.get_ylim()[0]+(ax1.get_ylim()[1]-ax1.get_ylim()[0])*(1-fy), 
        r"%s" % lab, size=size,
        rotation=0., ha=ha, va=va,
        color=color,weight='regular',bbox=bbox )


def icplot(dostreamlines=False,maxaphi=500,domakeframes=1,plotlen=85,ncont=100,doreload=True,aspect=2.0,vmin=-6.5,vmax=0.5,gs=None,fig=None,lwbold=4):
    global bsq, rho, gdetB
    bbox = dict(boxstyle="round,pad=0.1", fc="w", ec="w", alpha=0.5)
    #Rz
    if gs is None or fig is None:
        fig = plt.figure(1, figsize=(10,5), dpi=100)
        plt.clf()
        gs1 = GridSpec(2, 2)
        gs1.update(left=0.07, right=0.90, top=0.95, bottom=0.12, wspace=0.05)
    else:
        gs1 = gs
    #plt.subplots_adjust(hspace=0.02) #increase vertical spacing to avoid crowding
    if domakeframes:
        #RETROGRADE
        os.chdir("/home/atchekho/run/rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2")
        grid3d("gdump.bin",use2d=True)
        if doreload:
            avgmem = get2davg(usedefault=1)
            assignavg2dvars(avgmem)
            rho = avg_rho
            bsq = avg_bsq
            gdetB=np.zeros((4,nx,ny,nz))
            gdetB[1]=avg_gdetB[0]
            #rfd("fieldline3000.bin")
            #cvel()
            aphi = fieldcalc()
            aphibh=aphi[iofr(rhor),ny/2,0]
        ax2 = plt.subplot(gs1[1, 0])
        mkframe("topleft", vmin=vmin,vmax=vmax,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=dostreamlines,ncont=ncont,aspect=aspect,maxaphi=maxaphi)
        ax2.set_aspect('equal')   
        plc(aphi,levels=(aphibh,),xcoord=r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        plc(aphi,levels=(aphibh,),xcoord=-r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        ax2.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        ax2.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        placeletter( ax2,"$\mathrm{(b)}$",bbox=bbox)
        placeletter( ax2,"$t$-$,\\varphi$-$\mathrm{average}$",fx=0.97,ha="right",bbox=bbox)
        ax1 = plt.subplot(gs1[0, 0])
        if doreload:
            rfd("fieldline0000.bin")
            cvel()
            aphi=fieldcalc()
        mkframe("topleft", vmin=vmin,vmax=vmax,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=dostreamlines,ncont=ncont,aspect=aspect,maxaphi=maxaphi)
        plc(aphi,levels=(aphibh,),xcoord=r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        plc(aphi,levels=(aphibh,),xcoord=-r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        ax1.set_aspect('equal')   
        plt.setp( ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel(r'$z\ [r_g]$',fontsize=16,ha='center')
        placeletter( ax1,"$\mathrm{(a)}$",bbox=bbox)
        placeletter( ax1,"$t=%g$" % t,fx=0.97,ha="right",bbox=bbox)
        plt.title(r"${\rm Retrograde\ BH,\ a = -0.9,\ \eta = 34\%\ (model\ A-0.9f})$")
        #PROGRADE
        os.chdir("/home/atchekho/run/rtf2_15r34.1_pi_0_0_0")
        grid3d("gdump.bin",use2d=True)
        ax2 = plt.subplot(gs1[1, 1])
        if doreload:
            avgmem = get2davg(usedefault=1)
            assignavg2dvars(avgmem)
            rho = avg_rho
            bsq = avg_bsq
            gdetB=np.zeros((4,nx,ny,nz))
            gdetB[1]=avg_gdetB[0]
            # rfd("fieldline3000.bin")
            # cvel()
            aphi=fieldcalc()
            aphibh=aphi[iofr(rhor),ny/2,0]
        mkframe("topleft", vmin=vmin,vmax=vmax,len=plotlen,ax=ax2,cb=False,pt=False,dostreamlines=dostreamlines,ncont=ncont,aspect=aspect,maxaphi=maxaphi)
        plc(aphi,levels=(aphibh,),xcoord=r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        plc(aphi,levels=(aphibh,),xcoord=-r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        ax2.set_aspect('equal')   
        #plt.setp( ax2.get_yticklabels(), visible=False)
        ax2.set_xlabel(r'$x\ [r_g]$',fontsize=16)
        placeletter( ax2,"$\mathrm{(g)}$",bbox=bbox)
        placeletter( ax2,"$t$-$,\\varphi$-$\mathrm{average}$",fx=0.97,ha="right",bbox=bbox)
        #
        ax1 = plt.subplot(gs1[0, 1])
        if doreload:
            rfd("fieldline0000.bin")
            cvel()
            aphi=fieldcalc()
        mkframe("topleft", vmin=vmin,vmax=vmax,len=plotlen,ax=ax1,cb=False,pt=False,dostreamlines=dostreamlines,ncont=ncont,aspect=aspect,maxaphi=maxaphi)
        plt.setp( ax1.get_xticklabels(), visible=False)
        #plt.setp( ax1.get_yticklabels(), visible=False)
        plt.title(r"${\rm Prograde\ BH,\ a = 0.9,\ \eta=102\%\ (model\ A0.9f)}$")
        placeletter( ax1,"$\mathrm{(f)}$",bbox=bbox)
        placeletter( ax1,"$t=%g$" % t,fx=0.97,ha="right",bbox=bbox)
        plc(aphi,levels=(aphibh,),xcoord=r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        plc(aphi,levels=(aphibh,),xcoord=-r*np.sin(h),ycoord=r*np.cos(h),linestyles="solid",colors='k',linewidths=lwbold)
        ax1.set_aspect('equal')   
        #
        ax1 = fig.add_axes([0.94, 0.49, 0.02, 0.29])
        #
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                           norm=norm,
                                           orientation='vertical')
        tcks=[x for x in range(int(vmin),int(vmax)+1)]
        labs=[r'$10^{%d}$'%x for x in range(int(vmin),int(vmax)+1)]
        cb1.set_ticks(tcks)
        cb1.set_ticklabels(labs)
        cb1.update_ticks()
        ax1.text(ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/1., 
             1.*ax1.get_ylim()[1], r"$\ \rho c^2$", size=14, rotation=0.,
             ha="left", va="center",
             color='k',weight='regular'
             )

    plt.savefig("figic.eps",bbox_inches='tight',pad_inches=0.02)
    plt.savefig("figic.pdf",bbox_inches='tight',pad_inches=0.02)

def plotflux(doreload=True):
    global reslist, avgmemlist
    fig = plt.figure(1, figsize=(10,5), dpi=100)
    plt.clf()
    dirlist=["/home/atchekho/run/rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2",
             #"/home/atchekho/run/rtf2_15r34.1_0_0_0_spinflip",
             #"/home/atchekho/run/rtf2_15r34.1_betax0.5_0_0_0_2xphi_restart15000",
             "/home/atchekho/run/rtf2_15r34.1_betax0.5_0_0_0",
             "/home/atchekho/run/rtf2_15r34.1_pi_0_0_0",
             "/home/atchekho/run/rtf2_15r34.1_betax2_0_0_0",
             "/home/atchekho/run/rtf2_15r34.1_betax4_0_0_0"]
    #caplist=[r"$\mathrm{A-0.9f}$", r"$\mathrm{A0.9f}$", r"$\mathrm{A0.9N50}$", r"$\mathrm{A0.9N25}$"]
    caplist=[r"$\beta_{\rm min}=100$ $\mathrm{(A-0.9f)}$", 
             #r"$\beta_{\rm min}=100$ $\mathrm{(A-0.9flip)}$", 
             #r"$\beta_{\rm min}=200$ $\mathrm{(A0.9N200h_\varphi)}$", 
             r"$\beta_{\rm min}=200$ $\mathrm{(A0.9N200)}$", 
             r"$\beta_{\rm min}=100$ $\mathrm{(A0.9f)}$", 
             r"$\beta_{\rm min}=50$ $\mathrm{(A0.9N50)}$", 
             r"$\beta_{\rm min}=25$ $\mathrm{(A0.9N25)}$"]
    lslist=["-",#"--",#"--",
            "--","-","-.",":"]
    clrlist=["blue", #"cyan", #"orange", 
             "pink", "red","magenta","brown"]
    lwlist=[2,#2,#2,
            2,2,2,2]
    dirlist.reverse()
    caplist.reverse()
    lslist.reverse()
    clrlist.reverse()
    lwlist.reverse()
    crvlist1=[]
    crvlist2=[]
    lablist1=[]
    lablist2=[]
    if doreload:
        reslist=[]
        avgmemlist=[]
    for i,dirpath in enumerate(dirlist):
        os.chdir(dirpath)
        grid3d("gdump.bin",use2d=True)
        if doreload:
            res = takeoutfloors(doreload=doreload,isinteractive=0,writefile=False)
            avgmem = get2davg(usedefault=1)
            reslist.append(res)
            avgmemlist.append(avgmem)
        else:
            res = reslist[i]
            avgmem = avgmemlist[i]
        a_eta,a_Fm,a_Fe,a_Fl = res
        assignavg2dvars(avgmem)
        rho = avg_rho
        bsq = avg_bsq
        if True:
            aphi = fieldcalc(gdetB1=avg_gdetB[0])
            aphibh=aphi[iofr(rhor),ny/2,0]
        else:
            aphi = np.zeros_like(avg_B[0])
            aphi = (0.5*np.abs(avg_gdetB[0]).sum(1)*_dx2)[:,:,None]+aphi*0
            aphibh = aphi[iofr(rhor),0,0]
        #aphi = scaletofullwedge(nz*(avg_psisq)**0.5)
        #old way:
        #unitsfactor=(4*np.pi)**0.5*2*np.pi
        #phibh=fstot[:,ihor]/4/np.pi/FMavg**0.5*unitsfactor
        #where fstot = (gdetB1).sum(2).sum(1)*_dx2*_dx3 at horizon
        phibh = (4*np.pi)**0.5*aphi/a_Fm**0.5
        risco=Risco(a)
        iisco=iofr(risco)
        if dirpath == "/home/atchekho/run/rtf2_15r34.1_betax0.5_0_0_0_2xphi_restart15000" or \
           dirpath == "/home/atchekho/run/rtf2_15r34.1_betax0.5_0_0_0":
            iof10 = iofr(10)
            crv=plt.plot(r[:iof10,ny/2,0],phibh[:iof10,ny/2,0],label=caplist[i],ls=lslist[i],color=clrlist[i],lw=lwlist[i])
        else:
            crv=plt.plot(r[:,ny/2,0],phibh[:,ny/2,0],label=caplist[i],ls=lslist[i],color=clrlist[i],lw=lwlist[i])
        #print a, risco, iisco, r[iisco,ny/2,0], phibh[iisco,ny/2,0] 
        plt.plot(r[iisco,ny/2,0],phibh[iisco,ny/2,0],'o',color=clrlist[i],lw=lwlist[i])
        if a > 0:
            crvlist1.append(crv)
            lablist1.append(caplist[i])
        else:
            crvlist2.append(crv)
            lablist2.append(caplist[i])
    plt.xlim(rhor,20.-1e-5)
    plt.ylim(0,139.99)
    plt.xlabel(r'$r\ [r_g]$',fontsize=20)
    plt.ylabel(r'$\langle\phi(r,\theta=\pi/2)\rangle$        ',fontsize=22,ha="center")
    #plt.legend(loc="upper left",ncol=1)
    leg1=plt.legend(crvlist1,lablist1,loc="upper left",title=r"${\rm Prograde,}\ a=0.9$:",frameon=True,labelspacing=0.15,ncol=1)
    leg2=plt.legend(crvlist2,lablist2,loc="lower right",title=r"${\rm Retrograde,}\ a=-0.9$:",frameon=True,labelspacing=0.15)
    plt.gca().add_artist(leg1)
    for t in leg1.get_texts() + leg2.get_texts():
        t.set_fontsize(20)    # the legend text fontsize
    leg1.get_title().set_fontsize(20)
    leg2.get_title().set_fontsize(20)
    plt.grid(b=True)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(18)
    plt.savefig("plotflux.eps",bbox_inches='tight',pad_inches=0.02,dpi=100)
    plt.savefig("plotflux.pdf",bbox_inches='tight',pad_inches=0.02,dpi=100)

def mkpulsarmovie():
    grid3d("gdump.bin",use2d=True)
    flist = np.sort(glob.glob( os.path.join("dumps/", "fieldline[0-9][0-9][0-9][0-9].bin") ) )
    flist.sort()
    for fldindex, fldname in enumerate(flist):
        print( "Reading " + fldname + " ..." )
        sys.stdout.flush()
        rfd("../"+fldname)
        sys.stdout.flush()
        aphi=fieldcalc()
        if fldindex == 0:
            maxaphi = (5*10)**0.5*3*3*3.2 #aphi.max()
        #fig=plt.figure(1,figsize=(10,10))
        #plt.clf()
        #ax = fig.add_subplot(111, aspect='equal')
        numc=10
        cvel()
        if True:
            plco(aphi,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=np.arange(1,numc)*maxaphi/np.float(numc),colors='k')
            plc(np.log10(bsq/rho),xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True,levels=np.arange(0.,2,0.1));plt.xlim(0,10);plt.ylim(-5,5)
            #plc(uu[2]*dxdxp[2][2],xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=False,levels=np.arange(-0.5,0.5,0.1));plt.xlim(0,10);plt.ylim(-5,5)            #plc(np.log10(ug),xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True,levels=np.arange(-3,2,0.1));plt.xlim(0,10);plt.ylim(-5,5)
            #plc(np.log10(ug/rho),xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True,levels=np.arange(0,1,0.1));plt.xlim(0,10);plt.ylim(-5,5)
            #plc(lrho,xcoord=r*np.sin(h),ycoord=r*np.cos(h),cb=True,levels=np.arange(-3,.8,0.1));plt.xlim(0,10);plt.ylim(-5,5)
            #cts=plc(bsq/rho,xcoord=r*np.sin(h),ycoord=r*np.cos(h),levels=np.arange(1,5.5,0.5))
            #cbar=plt.colorbar(cts)
            #cbar.ax.set_ylabel(r'$b^2\!/\rho$',fontsize=16)
            #plco(lrho,cb=True,levels=np.arange(1,10),xcoord=r*np.sin(h),ycoord=r*np.cos(h));plt.xlim(0,10);plt.ylim(-5,5)
            #draw NS
            ax=plt.gca()
            ax.set_aspect('equal')   
            el = Ellipse((0,0), 2, 2, facecolor='k', alpha=1)
            art=ax.add_artist(el)
            art.set_zorder(20)
            rmax = 10
            plt.xlim(0,rmax)
            plt.ylim(-0.5*rmax,0.5*rmax)
        else:
            plt.clf()
            plt.plot(r[:,0,0],ug[:,0.75*ny,0],'g');plt.xlim(Rin,10);plt.ylim(0,50)
            plt.plot(r[:,0,0],rho[:,0.75*ny,0],'r');plt.xlim(Rin,10);plt.ylim(0,50)
            plt.plot(r[:,0,0],(bsq/rho)[:,0.75*ny,0],'b');plt.xlim(Rin,10);plt.ylim(0,50)
        plt.title(r"$b^2\!/\rho=10^2$, t=%3.3g" % t,    fontsize=16, color='k')
        plt.savefig( 'frame%04d.png' % fldindex )
        #
        plt.draw()
        #if fldindex >= 500:
        #    break

def mkjetretrofig1():
        fig=plt.figure(1, figsize=(12,9), dpi=100)
        gs2 = GridSpec(2, 2)
        gs2.update(left=0.053, right=0.93, top=0.78, bottom=0.49, hspace=0.04, wspace=0.085)
        icplot(gs=gs2,fig=fig,aspect=3.9,plotlen=90,lwbold=2)
        #################
        #
        # mdot, phibh, etabh
        #
        #################
        fti=8000
        ftf=1e5
        sigma=None
        doreload = 1
        #plt.clf()
        gs3a = GridSpec(3, 3)
        gs3a.update(left=0.055, right=0.4735, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
        title = r"${\rm Retrograde\ BH,\ a = -0.9,\ \eta = 34\%\ (model\ A-0.9f})$"
        os.chdir("/home/atchekho/run/rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2")
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,fti=fti,ftf=ftf,returndf=1,isinteractive=0,writefile=False)
        print epsFm, epsFke, epsetaj, epsFm30
        mkmdot(doreload=doreload,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=100.,sigma=sigma,usegaussianunits=True,arrowsize=0.5,gs3=gs3a,dotwinx=False,lab=["c","d","e"],title=None,plotFM30=False)
        gs3b = GridSpec(3, 3)
        gs3b.update(left=0.5125, right=0.96, top=0.42, bottom=0.06, wspace=0.01, hspace=0.04)
        title=r"${\rm Prograde\ BH,\ a = 0.9,\ \eta = 102\%\ (model\ A0.9f)}$"
        os.chdir("/home/atchekho/run/rtf2_15r34.1_pi_0_0_0")
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,fti=fti,ftf=ftf,returndf=1,isinteractive=0,writefile=False)
        print epsFm, epsFke, epsetaj, epsFm30
        mkmdot(doreload=doreload,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,prefactor=100.,sigma=sigma,usegaussianunits=True,arrowsize=0.5,gs3=gs3b,dotwinx=True,doylab=False,lab=["h", "i", "j"],title=None,plotFM30=False)
        plt.savefig("plotmkmdot.eps",bbox_inches='tight',pad_inches=0.02)
        plt.savefig("plotmkmdot.pdf",bbox_inches='tight',pad_inches=0.02)


def radavg(vec):
    avgvec = np.copy(vec)
    avgvec[1:-1] = (avgvec[0:-2]+avgvec[1:-1]+avgvec[2:])/3.
    return(avgvec)

def plotBavg(doradavg=True):
    Br = dxdxp[1,1]*avg_B[0]+dxdxp[1,2]*avg_B[1]
    Bh = dxdxp[2,1]*avg_B[0]+dxdxp[2,2]*avg_B[1]
    Bp = avg_B[2]*dxdxp[3,3]
    #
    Brnorm=Br
    Bhnorm=Bh*np.abs(r)
    Bpnorm=Bp*np.abs(r*np.sin(h))
    #
    Bznorm=Brnorm*np.cos(h)-Bhnorm*np.sin(h)
    BRnorm=Brnorm*np.sin(h)+Bhnorm*np.cos(h)
    #B_z
    Bznorm = radavg(Bznorm)
    Bpnorm = radavg(Bpnorm)
    dj=5
    plt.plot(r[:,ny/2+dj,0],Bznorm[:,ny/2+dj,0],label=r"$B_z$",color="blue")
    #B_{\hat,phi}
    plt.plot(r[:,ny/2+dj,0],Bpnorm[:,ny/2+dj,0],label=r"$B_{\hat \varphi}$",color="red")
    plt.plot(r[:,ny/2+dj,0],-Bpnorm[:,ny/2+dj,0],ls=":",color="red")  #,label=r"$-B_{\hat \varphi}$"
    plt.plot(r[:,ny/2+dj,0],(Bpnorm/Bznorm)[:,ny/2+dj,0],label=r"$B_{\hat \varphi}/B_z$",color="green")
    plt.plot(r[:,ny/2+dj,0],(-Bpnorm/Bznorm)[:,ny/2+dj,0],color="green",ls=":") #,label=r"$-B_{\hat \varphi}/B_z$"
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(rhor,100)
    plt.ylim(1e-4,10)
    plt.grid(visible=True)
    plt.legend(loc="lower center")
    
def plotQmriavg(hor=None):
    global Q2mri
    res=Qmriavg()
    resphi=Qmriavg(dir=3)
    plt.clf()
    #
    gs3 = GridSpec(3, 3)
    #gs3.update(left=0.05, right=0.95, top=0.30, bottom=0.03, wspace=0.01, hspace=0.04)
    #mdot
    ax31 = plt.subplot(gs3[-3,:])
    plt.plot(r[:,ny/2,0],res[:,ny/2,0],color="red",label=r"$\mathrm{Q1mri}_\theta$")
    plt.plot(r[:,ny/2,0],10*(resphi*(np.abs(h-np.pi/2)<hor[:,None,None]))[:,:,0].mean(-1),color="blue",label=r"$10\!\times\mathrm{Q1mri}_\varphi$")
    plt.xlim(rhor,30)
    if hor is not None:
        print( "Plotting Q2mri...")
        lambdamri = res*_dx2*dxdxp[2,2]
        Q2mri = 2*hor[:,None,None]/lambdamri
        #pdb.set_trace()
        plt.plot(r[:,ny/2,0],Q2mri[:,ny/2,0]*100,color="green",label=r"$\mathrm{Q2mri\ [\%]}$")
    plt.ylim(0,150)
    #plt.xlabel(r"$r\ [r_g]$",fontsize=16)
    plt.ylabel("Q#mri",fontsize=16)
    plt.grid()
    plt.legend(ncol=3,borderpad = 0.3,borderaxespad=0.2,frameon=False,labelspacing=0)
    ax32 = plt.subplot(gs3[-2,:])
    vatheta = np.abs(avg_bu[2])/np.sqrt(avg_rho+avg_bsq+gam*avg_ug)*dxdxp[2,2]
    plt.plot(r[:,ny/2,0],10*vatheta[:,ny/2,0],color="red",label=r"$10\!\times v_{\rm A}^{\theta}$")
    omega = (dxdxp[3][3]*np.abs(avg_uu[3])/avg_uu[0])
    plt.plot(r[:,ny/2,0],omega[:,ny/2,0],color="green",label=r"$\Omega$")
    plt.xlim(rhor,30)
    plt.ylim(1e-3,1)
    plt.yscale("log")
    #plt.xlabel(r"$r\ [r_g]$",fontsize=16)
    plt.ylabel(r"$v_{\rm A}^\theta, \Omega$",fontsize=16)
    plt.grid()
    plt.legend(ncol=2,borderpad = 0.1,borderaxespad=0.2,frameon=True,labelspacing=0)
    ax34 = plt.subplot(gs3[-1,:])
    plt.plot(r[:,ny/2,0],hor,color="red",label=r"$h/r$")
    plt.xlim(rhor,30)
    plt.ylim(0,0.5)
    plt.xlabel(r"$r\ [r_g]$",fontsize=16)
    plt.ylabel(r"$h/r$",fontsize=16)
    plt.grid()
    plt.legend(ncol=2,borderpad = 0.1,borderaxespad=0.2,frameon=True,labelspacing=0,loc="lower right")
    rfid=10
    i = iofr(rfid)
    dr = dxdxp[1,1]*_dx1
    dz = dxdxp[2,2]*_dx2*r
    dy = dxdxp[3,3]*_dx3*r
    print( "%3g : %3g : %3g" % ( (dr/dz)[i,ny/2,0], (dz/dz)[i,ny/2,0], (dy/dz)[i,ny/2,0] ) )
    #plt.xscale("log")


def plotallbz():    
    readmytests1()
    plt.figure(1,figsize=(12,8))
    plt.clf()
    factor_to_convert_to_cgs = (4*np.pi)**(-1)*(2*np.pi)**(-2)
    y_value_to_norm_by = (mpow2abz[mhor6==0]*factor_to_convert_to_cgs)[-1]
    x_value_to_norm_by = 0.5
    mya = mspina2[mhor6==0]
    myrhor = 1+(1-mya**2)**0.5
    h = mya/myrhor
    #angular integration prefactor
    fh = ((1+h**2)/h**2)*((h+1/h)*np.arctan(h)-1)*3./2.
    #magnetic field
    bh = (4/(myrhor**2+mya**2))
    plee = mpow2abz[mhor6==0]*bh**2*fh
    #plt.plot(momh2[mhor2==0]/x_value_to_norm_by,0.0534*momh2[mhor2==0]**2*(4*np.pi,lw=2)**(-1,lw=2)/y_value_to_norm_by,'x',lw=2)
    l3,=plt.plot(momh6[mhor6==0]/x_value_to_norm_by,mpow6[mhor6==0]*factor_to_convert_to_cgs/y_value_to_norm_by,lw=2,label=r"${\rm BZ6,\ P_{\rm jet}\propto \Omega_{\rm H}^2[1+\alpha (\Omega_{\rm H}r_g/c)^2\!+\beta (\Omega_{\rm H}r_g/c)^{4}]\ (TNM10)}$",color='r')
    l1,=plt.plot(momh2[mhor2==0]/x_value_to_norm_by,mpow2a[mhor2==0]*factor_to_convert_to_cgs/y_value_to_norm_by,lw=2,ls='--',label=r"${\rm BZ2,\ P_{\rm jet}\propto \Omega_{\rm H}^2 \ \ \ \ \ \ \ \ \ (TNM10)}$",color='b')
    l2,=plt.plot(momh2[mhor2==0]/x_value_to_norm_by,mpow2abz[mhor2==0]*factor_to_convert_to_cgs/y_value_to_norm_by,lw=2,ls='-.',label=r"${\rm BZ,\ P_{\rm jet}\propto a^2 \ \ \ \ \ \ \ \ \ \ \ \ (BZ77)}$",color='g')
    l4,=plt.plot(momh2[mhor2==0]/x_value_to_norm_by,plee*factor_to_convert_to_cgs/y_value_to_norm_by,lw=2,ls=':',label=r"${\rm LWB00,\ P_{\rm jet}\propto a^2f(h)/r_{\rm H}^4\ \ \ \ \ \ \ \ \ (LWB00)}$",color='cyan')
    l1.set_dashes([10,5])
    l2.set_dashes([10,3,2,3])
    l4.set_dashes([10,5,5,5])
    if False:
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.xscale('linear')
        plt.yscale('linear')
    plt.grid()
    leg1=plt.legend(loc="upper left",title=r"${\rm BHs\ with\ razor-thin\ disks}\ (H/R=0)$:",frameon=True, fancybox=True) #,labelspacing=0.15,ncol=1)
    lab4=r"${\rm BZ6},\ P_{\rm jet}\propto\Omega_{\rm H}^4\ {\rm \ \ \ \ \ \ \ \ (TNM10)} \quad$"
    l4,=plt.plot(momh6[mhor6==1]/x_value_to_norm_by,mpow6[mhor6==1]*factor_to_convert_to_cgs/y_value_to_norm_by,lw=2,ls=':',label=lab4,color='k')
    l4.set_dashes([20,5])
    leg2=plt.legend([l4],[lab4],loc="center left",title=r"${\rm BHs\ with\ thick\ disks}\ (H/R=1)$:",frameon=True, fancybox=True) #,labelspacing=0.15)
    plt.gca().add_artist(leg1)
    #for t in leg1.get_texts() + leg2.get_texts():
    #    #t.set_fontsize(16)    # the legend text fontsize
    #    t.set_horizontalalignment("left")    # doesn't work
    leg1.get_title().set_horizontalalignment("left")
    leg1.get_title().set_fontsize(16)
    leg2.get_title().set_fontsize(16)

    #plt.legend(loc='upper left', frameon=True, fancybox=True)
    plt.ylabel(r"${\rm Jet\ power},\ P_{\rm jet}(a)/P_{\rm BZ}(a=1)$",fontsize=20)
    plt.xlabel(r"${\rm Black\ hole\ angular\ frequency},\ \Omega_{\rm H}(a)/\Omega_{\rm H}(a=1)$",fontsize=20)
    plt.xlim(0,1)
    plt.ylim(ymin=1e-5,ymax=5-1e-5)
    ax1=plt.gca()
    # tck_val = ax1.get_xticks()
    # tck_lab=["%g"%np.rint(x) for x in tck_val]
    # ax1.set_xticks(tck_val)
    # ax1.set_xticklabels(tck_lab)
    ax2 = ax1.twinx()
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_ylim(ax1.get_ylim())
    ax2 = ax1.twiny()
    tck_lab=[0,0.2,0.4,0.6,0.8,0.9,0.95,0.98,0.998,1]
    tck_val=[a/2./(1+(1-a**2)**0.5)/x_value_to_norm_by for a in tck_lab]
    ax2.set_xticks(tck_val)
    ax2.set_xticklabels(tck_lab)
    plt.xlabel(r"${\rm Black\ hole\ spin},\ a$",fontsize=20)
    plt.savefig("figbz.eps",bbox_inches='tight',pad_inches=0.02)
    plt.savefig("figbz.pdf",bbox_inches='tight',pad_inches=0.02)

def plotbetajet():
    #load metric
    #[here, use2d=True instructs the routines to make use of axisymmetry of the metric, which saves memory]
    grid3d("gdump.bin",use2d=True)
    #load time-averages
    #avgmem=rdavg2d(fname="avg2d20_0200_0314.npy") #,usedefault=1
    #avgmem=rdavg2d(fname="avg2d20_0070_0305.npy") #,usedefault=1
    avgmem=rdavg2d(usedefault=1)
    usestaggeredfluxes = False
    #remove floors from mass (Fm) and extractable energy (Fm-Fe) fluxes
    Fm_floorremoved, FmMinusFe_floorremoved1, FmMinusFe_floorremoved2 \
        = removefloorsavg2d(usestaggeredfluxes=usestaggeredfluxes)
    pg = (gam-1)*avg_ug
    pm = avg_bsq/2.
    beta = pg/pm
    #num = (beta*FmMinusFe_floorremoved1*_dx2*_dx3).sum(-1).mean(-1)
    num = (avg_bsq/avg_rho*FmMinusFe_floorremoved1*_dx2*_dx3).sum(-1).mean(-1)
    den = (FmMinusFe_floorremoved1*_dx2*_dx3).sum(-1).mean(-1)
    ans = num/den
    plt.plot(r[:,0,0],ans)
    plt.xlim(rhor,1e5)
    plt.ylim(1e-3,1e3)
    plt.xscale('log')
    plt.yscale('log')
    return(ans)

def plotbsqorhosigma():    
        #load metric
        #[here, use2d=True instructs the routines to make use of axisymmetry of the metric, which saves memory]
        grid3d("gdump.bin",use2d=True)
        #load time-averages
        avgmem=rdavg2d(usedefault=1)
        if os.path.basename(os.getcwd()) == "rtf2_15r34_2pi_a-0.9gg50rbr1e3_0_0_0_faildufix2":
            #^^^ hack ^^^ to avoid using this for all models except the a = -0.9 model (for now)
            #this is because some of the models were restarted half-way with this diagnostic added,
            #and so their averages will not be correct (since the missing data is filled with zeros)
            #---> saved face-centered fluxes exist
            usestaggeredfluxes = True
        else:
            usestaggeredfluxes = False
        #remove floors from mass (Fm) and extractable energy (Fm-Fe) fluxes
        Fm_floorremoved, FmMinusFe_floorremoved1, FmMinusFe_floorremoved2 \
            = removefloorsavg2d(usestaggeredfluxes=usestaggeredfluxes)
        lab = "(%g,%g)" % (np.rint(avg_ts[0]),np.rint(avg_te1[0]))
        plt.figure(1)
        plt.clf()
        ax = plt.gca()
        a_Fm = (Fm_floorremoved[:,:,0:1]*_dx2*_dx3).sum(-1).sum(-1)
        a_Fm_raw = (-gdet[:,:,0:1]*avg_rhouu[1]*_dx2*_dx3*nz).sum(-1).sum(-1)
        sigvalfm = a_Fm / (-4*np.pi*r[:,ny/2,0]*avg_uu[1,:,ny/2,0]*dxdxp[1,1,:,0,0]/avg_uu[0,:,ny/2,0])
        corrfactor=Fm_floorremoved/(-gdet[:,:,0:1]*avg_rhouu[1]*nz)
        corrfactor[avg_uu[1]>0]=corrfactor[avg_uu[1]>0]*0+1
        sigvalcorr = (gdet[:,:,0:1]*avg_rhouu[0]*corrfactor*_dx2*_dx3*nz).sum(-1).sum(-1)/dxdxp[1,1,:,0,0]*scaletofullwedge(1.)/(2*np.pi*r[:,ny/2,0])
        sigval = (gdet[:,:,0:1]*avg_rhouu[0]*_dx2*_dx3*nz).sum(-1).sum(-1)/dxdxp[1,1,:,0,0]*scaletofullwedge(1.)/(2*np.pi*r[:,ny/2,0])
        plt.plot( r[:,0,0], sigval, 'k', label=r"$\Sigma\ "+lab+"$" )
        plt.plot( r[:,0,0], sigvalcorr, 'g', label=r"$\Sigma_{\rm corr}\ "+lab+"$" )
        plt.plot( r[:,0,0], sigvalfm, 'r', label=r"$\Sigma_{\rm fm}\ "+lab+"$" )
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(rhor,100)
        plt.ylim(0.1,15)
        plt.xlabel(r"$r$",fontsize=16)
        plt.ylabel(r"$\Sigma$",fontsize=16)
        plt.grid(b=True)
        plt.legend(loc='best')
        plt.figure(2)
        plt.clf()
        plt.plot(r[:,0,0],a_Fm_raw)
        plt.plot(r[:,0,0],a_Fm)
        plt.xlim(rhor,20)
        plt.ylim(0,20)
        plt.grid()
        plt.figure(3)
        plt.clf()
        ax = plt.gca()
        plt.plot(r[:,0,0],avg_bsq[:,ny/2,0]/(sigvalcorr/r[:,ny/2,0]**2)*(r[:,ny/2,0]/rhor)**1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(rhor,100)
        plt.ylim(0.1,10)
        plt.grid()
        plt.figure(4)
        plt.clf()
        ax = plt.gca()
        plt.plot(r[:,0,0],gam*avg_ug[:,ny/2,0]/avg_rho[:,ny/2,0]**gam*(r[:,ny/2,0]/rhor)**0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(rhor,1e4)
        plt.ylim(0.001,10)
        plt.grid()

def extract_trajjet(trajdisk, trajpole,di=1):
    #obtain the closest approach
    x1min = trajpole[0].min()
    imin = trajpole[0].argmin()
    print imin
    x2min = trajpole[1][imin]
    x2mid = 0.5*(x2.max()+x2.min())
    #orient array in increasing coordinate order and update the index
    if (x2min > x2mid and (trajpole[1][imin-di] > trajpole[1][imin+di])) or \
       (x2min < x2mid and (trajpole[1][imin-di] < trajpole[1][imin+di])):
        trajpole = trajpole[0][::-1], trajpole[1][::-1]
        imin = trajpole[0].argmin()
    traj1jet = trajdisk[0][trajdisk[0]<x1min], trajdisk[1][trajdisk[0]<x1min]
    traj2jet = trajpole[0][imin+1:], trajpole[1][imin+1:]
    return np.concatenate((traj1jet[0],traj2jet[0])), np.concatenate((traj1jet[1],traj2jet[1]))
    
if __name__ == "__main__":
    if False:
        #compute energy flux weighted pg/pm
        plotbetajet()
    if False:
        grid3d("gdump.bin",use2d=True)
        #load time-averages
        if os.path.isfile("avg2d20_0264_0314.npy"):
            #use late-time average for a = 0.99 simulation
            avgmem=rdavg2d(fname="avg2d20_0264_0314.npy")  #specify a file name
        else:
            avgmem=rdavg2d(usedefault=1)  #usedefault=1 reads in from "avg2d.npy"
        if 'qtymem' not in globals():
            qtymem = getqtyvstime( iofr(rhor) )
        ts=qtymem[0,:,0]
        hoverr=qtymem[1]
        hoverravg=timeavg(hoverr,ts,avg_ts[0],avg_te1[0])
        plt.figure(1)
        plotQmriavg(hor=hoverravg)
        plt.figure(2)
        plt.clf()
        plotBavg()
    if False:
        if False:
            grid3d("gdump.bin",use2d=True)
            avgmem=rdavg2d(usedefault=1)  #usedefault=1 reads in from "avg2d.npy"
            trajdisk_up,trajpole_up = finddiskjetbnds(r0=10,upperx2=True,doplot=True)
            trajdisk_dn,trajpole_dn = finddiskjetbnds(r0=10,upperx2=False,doplot=True)
        #plt.clf()
        #plt.plot(trajdisk_up[0],trajdisk_up[1],'b',lw=2)
        #plt.plot(trajdisk_dn[0],trajdisk_dn[1],'b',lw=2)
        #plt.plot(trajpole_up[0],trajpole_up[1],'r',lw=2)
        #plt.plot(trajpole_dn[0],trajpole_dn[1],'r',lw=2)
        trajjet_up = extract_trajjet(trajdisk_up,trajpole_up)
        trajjet_dn = extract_trajjet(trajdisk_dn,trajpole_dn)
        #plt.plot(trajjet_up[0],trajjet_up[1],'y--',lw=2)
        #plt.plot(trajjet_dn[0],trajjet_dn[1],'y--',lw=2)
        #interpolate to our grid (x1,x2) -> x2(ti)
        jet1x2 = interp1d(trajjet_dn[0],trajjet_dn[1], kind = 'linear', bounds_error=False)(x1[:,0,0])
        jet2x2 = interp1d(trajjet_up[0],trajjet_up[1], kind = 'linear', bounds_error=False)(x1[:,0,0])
        F_jet1, F_jet2, F_wind = removefloorsavg2djetwind(usestaggeredfluxes=False,DFfloor=None, jet1x2=jet1x2, jet2x2=jet2x2)
        plt.draw()
    if False:
        #takeoutfloors(dotakeoutfloors=1,doplot=True,doreload=1,isinteractive=1,writefile=True,aphi_j_val=0)
        #takeoutfloors(dotakeoutfloors=1,doplot=True,doreload=1,isinteractive=1,writefile=False,aphi_j_val=0)
        #use this in a shell script
        grid3d( "gdump.bin",use2d=True )
        avgmem=rdavg2d(usedefault=1)
        takeoutfloors(dotakeoutfloors=1,doplot=False,doreload=1,isinteractive=1,writefile=True,aphi_j_val=0)
        #takeoutfloors(dotakeoutfloors=1,doplot=True,doreload=1,isinteractive=1,writefile=False,aphi_j_val=0)
        #takeoutfloors(dotakeoutfloors=1,doplot=False)
    if False:
        provsretro()
    if False:
        #make a movie
        #fti=7000
        #ftf=30500
        doreload = 1
        domakeframes=1
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,returndf=1,isinteractive=0)
        #epsFm = 
        #epsFke = 
        #print epsFm, epsFke
        mkmovie(prefactor=100.,sigma=1500.,usegaussianunits=True,domakeframes=domakeframes)
        #mkmovie(prefactor=100.,usegaussianunits=True,domakeframes=domakeframes)
    if False:
        #make a movie
        #fti=7000
        #ftf=30500
        doreload = 1
        domakeframes=1
        #epsFm, epsFke, epsFm30 = takeoutfloors(doreload=doreload,returndf=1,isinteractive=0)
        #epsFm = 
        #epsFke = 
        #print epsFm, epsFke
        mkmovie(prefactor=100.,usegaussianunits=True,domakeframes=domakeframes,frametype='Rzzypanels',dostreamlines=False)
        #mkmovie(prefactor=100.,usegaussianunits=True,domakeframes=domakeframes)
    if False:
        #make a movie with floor removal
        fti=7000
        ftf=30500
        grid3d( "gdump.bin",use2d=True )
        avgmem=rdavg2d(usedefault=1)
        doreload = 1
        domakeframes=1
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,returndf=1,isinteractive=0,doplot=False,writefile=False)
        #epsFm = 
        #epsFke = 
        #print epsFm, epsFke
        mkmovie(prefactor=100.,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,usegaussianunits=True,domakeframes=domakeframes,frametype='5panels',dostreamlines=True,sigma=1500)
        #mkmovie(prefactor=100.,usegaussianunits=True,domakeframes=domakeframes)
    if False:
        readmytests1()
        plotpowers('powerlist.txt',format=0) #old format
    if False:
        #Figure 3
        readmytests1()
        plotpowers('powerlist2davg.txt',format=1) #new format; data from 2d average dumps
    if False:
        #Pro vs. retrograde spins, updated diagnostics
        readmytests1()
        plotpowers('siminfo.txt',plotetas=True,format=2) #new format; data from 2d average dumps
    if False:
        #Jet efficiency vs. spin, update diagnostics
        readmytests1()
        plotpowers('siminfo.txt',plotetas=False,format=2) #new format; data from 2d average dumps
    if False:
        #Plot all BZs
        plotallbz()
    if True:
        #Power vs. spin, updated diagnostics
        readmytests1()
        plotpowers('siminfo.txt',plotetas=False,format=2) #new format; data from 2d average dumps
    if False:
        #2DAVG
        mk2davg()
    if False:
        #NEW FORMAT
        #Plot qtys vs. time
        generate_time_series(docompute=True)
    if False:
        #make a movie
        fti=7000
        ftf=30500
        doreload = 1
        domakeframes=1
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,fti=fti,ftf=ftf,returndf=1,isinteractive=0)
        #epsFm = 
        #epsFke = 
        #print epsFm, epsFke
        mkmovie(prefactor=100.,sigma=1500.,usegaussianunits=True,domakeframes=domakeframes)
    if False:
        #make a movie
        #fti=7000
        #ftf=30500
        doreload = 1
        domakeframes=1
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,returndf=1,isinteractive=0)
        #epsFm = 
        #epsFke = 
        #print epsFm, epsFke
        mkmovie(prefactor=100.,sigma=1500.,usegaussianunits=True,domakeframes=domakeframes)
        #mkmovie(prefactor=100.,usegaussianunits=True,domakeframes=domakeframes)
    if False:
        #fig2 with grayscalestreamlines and red field lines
        #mkstreamlinefigure(length=30,doenergy=False,frameon=True,dpi=600,showticks=False)
        if True: #remove floors
            mkstreamlinefigure(length=29.99,doenergy=False,frameon=True,dpi=600,showticks=True,dotakeoutfloors=1,usedefault=1)
        else: #don't do anything about floors
            mkstreamlinefigure(length=29.99,doenergy=False,frameon=True,dpi=600,showticks=True,dotakeoutfloors=0,usedefault=1)
        #mkstreamlinefigure(length=30,doenergy=False,frameon=True,dpi=600,showticks=True,dotakeoutfloors=0)
        #mkstreamlinefigure(length=4,doenergy=False)
    if False:
        #FIGURE 1 from jetretro w/ mdot, phibh, etabh
        mkjetretrofig1()
    if False:
        #FIGURE 1 LOTSOPANELS
        fti=7000
        ftf=30500
        doreload = 1
        domakeframes=1
        epsFm, epsFke, epsetaj, epsFm30 = takeoutfloors(doreload=doreload,fti=fti,ftf=ftf,returndf=1,isinteractive=0)
        #epsFm = 
        #epsFke = 
        print epsFm, epsFke, epsFm30
        mklotsopanels(doreload=doreload,epsFm=epsFm,epsFke=epsFke,epsetaj=epsetaj,epsFm30=epsFm30,fti=fti,ftf=ftf,domakeframes=domakeframes,prefactor=100.,sigma=1500.,usegaussianunits=True,arrowsize=0.5)
    if False:
        grid3d( "gdump.bin",use2d=True )
        fno=0
        rfd("fieldline%04d.bin" % fno)
        cvel()
        plt.clf();
        mkframe("lrho%04d" % 0, vmin=-8,vmax=0.2,dostreamlines=True,len=50)
        plt.savefig("lrho%04d.pdf" % fno)
    if False:
        mkpulsarmovie()
    if False:
        #Short tutorial.
        print( "Running a short tutorial: read in grid, 0th dump, plot and compute some things." )
        #1 read in gdump (specifying "use2d=True" reads in just one r-theta slice to save memory)
        grid3d("gdump.bin", use2d = True)
        #2 read in dump0000
        doreaddump = 0
        if doreaddump:
            rd("dump0000.bin")
        #   or, instead of dump, you could read in fieldline0000.bin
        rfd("fieldline0000.bin")
        #3 compute extra things
        docomputeextrathings = False
        if docomputeextrathings:
            cvel()
            Tcalcud()
            faraday()
        #4 compute vector potential
        aphi = fieldcalc()
        #5 plot density and overplotted vector potential
        plt.figure(1)  #open figure 1
        plco(lrho,cb=True,nc=50) #plco -- erases and plots; cb=True tells it to draw color bar, nc = number of contours
        plc(aphi,colors='k') #plc -- overplots without erasing; colors='k' says plot in blac'k'
        #5a compute same in x-z coordinates
        fig = plt.figure(2)  #open figure 2
        ax = fig.add_subplot(111, aspect='equal')
        plco(lrho,cb=True,nc=50,xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plc(aphi,nc=100,colors='k',xcoord=r*np.sin(h),ycoord=r*np.cos(h))
        plt.xlim(0,50)
        plt.ylim(-25,25)
        ax.set_aspect('equal')   
        #6 compute u^\phi
        uuphi = uu[3] * dxdxp[3,3]
        #7 compute u_\phi
        #  first, lower the index
        ud_computed = mdot(gv3,uu)  #<-- this is already computed as 'ud' inside of cvel() call
        #  then, take 3rd component and convert to phi from x3
        udphi = (ud_computed/dxdxp[3,3])[3]
        #8 phi-average density
        rhophiavg = rho.mean(axis=-1)  #-1 says to average over the last dimension
        #9 clean up some memory
        ud_computed = None
        uuphi = None
        udphi = None
        aphi = None
        gc.collect()
    if False:
        #######################
        #
        #  Example: compute b^2/rho/Sigma
        #
        #######################
        plotbsqorhosigma()
        
