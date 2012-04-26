import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from streamlines import streamplot
from streamlines import fstreamplot
from pychip import pchip_init, pchip_eval
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
from scipy.integrate import simps
from scipy.optimize import brentq
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
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



def compute_dNp_dEp( dNg_dEg ):
    #dNp_dEp = 
    print("nothing")

class SeedPhoton:
    """our seed photon class"""
    def __init__(self,Emin,Emax,s):
        self.Emin = Emin
        self.Emax = Emax
        self.s = s
        #minimum energy gamma-ray to be able to pair produce
        self.Egmin = 2/Emax
        self.Nprefactor = (1-s)/(Emax**(1-s)-Emin**(1-s))

    def canPairProduce(self,E):
        return( E > self.Egmin )

    def f(self,E):
        return( self.Nprefactor*E**(-self.s)*(E >= self.Emin)*(E <= self.Emax) )

def fmagic( E ):
    return( E*(E>0)+1*(E<=0) )


def fg( Eg, Ee, seed):
    Eseed = Eg/(2*Ee*(fmagic(Ee-Eg)))
    fEseed = seed.f(fmagic(Eseed))
    fgval = fEseed / (2*(fmagic(Ee-Eg))**2)
    fgval *= (Ee-Eg>0)*(Eseed>0)
    # del Eseed
    # del fEseed
    # gc.collect()
    return( fgval )

def K( Enew, Eold, seed ):
    K = 4*fg(2*Enew,Eold,seed)*(2*Enew>=seed.Egmin)+fg(Eold-Enew,Eold,seed)
    return( K )

def flnew( Evec, flold, seed, nskip = 1 ):
    """Expect E and flold defined on a regular log grid, Evec"""
    dx = np.log(Evec[1]/Evec[0])
    x = np.log(Evec)
    flnew = np.empty_like(flold)
    for i in xrange(0,int(len(Evec)/nskip)):
        flnew[i*nskip:(i+1)*nskip] = simps( K(Evec[i*nskip:(i+1)*nskip,None],Evec[None,:],seed)*(flold*Evec)[None,:], dx=dx,axis=-1 )         
        # gc.collect()
    return( flnew )

if __name__ == "__main__":
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
    Emax = 1e14
    Ngrid = 1e3
    Evec = exp(np.linspace(0,np.log(Emax),Ngrid))
    ivec = np.arange(len(Evec))
    #1 eV in units of m_e c^2
    eV = 1/(512.e3)
    #spectral index
    s = 2
    #lower cutoff
    Esmin = 0.5e-3 * eV
    Esmax = 2 * eV
    seed = SeedPhoton( Esmin, Esmax, s )
    #
    Ngenmax = 10
    #
    E0 = 1e13
    ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
    dx = np.log(Evec[1]/Evec[0])
    dE = Evec[ii] * dx
    dN = np.zeros_like(Evec)
    dN[ii]  = 1/dE
    dNold = dN
    dNnew = np.copy(dN)
    nskip = 1000
    plt.plot(Evec, dNold)
    for gen in xrange(0,Ngenmax):
        Ntot = simps( dNnew*Evec, dx=dx,axis=-1 )
        print( gen, Ntot )
        dNold = np.copy(dNnew)
        dNnew = flnew( Evec, dNold, seed, nskip = nskip )
        #pdb.set_trace()
        plt.plot(Evec, dNnew)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-15,1e-4)
        plt.xlim(1e4,Emax)
        plt.draw()
