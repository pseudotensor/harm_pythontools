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
    return( fgval )

def K( Enew, Eold, seed ):
    K = 4*fg(2*Enew,Eold,seed)*(2*Enew>=seed.Egmin)+fg(Eold-Enew,Eold,seed)
    return( K )

def flnew( Evec, flold, seed ):
    """Expect E and flold defined on a regular log grid"""
    dx = np.log(Evec[1]/Evec[0])
    x = np.log(Evec)
    flnew = np.empty_like(flold)
    for i,E in enumerate(Evec):
        flnew[i] = simps( K(E,Evec,seed)*flold*Evec, dx=dx ) 
    return( flnew )

if __name__ == "__main__":
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
    Emax = 1e9
    Ngrid = 1e4
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
    E0 = 1e8
    ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
    dx = np.log(Evec[1]/Evec[0])
    dE = Evec[ii] * dx
    dN = np.zeros_like(Evec)
    dN[ii]  = 1/dE
    dNold = dN
    dNnew = np.copy(dN)
    for gen in xrange(0,Ngenmax):
        dNold = np.copy(dNnew)
        dNnew = flnew( Evec, dNold, seed )
        #pdb.set_trace()
        plt.plot(Evec, dNnew)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1e-15,1e-4)
        plt.xlim(1e4,1e9)
        plt.draw()
