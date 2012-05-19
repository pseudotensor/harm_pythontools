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

import casc as casc
reload(casc)

def test_fg( Eold, Enew, seed ):
    Egmin = 2*seed.Emin*Enew**2 / (1.-2*seed.Emin*Enew)
    Egmax = 2*seed.Emax*Enew**2 / (1.-2*seed.Emax*Enew)
    if np.float(Egmax) < 0:
        Egmax = Eold[-1]
    plt.plot(Eold, casc.fg_p(Eold-Enew, Eold, seed))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.5*np.float(Egmin),2*np.float(Egmax))

#res  = test_fg1(Evec,1e6+0*Evec,seed)
def test_fg1( Eold, Enew, seed ):
    res = 4*casc.fg_p(2*Enew, Eold, seed)*(2*Enew-seed.Egmin>=0)
    #pdb.set_trace()
    plt.plot(Eold, res)
    plt.xscale("log")
    plt.yscale("log")
    return res
    #plt.plot(Evec,(casc.fg_p(2*Evec,1e8+0*Evec,seed)*(2*Evec>=seed.Egmin)))

def main(dim2=100):
    global dNold, dNnew,fout
    #
    Ngenmax = 300
    #
    E0 = 1e8
    ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
    dx = grid.dx
    if False:
        dE = Evec[ii] * dx
        dN = np.zeros_like(Evec)
        dN[ii]  = 1/dE
    else:
        sigmaE = E0/100 #1*grid.dx*E0
        dN = (2*np.pi)**(-0.5)*exp(-0.5*((Evec-E0)/sigmaE)**2)/sigmaE
    dNold = dN
    dNnew = np.copy(dN)
    nskip = 1
    plt.plot(Evec, Evec*dNold,'-x')
    plt.draw()
    fout = casc.Func.empty(dim2)
    for gen in xrange(0,Ngenmax):
        Ntot = np.sum( dNnew*Evec*dx,axis=-1 )
        print( gen, Ntot )
        sys.stdout.flush()
        dNold = dNnew
        #pdb.set_trace()
        dNnew = casc.flnew( grid, dNold, seed, fout, dim2=dim2 )
        #pdb.set_trace()
        plt.plot(Evec, Evec*dNnew, '-')
        #plt.plot(Evec, dNnew, 'x')
        plt.xscale("log")
        plt.yscale("log")
        # plt.ylim(1e-15,1e-4)
        plt.ylim(1e-8,1e2)
        plt.xlim(1e4,Emax)
        plt.draw()


if __name__ == "__main__":
    #main()
    print ("Hello")
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
    Emin = 1e-4
    Emax = 1e10
    Ngrid = 1e4
    # Evec = exp(np.linspace(-5,np.log(Emax),Ngrid))
    E0grid = 0
    grid = casc.Grid(Emin, Emax, E0grid, Ngrid)
    Evec = grid.Egrid
    ivec = np.arange(len(Evec))
    #1 eV in units of m_e c^2
    eV = 1/(511.e3)
    #spectral index
    s = 2
    #lower cutoff
    Esmin = 0.5e-3 * eV
    Esmax = 2 * eV
    seed = casc.SeedPhoton( Esmin, Esmax, s )
