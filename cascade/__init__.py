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
import warnings

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

def main(Ngen = 10,startN=1,rf=1):
    global dNold, dNnew,fout
    #
    E0 = 0.1*1e8
    ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
    dx = grid.get_dx()
    altgrid = casc.Grid(grid.get_Emin(), grid.get_Emax(), grid.get_E0(), grid.get_Ngrid()*rf, di = 0.5)
    if False:
        dE = Evec[ii] * dx
        dN = np.zeros_like(Evec)
        dN[ii]  = 1/dE
    elif False:
        sigmaE = E0/100 #1*grid.dx*E0
        dN = (2*np.pi)**(-0.5)*exp(-0.5*((Evec-E0)/sigmaE)**2)/sigmaE
    else: #Avery's method
        fEw = 0.01 #1*grid.dx*E0
        dN = np.exp(-0.5*((np.log10(Evec)-np.log10(E0))/fEw)**2)
        dN /= (dN.sum()*Evec*dx)
    if startN == 1:
        dNold = casc.Func.fromGrid(grid)
        dNold.set_func(dN)
        dNnew = casc.Func.fromGrid(grid)
        dNnew.set_func(dN)
        plt.plot(Evec, Evec*dNold.func_vec,'-x')
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-15,1e-4)
    plt.ylim(1e-8,1e4)
    plt.xlim(1e4,Emax)
    plt.draw()
    #generation number
    gen = 0
    #error in evolution of electron number
    deltaN = 0
    warnings.simplefilter("error")
    if startN == 1:
        Ntot = np.sum( dNnew.func_vec*Evec*dx,axis=-1 )
        print( gen, Ntot, deltaN )
    np.seterr(divide='raise')
    for gen in xrange(startN,Ngen+1):
        sys.stdout.flush()
        dNold.set_func( dNnew.func_vec )
        #pdb.set_trace()
        Nreordered = casc.flnew( dNold, dNnew, seed, altgrid )
        deltaN += (Nreordered - Ntot)
        #pdb.set_trace()
        plt.plot(Evec, Evec*dNnew.func_vec, '-')
        # #plt.plot(Evec, dNnew, 'x')
        Ntot = np.sum( dNnew.func_vec*Evec*dx,axis=-1 )
        print( gen, Ntot, deltaN )
        plt.draw()

def plot_convergence(wf = 0):
    s1Gen, s1N = np.loadtxt("casc_sasha_E0_1e8_di0.5.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    s0Gen, s0N = np.loadtxt("casc_sasha_E0_1e8_di0.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    aGen, aN = np.loadtxt("casc_avery_E0_1e8.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    shGen, shN = np.loadtxt("casc_sasha_E0_1e8_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    shx2Gen, shx2N = np.loadtxt("casc_sasha_E0_1e8_hybrid_N2e4.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh5e8Gen, sh5e8N = np.loadtxt("casc_sasha_E0_5e8_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh1e9Gen, sh1e9N = np.loadtxt("casc_sasha_E0_1e9_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh1e10Gen, sh1e10N = np.loadtxt("casc_sasha_E0_1e10_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    if wf == 0 or wf == 1:
        plt.figure(1)
        plt.clf()
        l8, = plt.plot(1+sh1e10Gen, sh1e10N, 'g:', label=r"${\rm Sasha},\ {\rm hybrid},\ E_0 = 10^{10},\ n = 10^4$", lw = 2)
        l8.set_dashes([5,2,2,2,2,2,2,2,2,2])
        l7, = plt.plot(1+sh1e9Gen, sh1e9N, 'g:', label=r"${\rm Sasha},\ {\rm hybrid},\ E_0 = 10^9,\ n = 10^4$", lw = 2)
        l7.set_dashes([10,3,3,3,3,3])
        l6, = plt.plot(1+sh5e8Gen, sh5e8N, 'r:', label=r"${\rm Sasha},\ {\rm hybrid},\ E_0 = 5\times 10^8,\ n = 10^4$", lw = 2)
        l6.set_dashes([10,3,3,3,3,3,3,3])
        # l5, = plt.plot(1+s0Gen, s0N, 'm:', label=r"${\rm Sasha},\ loc=0,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # l2, = plt.plot(1+aGen, aN, 'g-.', label=r"${\rm Avery},\ loc=0,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # l2.set_dashes([10,5,5,5])
        l3, = plt.plot(1+shGen, shN, 'c',label=r"${\rm Sasha},\ {\rm hybrid},\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # l4, = plt.plot(1+shx2Gen, shx2N, 'r', label=r"${\rm Sasha},\ {\rm hybrid},\ E_0 = 10^8,\ n = 2\times10^4$", lw = 2)
        # l1, = plt.plot(1+s1Gen, s1N, 'b--', label=r"${\rm Sasha},\ loc=0.5,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # l1.set_dashes([10,5])
        plt.text(30, 100, r"$E_0\!= 10^8$", size = 18)
        plt.text(120, 250, r"$E_0\!= 5\times 10^8$", size = 18, ha="left")
        plt.text(100, 1200, r"$E_0\!= 10^9$", size = 18, ha="right")
        plt.text(13, 4000, r"$E_0\!= 10^{10}$", size = 18, ha="right")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1, 10000)
        plt.xlabel(r"${\rm Generation}$", fontsize=18)
        plt.ylabel(r"$N_{\rm leptons}$", fontsize=18)
        plt.grid()
        plt.legend(loc="lower right",handlelength=3,labelspacing=0.15)
        plt.savefig("cascade.pdf", bbox_inches='tight', pad_inches=0.02)
    if wf == 0 or wf == 2:
        plt.figure(2)
        plt.clf()
        x = np.array((1e8,5e8,1e9,1e10))
        y = np.array((shN[-1],sh5e8N[-1],sh1e9N[-1],sh1e10N[-1]),dtype=np.float64)
        plt.plot(x,y,"-o",lw=2)
        x1 = 10**np.arange(0,12,1)
        y1 = x1 / 1e6
        plt.plot(x1,y1,":",lw=2,label=r"$N=10^{-3}E_0$")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(50,2e4)
        plt.xlim(0.5e8,2e10)
        plt.xlabel(r"$E_0$", fontsize=18)
        plt.ylabel(r"$N_{\rm leptons,\infty}$", fontsize=18)
        plt.grid(b=False)
        plt.legend(loc="lower right")
        plt.savefig("NvsE0.pdf", bbox_inches='tight', pad_inches=0.02)

        
    # pdb.set_trace()

if __name__ == "__main__":
    #main()
    print ("Hello")
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
    Emin = 1e-5
    Emax = 1e8
    Ngrid = 1e4
    # Evec = exp(np.linspace(-5,np.log(Emax),Ngrid))
    E0grid = 0
    grid = casc.Grid(Emin, Emax, E0grid, Ngrid, di = 0.0)
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
