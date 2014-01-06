import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from streamlines import streamplot
from streamlines import fstreamplot
from pychip import pchip_init, pchip_eval
#rc('verbose', level='debug')
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('mathtext',fontset='cm')
#rc('mathtext',rm='stix')
rc('text', usetex=True)

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

def get_cascade_info(**kwargs):
    E0 = kwargs.pop("E0", 4.25e8)
    Ngrid = kwargs.pop("Ngrid", 1e4)
    s = kwargs.pop("s", 2)
    Esmin = kwargs.pop("Esmin", 0.5e-3)
    Esmax = kwargs.pop("Esmax", 2)
    fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g.npz" % (E0, Ngrid, s, Esmin, Esmax)
    fname = kwargs.pop("fname", fnamedefault)
    #########################
    #
    # Open file
    #
    #########################
    #retrieve saved snapshot
    print("Opening %s ..." % fname)
    npzfile = np.load( fname )
    Emin = npzfile["Emin"]
    Emax = npzfile["Emax"]
    E0grid = npzfile["E0grid"]
    Evec = npzfile["Evec"]
    Ngrid = npzfile["Ngrid"]
    #
    grid = casc.Grid(Emin, Emax, E0grid, Ngrid, di = 0.0)
    ivec = np.arange(len(Evec))
    #
    ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
    dx = grid.get_dx()
    #create an alternate grid with the same number of grid points but shifted by one half
    altgrid = casc.Grid(grid.get_Emin(), grid.get_Emax(), grid.get_E0(), grid.get_Ngrid(), di = 0.5)
    #create an alternate grid with the same number of grid points but shifted by one half
    gen_list = list(npzfile["gen_list"])
    dNdE_list = npzfile["dNdE_list"]
    deltaN_list = list(npzfile["deltaN_list"])
    deltaE_list = list(npzfile["deltaE_list"])
    Ntot_list = list(npzfile["Ntot_list"])
    Etot_list = list(npzfile["Etot_list"])
    E0 = npzfile["E0"]
    if "Esmin" in npzfile:
        Esmin = npzfile["Esmin"]
    if "Esmax" in npzfile:
        Esmax = npzfile["Esmax"]
    if "s" in npzfile:
        s = npzfile["s"]
    npzfile.close()
    #########################
    #
    # Closed file
    #
    #########################
    # print( "#%14s %21s %21s %21s" % ("Generation", "N", "deltaN", "E") )
    # print( "%15d %21.15g %21.15g %21.15e" % (gen, Ntot, deltaN, Etot) )
    return({"E0": E0, "gen": np.array(gen_list), "dNdE": dNdE_list, "deltaN": np.array(deltaN_list), "deltaE": np.array(deltaE_list), "Ntot": np.array(Ntot_list), "Etot": np.array(Etot_list), "Esmin": Esmin, "Esmax": Esmax, "s": s, "Evec": np.array(Evec)})
    
def main(Ngen = 10,resume=0,**kwargs):
    global dNold, dNnew,fout,dNdE_list,Evec
    E0 = kwargs.pop("E0", 4.25e8)  #=gammamaxIC from ~/Cascade.ipnb
    Ngrid = kwargs.pop("Ngrid", 1e4)
    #spectral index
    s = kwargs.pop("s", 2)
    #lower/upper cutoffs [eV]
    Esmin = kwargs.pop("Esmin", 0.5e-3)
    Esmax = kwargs.pop("Esmax", 2)
    #1 eV in units of m_e c^2
    eV = 1/(511.e3)
    #
    if resume == 0:
        seed = casc.SeedPhoton( Esmin*eV, Esmax*eV, s )
        #
        Emin = 1e-6
        Emax = 2*E0
        if Ngrid is None:
            Ngrid = 1e4
        #
        E0grid = 0
        grid = casc.Grid(Emin, Emax, E0grid, Ngrid, di = 0.0)
        Evec = grid.Egrid
        ivec = np.arange(len(Evec))
        #
        ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
        dx = grid.get_dx()
        #create an alternate grid with the same number of grid points but shifted by one half
        altgrid = casc.Grid(grid.get_Emin(), grid.get_Emax(), grid.get_E0(), grid.get_Ngrid(), di = 0.5)
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
        dNold = casc.Func.fromGrid(grid)
        dNold.set_func(dN)
        dNnew = casc.Func.fromGrid(grid)
        dNnew.set_func(dN)
        #no radiation initially
        dNold_rad = casc.Func.fromGrid(grid)
        dNold_rad.set_func(dN*0)
        dNnew_rad = casc.Func.fromGrid(grid)
        dNnew_rad.set_func(dN*0)
        plt.plot(Evec, Evec*dNold.func_vec,'-x')
        #
        gen_list = []
        dNdE_list = []
        dNdE_rad_list = []
        Ntot_list = []
        Etot_list = []
        deltaN_list = []
        deltaE_list = []
        Ntot = np.sum( dNnew.func_vec*Evec*dx,axis=-1 )
        Etot = np.sum( dNnew.func_vec*Evec**2*dx,axis=-1 )
        #print( gen, Ntot, deltaN, Etot )
        deltaN = 0
        deltaE = 0
        #generation number
        gen = 0
        print( "#%14s %21s %21s %21s %21s" % ("Generation", "N", "deltaN", "E", "deltaE") )
        print( "%15d %21.15g %21.15g %21.15e %21.15e" % (gen, Ntot, deltaN, Etot, deltaE) )
        startN = 1
        #initial conditions
        gen_list.append(gen)
        dNdE_list.append(list(dNnew.func_vec))
        dNdE_rad_list.append(list(dNnew_rad.func_vec))
        Ntot_list.append(Ntot)
        Etot_list.append(Etot)
        deltaN_list.append(deltaN)
        deltaE_list.append(deltaE)
    else:
        #restart from last snapshot
        fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g.npz" % (E0, Ngrid, s, Esmin, Esmax)
        npzfile = np.load(fnamedefault)
        Emin = npzfile["Emin"]
        Emax = npzfile["Emax"]
        E0grid = npzfile["E0grid"]
        Evec = npzfile["Evec"]
        Ngrid = npzfile["Ngrid"]
        if "Esmin" in npzfile:
            Esmin = npzfile["Esmin"]
        if "Esmax" in npzfile:
            Esmax = npzfile["Esmax"]
        if "s" in npzfile:
            s = npzfile["s"]
        #
        seed = casc.SeedPhoton( Esmin*eV, Esmax*eV, s )
        #
        grid = casc.Grid(Emin, Emax, E0grid, Ngrid, di = 0.0)
        ivec = np.arange(len(Evec))
        #
        ii = np.round(np.log(E0)/np.log(Emax)*Ngrid)
        dx = grid.get_dx()
        #create an alternate grid with the same number of grid points but shifted by one half
        altgrid = casc.Grid(grid.get_Emin(), grid.get_Emax(), grid.get_E0(), grid.get_Ngrid(), di = 0.5)
        gen_list = list(npzfile["gen_list"])
        dNdE_list = list(npzfile["dNdE_list"])
        dNdE_rad_list = list(npzfile["dNdE_rad_list"])
        Ntot_list = list(npzfile["Ntot_list"])
        Etot_list = list(npzfile["Etot_list"])
        deltaN_list = list(npzfile["deltaN_list"])
        deltaE_list = list(npzfile["deltaE_list"])
        E0 = npzfile["E0"]
        dNnew = casc.Func.fromGrid(grid)
        dNnew.set_func(dNdE_list[-1])
        dNold = casc.Func.fromGrid(grid)
        dNold.set_func(dNnew.func_vec)
        dNold_rad = casc.Func.fromGrid(grid)
        dNold_rad.set_func(dNdE_rad_list[-1].func_vec)
        dNnew_rad = casc.Func.fromGrid(grid)
        dNnew_rad.set_func(dNdE_rad_list[-1].func_vec)
        deltaN = deltaN_list[-1]
        deltaE = deltaE_list[-1]
        gen = gen_list[-1]
        startN = gen_list[-1]+1
        Ntot = Ntot_list[-1]
        Etot = Etot_list[-1]
        print( "#%14s %21s %21s %21s %21s" % ("Generation", "N", "deltaN", "E", "deltaE") )
        print( "%15d %21.15g %21.15g %21.15e %21.15e" % (gen, Ntot, deltaN, Etot, deltaE) )
        npzfile.close()
    plt.xscale("log")
    plt.yscale("log")
    # plt.ylim(1e-15,1e-4)
    plt.ylim(1e-8,1e4)
    plt.xlim(1e4,Emax)
    plt.draw()
    warnings.simplefilter("error")
    try:
        np.seterr(divide='raise')
        #save initial conditions
        for gen in xrange(startN,Ngen+1):
            sys.stdout.flush()
            #save the distribution from last time step
            dNold.set_func( dNnew.func_vec )
            #pdb.set_trace()
            Nreordered = casc.flnew( dNold, dNold_rad, dNnew, dNnew_rad, seed, altgrid )
            #change in number
            deltaN += (Nreordered - Ntot)
            #change in energy
            Eradggic_new = np.sum( (dNnew.func_vec+dNnew_rad.func_vec)*Evec**2*dx,axis=-1 )
            Eradggic_old = np.sum( (dNold.func_vec+dNold_rad.func_vec)*Evec**2*dx,axis=-1 )
            deltaE += Eradggic_new - Eradggic_old
            #pdb.set_trace()
            # #plt.plot(Evec, dNnew, 'x')
            Ntot = np.sum( dNnew.func_vec*Evec*dx,axis=-1 )
            Etot = np.sum( dNnew.func_vec*Evec**2*dx,axis=-1 )
            print( "%15d %21.15g %21.15g %21.15e %21.15e" % (gen, Ntot, deltaN, Etot, deltaE) )
            gen_list.append(gen)
            dNdE_list.append(list(dNnew.func_vec))
            dNdE_rad_list.append(list(dNnew_rad.func_vec))
            Ntot_list.append(Ntot)
            Etot_list.append(Etot)
            deltaN_list.append(deltaN)
            if gen % 10 == 0:
                plt.plot(Evec, Evec*dNnew.func_vec, '-')
                plt.draw()
            # print( gen, Ntot, deltaN, Etot )
            #plt.draw()
    except (KeyboardInterrupt, SystemExit):
        print '\n! Received keyboard interrupt, quitting threads.\n'
    print("Saving results to file...")
    fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g.npz" % (E0, Ngrid, s, Esmin, Esmax)
    np.savez(fnamedefault, Evec = Evec, E0 = E0, gen_list = gen_list, deltaN_list = deltaN_list, deltaE_list = deltaE_list, dNdE_list = dNdE_list, dNdE_rad_list = dNdE_rad_list, Ntot_list = Ntot_list, Etot_list = Etot_list, Emin = Emin, Emax = Emax, Ngrid = Ngrid, E0grid = E0grid, Esmin = Esmin, Esmax = Esmax, s = s)

def plot_convergence(wf = 0,fntsize=18):
    #
    # OLD
    #
    s1Gen, s1N = np.loadtxt("casc_sasha_E0_1e8_di0.5.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    s0Gen, s0N = np.loadtxt("casc_sasha_E0_1e8_di0.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    aGen, aN = np.loadtxt("casc_avery_E0_1e8.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    shGen, shN = np.loadtxt("casc_sasha_E0_1e8_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    shx2Gen, shx2N = np.loadtxt("casc_sasha_E0_1e8_hybrid_N2e4.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh5e8Gen, sh5e8N = np.loadtxt("casc_sasha_E0_5e8_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh1e9Gen, sh1e9N = np.loadtxt("casc_sasha_E0_1e9_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    sh1e10Gen, sh1e10N = np.loadtxt("casc_sasha_E0_1e10_hybrid.txt", dtype = np.float64, usecols = (0, 1), skiprows = 1, unpack = True)
    #
    # NEW
    #
    snE1e6     = get_cascade_info(fname="E1e+06_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    snE1e7     = get_cascade_info(fname="E1e+07_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    snE1e8     = get_cascade_info(fname="E1e+08_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    #snE1e8N5e4 = get_cascade_info(fname="E1e+08_N5e+04_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N1e2 = get_cascade_info(fname="E4.2e+08_N1e+02_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N2e2 = get_cascade_info(fname="E4.2e+08_N2e+02_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N4e2 = get_cascade_info(fname="E4.2e+08_N4e+02_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N1e3 = get_cascade_info(fname="E4.2e+08_N1e+03_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N2e3 = get_cascade_info(fname="E4.2e+08_N2e+03_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N4e3 = get_cascade_info(fname="E4.2e+08_N4e+03_s2_Esmin0.0005_Esmax2.npz")
    snE4e8     = get_cascade_info(fname="E4.2e+08_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N2e4 = get_cascade_info(fname="E4.2e+08_N2e+04_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N4e4 = get_cascade_info(fname="E4.2e+08_N4e+04_s2_Esmin0.0005_Esmax2.npz")
    snE4e8N5e4 = get_cascade_info(fname="E4.2e+08_N5e+04_s2_Esmin0.0005_Esmax2.npz")
    snE4e8list = [snE4e8N1e2, snE4e8N2e2, snE4e8N4e2, 
                  snE4e8N1e3, snE4e8N2e3, snE4e8N4e3,
                  snE4e8,     snE4e8N2e4, snE4e8N4e4]
    snE1e9     = get_cascade_info(fname="E1e+09_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    snE1e10    = get_cascade_info(fname="E1e+10_N1e+04_s2_Esmin0.0005_Esmax2.npz")
    if wf == 0 or wf == 1:
        plt.figure(1)
        plt.clf()
        #
        # LINES
        #
        l1, = plt.plot(1+snE1e6["gen"], snE1e6["Ntot"], color="red",label=r"$E_0 = 10^{6}$", lw = 2)
        l1.set_dashes([5,2])
        l2, = plt.plot(1+snE1e7["gen"], snE1e7["Ntot"], color="Orange", label=r"$E_0 = 10^{7}$", lw = 2)
        l2.set_dashes([5,2,2,2])
        l3, = plt.plot(1+snE1e8["gen"], snE1e8["Ntot"], color="DarkGreen", label=r"$E_0 = 10^{8}$", lw = 2)
        l3.set_dashes([5,2,2,2,2,2])
        l4, = plt.plot(1+snE4e8["gen"], snE4e8["Ntot"], color="magenta", label=r"$E_0 = 4.2\times10^{8}$", lw = 2)
        l4.set_dashes([10,5])
        # l4c, = plt.plot(1+snE4e8N5e4["gen"], snE4e8N5e4["Ntot"], color="LightBlue", label=r"$E_0 = 4.2\times10^{8},\ n = 5\times10^4$", lw = 2)
        # l4c.set_dashes([10,2,5,2])
        l5, = plt.plot(1+snE1e9["gen"], snE1e9["Ntot"], color="blue", label=r"$E_0 = 10^{9}$", lw = 2)
        l5.set_dashes([10,2,2,2,5,2,2,2])
        l6, = plt.plot(1+snE1e10["gen"], snE1e10["Ntot"], color="black", label=r"$E_0 = 10^{10}$", lw = 2)
        l6.set_dashes([10,2,2,2,10,2,2,2])
        # l8, = plt.plot(1+sh1e10Gen, sh1e10N, 'g:', label=r"$E_0 = 10^{10},\ n = 10^4$", lw = 2)
        # l8.set_dashes([5,2,2,2,2,2,2,2,2,2])
        # l7, = plt.plot(1+sh1e9Gen, sh1e9N, 'g:', label=r"$E_0 = 10^9,\ n = 10^4$", lw = 2)
        # l7.set_dashes([10,3,3,3,3,3])
        # l6, = plt.plot(1+sh5e8Gen, sh5e8N, 'r:', label=r"$E_0 = 5\times 10^8,\ n = 10^4$", lw = 2)
        # l6.set_dashes([10,3,3,3,3,3,3,3])
        # # l5, = plt.plot(1+s0Gen, s0N, 'm:', label=r"${\rm Sasha},\ loc=0,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # # l2, = plt.plot(1+aGen, aN, 'g-.', label=r"${\rm Avery},\ loc=0,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # # l2.set_dashes([10,5,5,5])
        # l3, = plt.plot(1+shGen, shN, 'c',label=r"$E_0 = 10^8,\ n = 10^4$", lw = 2)
        # # l4, = plt.plot(1+shx2Gen, shx2N, 'r', label=r"$E_0 = 10^8,\ n = 2\times10^4$", lw = 2)
        # # l1, = plt.plot(1+s1Gen, s1N, 'b--', label=r"${\rm Sasha},\ loc=0.5,\ E_0 = 10^8,\ n = 10^4$", lw = 2)
        # # l1.set_dashes([10,5])
        #
        # LABELS
        #
        plt.text(66, 1.2, r"$E_0\!= 10^6$", size = fntsize,va = "bottom", ha="left")
        plt.text(40*1.25**2, 8, r"$E_0\!= 10^7$", size = fntsize,va = "top", ha="left")
        plt.text(40*1.25, 60, r"$E_0\!= 10^8$", size = fntsize,va = "top", ha="left")
        plt.text(40, 220, r"$E_0\!= 4.2\times 10^8$", size = fntsize, ha="left", va="center")
        plt.text(100, 1200, r"$E_0\!= 10^9$", size = fntsize, ha="right")
        plt.text(13, 4000, r"$E_0\!= 10^{10}$", size = fntsize, ha="right")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1, 10000)
        plt.xlim(1, 1e4)
        plt.xlabel(r"${\rm Generation}$", fontsize=fntsize)
        plt.ylabel(r"$N_{\rm leptons}$", fontsize=fntsize)
        plt.grid()
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fntsize)
        #plt.legend(loc="lower right",handlelength=3,labelspacing=0.15)
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
        plt.grid(b=True)
        plt.legend(loc="lower right")
        plt.savefig("NvsE0.pdf", bbox_inches='tight', pad_inches=0.02)
    if wf == 0 or wf == 3:
        plt.figure(3)
        plt.clf()
        ngen = 200
        resolution=[]
        photoncount=[]
        energyperlepton=[]
        for sim in snE4e8list:
            resolution.append(len(sim["dNdE"][0]))
            photoncount.append(sim["Ntot"][sim["gen"]==ngen])
            energyperlepton.append(sim["Etot"][sim["gen"]==ngen]/sim["Ntot"][sim["gen"]==ngen])
        resolution = np.array(resolution)
        photoncount = np.array(photoncount)
        plot(resolution[:-1],np.abs(photoncount[:-1]-photoncount[-1])/photoncount[-1],"ko-")
        plot(resolution[:-1],np.abs(energyperlepton[:-1]-energyperlepton[-1])/energyperlepton[-1],"bo-.")
        plt.xlim(50,1e5)
        plt.ylim(1e-5,2)
        plt.xscale("log")
        plt.yscale("log")
        
        
    # pdb.set_trace()

if __name__ == "__main__":
    #main()
    print ("Hello")
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
