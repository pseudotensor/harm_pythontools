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
    if "dNdE_rad_list" in npzfile:
        dNdE_rad_list = npzfile["dNdE_rad_list"]
    else:
        print("dNdE_rad_list not defined; setting to 1e-300")
        dNdE_rad_list = dNdE_list*0 + 1e-300
    deltaN_list = list(npzfile["deltaN_list"])
    Ntot_list = list(npzfile["Ntot_list"])
    Etot_list = list(npzfile["Etot_list"])
    E0 = npzfile["E0"]
    if "Esmin" in npzfile:
        Esmin = npzfile["Esmin"]
    if "Esmax" in npzfile:
        Esmax = npzfile["Esmax"]
    if "s" in npzfile:
        s = npzfile["s"]
    Eall = ((np.array(dNdE_list)+np.array(dNdE_rad_list))*np.array(Evec)[None,:]**2*dx).sum(axis=-1)
    Erad = (np.array(dNdE_rad_list)*np.array(Evec)[None,:]**2*dx).sum(axis=-1)
    Elep = (np.array(dNdE_list)*np.array(Evec)[None,:]**2*dx).sum(axis=-1)
    npzfile.close()
    #########################
    #
    # Closed file
    #
    #########################
    # print( "#%14s %21s %21s %21s" % ("Generation", "N", "deltaN", "E") )
    # print( "%15d %21.15g %21.15g %21.15e" % (gen, Ntot, deltaN, Etot) )
    return({"E0": E0, "gen": np.array(gen_list), "dNdE": dNdE_list, "dNdE_rad": dNdE_rad_list, "deltaN": np.array(deltaN_list), "Ntot": np.array(Ntot_list), "Etot": np.array(Etot_list), "Esmin": Esmin, "Esmax": Esmax, "s": s, "Evec": np.array(Evec), "dx": dx, "Eall": Eall, "Erad": Erad, "Elep": Elep})
    
def main(Ngen = 10,resume=0,**kwargs):
    global dNold, dNnew,fout,dNdE_list,Evec
    E0 = kwargs.pop("E0", 4.25e8)  #=gammamaxIC from ~/Cascade.ipnb
    do_enforce_energy_conservation = kwargs.pop("do_enforce_energy_conservation", 0)
    Ngrid = kwargs.pop("Ngrid", 1e4)
    #spectral index
    s = kwargs.pop("s", 2)
    #lower/upper cutoffs [eV]
    Esmin = kwargs.pop("Esmin", 0.5e-3)
    Esmax = kwargs.pop("Esmax", 2)
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
        if "do_enforce_energy_conservation" in npzfile:
            do_enforce_energy_conservation = npzfile["do_enforce_energy_conservation"]
        else:
            do_enforce_energy_conservation = 0
        print
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
        dNold_rad.set_func(dNdE_rad_list[-1])
        dNnew_rad = casc.Func.fromGrid(grid)
        dNnew_rad.set_func(dNdE_rad_list[-1])
        deltaN = deltaN_list[-1]
        deltaE = deltaE_list[-1]
        gen = gen_list[-1]
        startN = gen_list[-1]+1
        Ntot = Ntot_list[-1]
        Etot = Etot_list[-1]
        print( "#%14s %21s %21s %21s %21s" % ("Generation", "N", "deltaN", "E", "deltaE") )
        print( "%15d %21.15g %21.15g %21.15e %21.15e" % (gen, Ntot, deltaN, Etot, deltaE) )
        npzfile.close()
    if do_enforce_energy_conservation:
        print( "Energy conservation is enabled." )
    else:
        print( "Energy conservation is disabled." )
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
            dNold_rad.set_func( dNnew_rad.func_vec )
            #pdb.set_trace()
            Nreordered = casc.flnew( dNold, dNold_rad, dNnew, dNnew_rad, seed, 
                                     grid, altgrid, do_enforce_energy_conservation )
            #change in number
            deltaN += (Nreordered - Ntot)
            #change in energy
            Eradggic_new = np.sum( (dNnew.func_vec+dNnew_rad.func_vec)*Evec**2*dx,axis=-1 )
            Eradggic_old = np.sum( (dNold.func_vec+dNold_rad.func_vec)*Evec**2*dx,axis=-1 )
            deltaE += (Eradggic_new - Eradggic_old)
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
            deltaE_list.append(deltaE)
            if gen % 10 == 0:
                plt.plot(Evec, Evec*dNnew.func_vec, '-')
                plt.draw()
            # print( gen, Ntot, deltaN, Etot )
            #plt.draw()
    except (KeyboardInterrupt, SystemExit):
        print '\n! Received keyboard interrupt, quitting threads.\n'
    print("Saving results to file...")
    fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g.npz" % (E0, Ngrid, s, Esmin, Esmax)
    np.savez(fnamedefault, Evec = Evec, E0 = E0, gen_list = gen_list, deltaN_list = deltaN_list, deltaE_list = deltaE_list, dNdE_list = dNdE_list, dNdE_rad_list = dNdE_rad_list, Ntot_list = Ntot_list, Etot_list = Etot_list, Emin = Emin, Emax = Emax, Ngrid = Ngrid, E0grid = E0grid, Esmin = Esmin, Esmax = Esmax, s = s, do_enforce_energy_conservation = do_enforce_energy_conservation)

def plot_convergence(wf = 0,fntsize=18,dosavefig=0):
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
    sim_list = [snE1e6, snE1e7, snE1e8, snE4e8, snE1e10]
    dashes_list = [[5,2], [5,2,2,2], [5,2,2,2,2,2], [10,5], [10,2,2,2,5,2,2,2], [10,2,2,2,10,2,2,2]]
    colors_list = ["red", "Orange", "DarkGreen", "magenta", "blue", "black"]
    simname_list = []
    if wf == 0 or wf == 1:
        plt.figure(1,figsize=(6,9))
        plt.clf()
        gs = GridSpec(3, 3)
        gs.update(left=0.15, right=0.97, top=0.97, bottom=0.08, wspace=0.04, hspace=0.04)
        ax1=plt.subplot(gs[0,:])
        #
        # LINES
        #
        for sim,dash,color in zip(sim_list,dashes_list,colors_list):
            l, = plt.plot(1+sim["gen"], sim["Ntot"], color=color, lw = 2)
            l.set_dashes(dash)
            valstr = get_sci_string_form(sim["E0"])
            simname_list.append(r"$E_0 = %s$" % valstr)
        #
        # LABELS
        #
        plt.text(500, 1.2, simname_list[0], size = fntsize,va = "bottom", ha="left")
        plt.text(40*1.25**2, 8, simname_list[1], size = fntsize,va = "top", ha="left")
        plt.text(40*1.25, 60, simname_list[2], size = fntsize,va = "top", ha="left")
        plt.text(40, 1100, simname_list[3], size = fntsize, ha="left", va="center")
        # plt.text(100, 1200, r"$E_0\!= 10^9$", size = fntsize, ha="right")
        plt.text(13, 4000, simname_list[4], size = fntsize, ha="right")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.5, 20000)
        plt.xlim(1, 1e4)
        plt.ylabel(r"$N_{e^\pm}$", fontsize=fntsize)
        plt.grid()
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(fntsize)
        plt.setp( ax1.get_xticklabels(), visible=False)
        #
        # AVERAGE ENERGY PER PARTICLE
        #
        ax2=plt.subplot(gs[1,:])
        #
        # LINES
        #
        for sim,dash,color in zip(sim_list,dashes_list,colors_list):
            l, = plt.plot(1+sim["gen"], sim["Etot"]/sim["Ntot"], color=color, lw = 2)
            l.set_dashes(dash)
            # l,=plt.plot(1+sim["gen"], sim["Eall"], ":", color="gray", lw = 2)
            # l.set_dashes([2,2])
        plt.text(1.5, 0.7e6, simname_list[0], size = fntsize,va = "top", ha="left")
        plt.text(3, 0.7e10, simname_list[-1], size = fntsize,va = "top", ha="left")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1, 1e4)
        plt.ylim(0.5e4, 2e10)
        plt.xlabel(r"${\rm Generation}$", fontsize=fntsize)
        plt.ylabel(r"$\langle E_{e^\pm}\rangle \equiv \langle\gamma_{e^\pm}\rangle$", fontsize=fntsize)
        plt.grid()
        plt.xlabel(r"${\rm N_{\rm gen}+1}$", fontsize=fntsize)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(fntsize)
        plt.setp( ax2.get_xticklabels(), visible=False)
        #
        ax3=plt.subplot(gs[2,:])
        #
        # LINES
        #
        for sim,dash,color in zip(sim_list,dashes_list,colors_list):
            l, = plt.plot(1+sim["gen"], sim["Elep"], color=color, lw = 2)
            l.set_dashes(dash)
            # l, = plt.plot(1+sim["gen"], sim["Erad"], color=color, lw = 1)
            # l.set_dashes(dash)
            l,=plt.plot(1+sim["gen"], sim["Eall"], color=color, lw = 0.5)
            #l.set_dashes([2,2])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1, 1e4)
        plt.ylim(0.5e4, 2e10)
        plt.xlabel(r"${\rm Generation}$", fontsize=fntsize)
        plt.ylabel(r"$E_{e^\pm},\ E_{\rm tot}$", fontsize=fntsize)
        plt.grid()
        plt.xlabel(r"${\rm N_{\rm gen}+1}$", fontsize=fntsize)
        plt.text(1.5, 0.7e6, simname_list[0], size = fntsize,va = "top", ha="left")
        plt.text(1.5, 0.7e10, simname_list[-1], size = fntsize,va = "top", ha="left")
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontsize(fntsize)
        plt.setp( ax3.get_xticklabels(), visible=False)
        if dosavefig:
            plt.savefig("cascade.eps", bbox_inches='tight', pad_inches=0.04)
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
        if dosavefig:
            plt.savefig("NvsE0.pdf", bbox_inches='tight', pad_inches=0.04)
    if wf == 0 or wf == 3:
        #
        # CONVERGENCE WITH INCREASING RESOLUTION
        #
        plt.figure(3)
        plt.clf()
        ngen = 310
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
    if wf == 0 or wf == 4:
        plt.figure(4,figsize=(12,10))
        plt.clf()
        gs = GridSpec(4, 4)
        gs.update(left=0.08, right=0.97, top=0.91, bottom=0.09, wspace=0.5, hspace=0.08)
        sim_list = [snE1e6, snE1e7,snE4e8,snE1e10]
        numpanels = 4
        for j in xrange(0,numpanels):
            sim = sim_list[j]
            E0 = sim["E0"]
            ax1=plt.subplot(gs[j:j+1,0:numpanels/2])
            ax2=plt.subplot(gs[j:j+1,numpanels/2:numpanels])
            ngen_list = [0, 1,2,3,4,10,20,40,100,200,400,1000,2000]
            lw_list =  np.array(ngen_list)*0+2
            color_list = cm.rainbow_r(np.linspace(0, 1, len(ngen_list)))
            for ngen,lw,color in zip(ngen_list,lw_list,color_list):
                #if generation number, has been computed, plot it
                if ngen in sim["gen"]:
                    ax1.plot((2*sim["Evec"]),0.25*(2*sim["Evec"])**2*sim["dNdE_rad"][ngen],lw=lw,color=np.array(color))
                    ax2.plot(sim["Evec"],sim["Evec"]**2*sim["dNdE"][ngen],lw=lw,color=np.array(color),label=r"$%g$" % ngen)
                    # plt.plot(hr_sim["Evec"],hr_sim["Evec"]*hr_sim["dNdE"][ngen],"b:",lw=lw)
                    # plt.plot(hr_sim["Evec"],hr_sim["Evec"]*hr_sim["dNdE_rad"][ngen],"g:",lw=2)
            ax1.set_xlim(1,1e6)
            ax1.set_ylim(1e-6,1e12)
            ax1.set_yticks(10.**np.arange(-5,15,5))
            ax1.set_ylabel(r"$E_\gamma^2dN/dE_\gamma$",fontsize=fntsize,labelpad=-7)
            ax2.set_ylabel(r"$E_{e^\pm}^2dN/dE_{e^\pm}$",fontsize=fntsize,labelpad=-7)
            ax2.set_xlim(1e3,2e10)
            if j == 0:
                leg=ax2.legend(loc="upper right",title=r"${\rm Generation}\!:$",frameon=True, fancybox=True,ncol=3,labelspacing=0.05,columnspacing=0.5,handletextpad=0.04)
                leg.get_title().set_fontsize(0.8*fntsize)
                for label in leg.get_texts():
                    label.set_fontsize(0.8*fntsize)
            for ax in [ax1, ax2]:
                #x-axis on top
                if j == 0:
                    axt = ax.twiny()
                    axt.set_xscale("log")
                    axt.set_xlim(np.array(ax.get_xlim())/eV/1.e9) #get it in GeV
                    if ax == ax1:
                        axt.set_xlabel(r"$E_\gamma\ {\rm [GeV]}$",fontsize=fntsize)
                    if ax == ax2:
                        axt.set_xlabel(r"$E_{e^\pm}\ {\rm [GeV]}$",fontsize=fntsize)
                    for label in axt.get_xticklabels() + axt.get_yticklabels():
                        label.set_fontsize(fntsize)
                #hide tick labels in intermediate panels
                if j < numpanels-1:
                    plt.setp( ax.get_xticklabels(), visible=False)
                ax.set_xscale("log")
                ax.set_yscale("log",subsy=[0])
                ax.grid(b=1)
                ax.set_ylim(1e-6,1e12)
                ax.set_yticks(10.**np.arange(-5,15,5))
                #E0 label in top left corner of panels
                valstr = get_sci_string_form(E0)
                ax.text(ax.get_xlim()[0]*1.1,ax.get_ylim()[1]*1e-1,r"$E_0=%s$" % valstr,ha="left",va="top",fontsize=fntsize)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(fntsize)
        ax1.set_xlabel(r"$E_\gamma\ [m_e c^2]$", fontsize=fntsize)
        ax2.set_xlabel(r"$E_{e^\pm}\ [m_e c^2]\equiv \gamma_{e^\pm}$", fontsize=fntsize)
        if dosavefig:
            plt.savefig("dNdE.eps", bbox_inches='tight', pad_inches=0.04)

def get_sci_string_form(E0):
    ex = 1.*int(np.log10(E0))
    ma = E0/10**ex
    if ma == 1:
        valstr = "10^{%g}" % ex
    else:
        valstr = "%g\\times 10^{%g}" % (int(ma*10.+0.5)/10.,ex)
    return(valstr)

        
    # pdb.set_trace()

if __name__ == "__main__":
    global eV
    #main()
    print ("Hello")
    #energy grid, Lorentz factor of initial electron
    warnings.simplefilter("error")
    #1 eV in units of m_e c^2
    eV = 1/(511.e3)
