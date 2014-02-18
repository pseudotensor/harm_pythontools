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

def stagsurf(num_dumps=4,dn=1):
    #os.chdir("/home/atchekho/run/a09new")
    grid3d("gdump.bin",use2d=1)
    color_list = cm.rainbow_r(np.linspace(0, 1, num_dumps))
    plt.clf()
    for i in xrange(num_dumps): 
        rfd("fieldline%04d.bin"%(i+2000))
        plc(radavg(radavg(uu[1][...,0:2*dn+1].mean(-1),dn=dn),axis=1,dn=dn),
            levels=(0,),xy=1,xmax=15,ymax=7.5,
            colors=[tuple(color_list[i])])

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
    Esmin = kwargs.pop("Esmin", 0.0012)
    gammagg = 1.6e7
    Esmax = kwargs.pop("Esmax", 2./gammagg/eV)
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
    Nrad = (np.array(dNdE_rad_list)*np.array(Evec)[None,:]**1*dx).sum(axis=-1)
    Elep = (np.array(dNdE_list)*np.array(Evec)[None,:]**2*dx).sum(axis=-1)
    npzfile.close()
    #########################
    #
    # Closed file
    #
    #########################
    # print( "#%14s %21s %21s %21s" % ("Generation", "N", "deltaN", "E") )
    # print( "%15d %21.15g %21.15g %21.15e" % (gen, Ntot, deltaN, Etot) )
    return({"E0": E0, "gen": np.array(gen_list), "dNdE": dNdE_list, "dNdE_rad": dNdE_rad_list, "deltaN": np.array(deltaN_list), "Ntot": np.array(Ntot_list), "Etot": np.array(Etot_list), "Esmin": Esmin, "Esmax": Esmax, "s": s, "Evec": np.array(Evec), "dx": dx, "Eall": Eall, "Erad": Erad, "Nrad": Nrad, "Elep": Elep})
    
def main(Ngen = 10,resume=None,**kwargs):
    global dNold, dNnew,fout,dNdE_list,Evec
    E0 = kwargs.pop("E0", 1.6e9)  #=gammamaxIC from ~/Cascade.ipnb
    do_enforce_energy_conservation = kwargs.pop("do_enforce_energy_conservation", 0)
    Ngrid = kwargs.pop("Ngrid", 1e4)
    #spectral index
    s = kwargs.pop("s", 2.2)
    #lower/upper cutoffs [eV]
    Esmin = kwargs.pop("Esmin", 0.0012)
    Esmax = kwargs.pop("Esmax", 2./1.3e6/eV)
    #
    doenc = "_enc1" if do_enforce_energy_conservation else ""
    fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g%s.npz" % (E0, Ngrid, s, Esmin, Esmax, doenc)
    #if restart file does not exist, do not restart
    if not os.path.isfile(fnamedefault): 
        if resume:
            print( "File %s does not exist. Nothing to restart from." % fnamedefault )
        resume = 0
    elif resume is None:
        resume = 1
    if resume:
        print( "Restarting from %s" % fnamedefault )
    else:
        print( "Starting fresh." )
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
            do_enforce_energy_conservation_fromfile = npzfile["do_enforce_energy_conservation"]
            assert do_enforce_energy_conservation_fromfile == do_enforce_energy_conservation, \
                    ("Energy conservation in the file, %d, is different from requested value, %d" 
                    % (do_enforce_energy_conservation_fromfile, do_enforce_energy_conservation))
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
        npzfile.close()
    doenc = "_enc1" if do_enforce_energy_conservation else ""
    fnamedefault = "E%.2g_N%.2g_s%g_Esmin%.2g_Esmax%.2g%s.npz" % (E0, Ngrid, s, Esmin, Esmax, doenc)
    print( "Processing %s..." % fnamedefault )
    if do_enforce_energy_conservation:
        print( "Energy conservation is enabled." )
    else:
        print( "Energy conservation is disabled." )
    print( "#%14s %21s %21s %21s %21s" % ("Generation", "N", "deltaN", "E", "deltaE") )
    print( "%15d %21.15g %21.15g %21.15e %21.15e" % (gen, Ntot, deltaN, Etot, deltaE) )
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
    np.savez(fnamedefault, Evec = Evec, E0 = E0, gen_list = gen_list, deltaN_list = deltaN_list, deltaE_list = deltaE_list, dNdE_list = dNdE_list, dNdE_rad_list = dNdE_rad_list, Ntot_list = Ntot_list, Etot_list = Etot_list, Emin = Emin, Emax = Emax, Ngrid = Ngrid, E0grid = E0grid, Esmin = Esmin, Esmax = Esmax, s = s, do_enforce_energy_conservation = do_enforce_energy_conservation)

def plot_convergence(wf = 0,fntsize=18,dosavefig=0,do_enforce_energy_conservation = 0):

    doenc = ""
    snE16e9N1e2 = get_cascade_info(fname="E1.6e+09_N1e+02_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N2e2 = get_cascade_info(fname="E1.6e+09_N2e+02_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N4e2 = get_cascade_info(fname="E1.6e+09_N4e+02_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N1e3 = get_cascade_info(fname="E1.6e+09_N1e+03_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N2e3 = get_cascade_info(fname="E1.6e+09_N2e+03_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N4e3 = get_cascade_info(fname="E1.6e+09_N4e+03_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N1e4     = get_cascade_info(fname="E1.6e+09_N1e+04_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N2e4 = get_cascade_info(fname="E1.6e+09_N2e+04_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    snE16e9N4e4 = get_cascade_info(fname="E1.6e+09_N4e+04_s2.2_Esmin0.0012_Esmax0.064%s.npz" % doenc)
    convergence_list = [snE16e9N1e2, snE16e9N2e2, snE16e9N4e2, 
                  snE16e9N1e3, snE16e9N2e3, snE16e9N4e3,
                  snE16e9N1e4, snE16e9N2e4, snE16e9N4e4]

    #hack for now:
    doenc = ""
    snE1e6     = get_cascade_info(fname="E1e+06_N1e+04_s2.2_Esmin0.0012_Esmax0.79%s.npz" % doenc)
    snE1e7     = get_cascade_info(fname="E1e+07_N1e+04_s2.2_Esmin0.0012_Esmax0.79%s.npz" % doenc)
    snE1e8     = get_cascade_info(fname="E1e+08_N1e+04_s2.2_Esmin0.0012_Esmax0.79%s.npz" % doenc)
    snE1e9     = get_cascade_info(fname="E1.6e+09_N1e+04_s2.2_Esmin0.0012_Esmax0.79%s.npz" % doenc)
    snE1e10    = get_cascade_info(fname="E1e+10_N1e+04_s2.2_Esmin0.0012_Esmax0.79%s.npz" % doenc)

    spectra_list = [snE1e6, snE1e7, snE1e9, snE1e10]
    
    sim_list = [snE1e6, snE1e7, snE1e8, snE1e9, snE1e10]
    dashes_list = [[5,2], [5,2,2,2], [5,2,2,2,2,2], [10,5], [10,2,2,2,5,2,2,2], [10,2,2,2,10,2,2,2]]
    colors_list = ["red", "Orange", "DarkGreen", "magenta", "blue", "black"]
    simname_list = []
    if wf == 0 or wf == 1:
        Ngenmax = 0
        for sim in sim_list:
            Ngenmax = max(Ngenmax,len(sim["Etot"]))
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
        for i,sim in enumerate(sim_list):
            name = simname_list[i]
            # if i == 1: continue
            # if i == 0:
            #     name = r"$E_0 = 10^6, 10^7$"
            plt.text(30, 1.1*sim["Ntot"][-1], name, size = fntsize,va = "bottom", ha="left")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.5, 2e4)
        plt.xlim(1,Ngenmax)
        plt.ylabel(r"$N_{\rm lep}$", fontsize=fntsize)
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
        plt.text(1.5, 0.5e6, simname_list[0], size = fntsize,va = "top", ha="left")
        plt.text(3, 0.7e10, simname_list[-1], size = fntsize,va = "top", ha="left")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1,Ngenmax)
        plt.ylim(0.5e4, 0.95e11)
        #plt.xlabel(r"${\rm Generation}$", fontsize=fntsize)
        plt.ylabel(r"$\langle\gamma_{\rm lep}\rangle\equiv E_{\rm lep}/N_{\rm lep}$", fontsize=fntsize)
        plt.grid()
        # plt.xlabel(r"${\rm N_{\rm gen}+1}$", fontsize=fntsize)
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
        plt.xlim(1,Ngenmax)
        plt.ylim(0.5e4, 0.95e11)
        #plt.xlabel(r"${\rm Generation}$", fontsize=fntsize)
        plt.ylabel(r"$E_{\rm lep},\ E_{\rm tot}$", fontsize=fntsize)
        plt.grid()
        plt.xlabel(r"${\rm N_{\rm gen}+1}$", fontsize=fntsize)
        plt.text(4, 0.5e6, simname_list[0], size = fntsize,va = "top", ha="left")
        plt.text(4, 1.2e10, simname_list[-1], size = fntsize,va = "bottom", ha="left")
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontsize(fntsize)
        #plt.setp( ax3.get_xticklabels(), visible=False)
        labs = ["a", "b", "c"]
        axs = [ax1, ax2, ax3]
        for ax,lab in zip(axs,labs):
            ax.text(ax.get_xlim()[0]*1.2,ax.get_ylim()[1]**0.98*ax.get_ylim()[0]**0.02,
                    r"$({\rm %s})$" % lab,
                    ha="left",va="top",fontsize=fntsize)
        
        if dosavefig:
            plt.savefig("cascade.eps", bbox_inches='tight', pad_inches=0.04)
    if wf == 0 or wf == 2:
        #
        # DEPENDENCE OF NTOT ON E0
        #
        plt.figure(2,figsize=(6,4))
        plt.clf()
        x = []
        y = []
        for sim in sim_list:
            x.append( sim["E0"] )
            y.append( sim["Ntot"][-1] )
        plt.plot(x,y,"-o",lw=2)
        x1 = 10**np.arange(0,12,1)
        y1 = x1/2/1.3e6
        plt.plot(x1,y1,":",lw=2,label=r"$N=E_0/2\epsilon_{\gamma,\rm min}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(0.5,1e4)
        plt.xlim(0.5e6,2e10)
        plt.xlabel(r"$E_0$", fontsize=18)
        plt.ylabel(r"$N_{\rm lep,\infty}$", fontsize=18)
        plt.grid(b=True)
        leg = plt.legend(loc="lower right")
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels()  + leg.get_texts():
            label.set_fontsize(fntsize)
        if dosavefig:
            plt.savefig("NvsE0.eps", bbox_inches='tight', pad_inches=0.04)
    if wf == 0 or wf == 3:
        #
        # CONVERGENCE WITH INCREASING RESOLUTION
        #
        plt.figure(3,figsize=(6,4))
        plt.clf()
        ngen = 100
        resolution=[]
        photoncount=[]
        leptoncount=[]
        energyperlepton=[]
        energyperphoton=[]
        for sim in convergence_list:
            resolution.append(len(sim["dNdE"][0]))
            leptoncount.append(sim["Ntot"][sim["gen"]==ngen])
            photoncount.append(sim["Nrad"][sim["gen"]==ngen])
            energyperlepton.append(sim["Etot"][sim["gen"]==ngen]/sim["Ntot"][sim["gen"]==ngen])
            energyperphoton.append(sim["Erad"][sim["gen"]==ngen]/sim["Nrad"][sim["gen"]==ngen])
        resolution = np.array(resolution)
        photoncount = np.array(photoncount)
        plot(resolution[:-1],np.abs(leptoncount[:-1]-leptoncount[-1])/leptoncount[-1],
             "rs-",label=r"$dN_{\rm lep}/N_{\rm lep}$",lw=2,ms=10)
        plot(resolution[:-1],np.abs(photoncount[:-1]-photoncount[-1])/photoncount[-1],
             "bo-",label=r"$dN_\gamma/N_\gamma$",lw=2,ms=10)
        plot(resolution[:-1],np.abs(energyperlepton[:-1]-energyperlepton[-1])/energyperlepton[-1],
             "rs--",label=r"$dE_{\rm lep}/E_{\rm lep}$",lw=2,ms=10)
        plot(resolution[:-1],np.abs(energyperphoton[:-1]-energyperphoton[-1])/energyperphoton[-1],
             "bo--",label=r"$dE_{\gamma}/E_{\gamma}$",lw=2,ms=10)
        nres = 10**np.linspace(0,10,100)
        plt.plot(nres,3*(nres/1e3)**(-2.),"k:",lw=2)
        plt.plot(nres,0.8e-2*(nres/1e2)**(-1.),"k:",lw=2)
        plt.text(3e4,0.67e-2,r"$\propto N^{-2}$",fontsize=fntsize,va="bottom",ha="left")
        plt.text(1e3,4.5e-4,r"$\propto N^{-1}$",fontsize=fntsize,va="top",ha="right")
        plt.xlim(50,1e6)
        plt.ylim(1e-5,100)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"${\rm Resolution}$", fontsize=18)
        plt.ylabel(r"${\rm Relative\ error\ at\ %gth\ generation}$" % ngen, fontsize=18)
        leg = plt.legend(loc="best",handlelength=3)
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels()  + leg.get_texts():
            label.set_fontsize(fntsize)
        plt.grid(b=True)
        if dosavefig:
            plt.savefig("convergence.eps", bbox_inches='tight', pad_inches=0.04)
    if wf == 0 or wf == 4:
        #
        # SPECTRA
        #
        plt.figure(4,figsize=(12,10))
        plt.clf()
        gs = GridSpec(4, 4)
        gs.update(left=0.08, right=0.97, top=0.91, bottom=0.09, wspace=0.5, hspace=0.08)
        sim_list = spectra_list
        numpanels = 4
        labtext = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for j in xrange(0,numpanels):
            sim = sim_list[j]
            E0 = sim["E0"]
            ax1=plt.subplot(gs[j:j+1,0:numpanels/2])
            ax2=plt.subplot(gs[j:j+1,numpanels/2:numpanels])
            ngen_list = [0, 1,2,3,4,10,20,40,100,200,400,1000,2000,4000]
            lw_list =  np.array(ngen_list)*0+2
            color_list = cm.rainbow_r(np.linspace(0, 1, len(ngen_list)))
            for ngen,lw,color in zip(ngen_list,lw_list,color_list):
                #if generation number, has been computed, plot it
                if ngen in sim["gen"]:
                    ax1.plot((2*sim["Evec"]),0.25*(2*sim["Evec"])**2*sim["dNdE_rad"][ngen],lw=lw,color=np.array(color))
                    ax2.plot(sim["Evec"],sim["Evec"]**2*sim["dNdE"][ngen],lw=lw,color=np.array(color),label=r"$%g$" % ngen)
                    # plt.plot(hr_sim["Evec"],hr_sim["Evec"]*hr_sim["dNdE"][ngen],"b:",lw=lw)
                    # plt.plot(hr_sim["Evec"],hr_sim["Evec"]*hr_sim["dNdE_rad"][ngen],"g:",lw=2)
            ax1.set_xlim(1,5e6)
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
            labno = j
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
                #hide x-tick labels in intermediate panels
                if j < numpanels-1:
                    plt.setp( ax.get_xticklabels(), visible=False)
                ax.set_xscale("log")
                ax.set_yscale("log",subsy=[0])
                ax.grid(b=1)
                ax.set_ylim(1e-6,1e12)
                ax.set_yticks(10.**np.arange(-5,15,5))
                #E0 label in top left corner of panels
                valstr = get_sci_string_form(E0)
                ax.text(ax.get_xlim()[0]*1.4,ax.get_ylim()[1]*1e-1,
                        r"$({\rm %s})\quad E_0=%s$" % (labtext[labno],valstr),
                        ha="left",va="top",fontsize=fntsize)
                labno = labno + numpanels
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
