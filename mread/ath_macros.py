import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
from streamlines import streamplot
from streamlines import fstreamplot
from pychip import pchip_init, pchip_eval
#rc('verbose', level='debug')
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
#add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

#from pylab import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show

import time as time
import gc
import numpy as np
import array
#import scipy as sc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint
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
import operator as op
#import vis
import visit_writer

#global rho, ug, vu, uu, B, CS
#global nx,ny,nz,_dx1,_dx2,_dx3,ti,tj,tk,x1,x2,x3,r,h,ph,gdet,conn,gn3,gv3,ck,dxdxp

#so far only for hydro
def rdath(fname):
    global n1, n2, t, ti, tj, x1, x2, rho, v1, v2, v3, pg
    fin = open( fname , "rb" )
    header1 = fin.readline().split()
    header2 = fin.readline().split()
    header3 = fin.readline().split()
    header4 = fin.readline().split()
    header5 = fin.readline().split()
    n1 = np.float64(header1[3].split("'")[0])
    n2 = np.float64(header3[3].split("'")[0])
    t = np.float64(header5[6].split(",")[0])
    res = np.loadtxt(fname,dtype=np.float64,skiprows=0,unpack=1).reshape((-1,n1,n2),order="F")
    ti, tj, x1, x2, rho, v1, v2, v3, pg = res

def mkathtestmovie(sleepdt=0.5,**kwargs):
    fntsize = kwargs.pop("fontsize",20)
    plt.clf()
    for i in xrange(0,21):
        rdath("jetblob.%04d.tab"%i);
        #plco(np.log10(pg),levels=np.arange(-4,4,0.1),isfilled=1,antialiased=0,cb=1);
        ax = plt.gca()
        # plt.clf()
        p=ax.imshow(np.log10(pg[:,:].transpose()), extent=(0,n1,0,n2), cmap = cm.jet, norm = colors.Normalize(clip = True),origin='lower',interpolation="nearest",vmin=-4,vmax=4,**kwargs)
        if i==0:
            cbar = plt.colorbar(p)
            cbar.ax.set_ylabel(r'$\log_{10}p$',fontsize=fntsize)

        plt.title(r"$t=%g$" % t)
        plt.draw();
        time.sleep(sleepdt)
        plt.savefig("frame%04d.png"%i,bbox_inches='tight',pad_inches=0.02)

