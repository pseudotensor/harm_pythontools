import matplotlib
import numpy as np
from numpy import *
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
#add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 
import pdb
from scipy.interpolate import interp1d


def linsolve(a,b):
    """ solves a x = b. returns: x """
    x = float((b[0]*a[1,1]-b[1]*a[0,1])/(a[0,0]*a[1,1]-a[1,0]*a[0,1]))
    y = float((b[0]*a[1,0]-b[1]*a[0,0])/(a[0,1]*a[1,0]-a[1,1]*a[0,0]))
    return(x,y)
    
def plotbrsq(cachefname="psrangle.npz",alpha = 15):
    v = np.load(cachefname)
    psis = np.array([])
    thetas = []
    etots = np.array([])
    alphas = np.array([])
    ebrsqs = np.array([])
    ebrs = np.array([])
    thgrid = [0, 15, 30, 45, 60, 75, 90]
    for th in thgrid:
        psis = np.append(psis, v["psi%g" % th])
        thetas.append(v["th%g" % th])
        etots = np.append(etots, v["etot%g" % th])
        alphas = np.append(alphas, float(th))
        ebrsqs = np.append(ebrsqs, v["ebrsq%g" % th])
        ebrs = np.append(ebrs, v["ebr%g" % th])
    th0 = thetas[0]
    #rotation angle
    psisopsi0_func = interp1d(alphas/180.*pi,psis/psis[0])
    da = 60/180.*np.pi 
    brsqavg0 = v["brsqavg0"]
    br0_an_func_unnorm = interp1d(th0,cos(th0),bounds_error=0,fill_value=1)
    which = (th0<87./180.*pi)+(th0>93./180.*pi)
    norm = np.max(abs(v["Br2d0"]))
    print( "Norm = %g" % norm )
    f = interp1d(th0[which],np.abs(v["Br2d0"])[:,0][which]/norm,bounds_error=0,fill_value=1)
    br0_num_func = lambda th: f(th)*(2*(th<0.5*pi)-1)
    #analytic flux: due to vacuum dipole
    anflux = 0.5*(2*pi*abs(br0_an_func_unnorm(th0))*sin(th0)*(th0[1]-th0[0])).sum(-1)
    #numerical flux: due to axisymmetric MHD dipole
    numflux = 0.5*(2*pi*abs(br0_num_func(th0))*sin(th0)*(th0[1]-th0[0])).sum(-1)
    #now rescale aligned vacuum dipole such that its open flux is the same as that of numerical solution
    br0_an_func = lambda th: br0_an_func_unnorm(th)*(numflux/anflux)
    #old theta in terms of new theta, phi, and the amount of rotation, alpha
    oldth = lambda al,th,ph: arccos(sin(th)*cos(ph)*sin(al)+cos(th)*cos(al))
    #tilt both solutions
    br_alpha_an_func = lambda al,th,ph: br0_an_func(oldth(al,th,ph+0.95))
    br_alpha_num_func = lambda al,th,ph: br0_num_func(oldth(al,th,ph+1.4))
    # compute cell spacing
    dth = (v["th2d30"][1,0]-v["th2d30"][0,0])
    dph = (v["ph2d30"][0,1]-v["ph2d30"][0,0])
    v1 = {}
    # fitting functions
    for da in [0, 15, 30, 45, 60, 75, 90]:
        v1["br_an_%g" % da] = br_alpha_an_func(da/180.*pi,v["th2d%g" % da],v["ph2d%g" % da])
        v1["br_num_%g" % da] = br_alpha_num_func(da/180.*pi,v["th2d%g" % da],v["ph2d%g" % da])
    th = alpha
    # proposed analytic solution
    #w1=interp1d([0,30,60,90],[1,1.05,1.4,1])
    if 1:
        #analytical vacuum dipole for 90-degree solution
        adeg = array([0,30,60,75,90])
        arad = adeg * pi / 180.
        w1=interp1d(adeg,[1,.97,.95,1,1])
        #w1=interp1d(adeg,1+0*adeg)
        w2=interp1d(adeg,[1,0.47,0.55,0.65,1.03])
        #w2=interp1d(adeg,(1-cos(arad))/sin(arad))
        Br_fit = lambda th: v1["br_num_%g" % th] if th == 0 else v1["br_num_%g" % th]*cos(th/180.*pi)**0.5*w1(th)+v1["br_an_%g" % 90]*sin(th/180.*pi)*w2(th)
    else:
        #numerical MHD solution for 90-degree solution
        w1=interp1d([0,30,60,75,90],[1,.97,.95,1,1])
        w2=interp1d([0,30,60,75,90],[1,0.4,0.52,0.63,1])
        Br_fit = lambda th: v1["br_num_%g" % th] if th == 0 else v1["br_num_%g" % th]*cos(th/180.*pi)**0.5*w1(th)+v["Br2d%g" % 90]/(v["psi%g" % th]/v["psi0"])/norm*sin(th/180.*pi)*w2(th)
    Br_mhd_fit = lambda th: v1["br_num_%g" % th]*cos(th/180.*pi)*w1(th)
    Br_vac_fit = lambda th: v1["br_an_%g" % 90]*sin(th/180.*pi)*w2(th)
    Brsqavg_fit = lambda th: (Br_fit(th)**2).mean(-1)
    psi_fit = lambda th: 0.5*(2*pi*sin(v["th2d0"][:,:])*abs(Br_fit(th))*(v["th2d0"][1,0]-v["th2d0"][0,0])).sum() if th == 0 else 0.5*(sin(v["th2d30"])*abs(Br_fit(th))*dth*dph).sum()
    #
    # Plotting
    #
    #
    # Figure 0
    #
    plt.figure(0)
    plt.clf()
    alphagrid = array([0,30,60,75,90])
    alphafinegrid = linspace(0,90,100)
    plot(alphagrid, w1(alphagrid)*cos(alphagrid/180.*pi)**0.5,"o-",label=r"$w_1$")
    plot(alphagrid, w2(alphagrid)*sin(alphagrid/180.*pi),"o-",label=r"$w_2$")
    plot(alphafinegrid, cos(alphafinegrid/180.*pi)**0.5,"-",label=r"$\cos^{1/2}\alpha$")
    plot(alphafinegrid, 1-cos(alphafinegrid/180.*pi)**0.5,"-",label=r"$1-\cos^{1/2}\alpha$")
    plot(alphafinegrid, 1-cos(alphafinegrid/180.*pi),"-",label=r"$1-\cos\alpha$")
    plot(alphafinegrid, sin(alphafinegrid/180.*pi)**2,"-",label=r"$\sin^2\alpha$")
    legend(loc="best")
    #
    # Figure 1
    #
    plt.figure(1)
    plt.clf()
    plt.plot(v["ph2d%g" % th][128/2,:],v["Br2d%g" % th][128/2,:]/(v["psi%g" % th]/v["psi0"])/norm,"r")
    plt.plot(v["ph2d%g" % th][128/2,:],v1["br_an_%g" % th][128/2,:],"g")
    plt.plot(v["ph2d%g" % th][128/2,:],v1["br_num_%g" % th][128/2,:],"b")
    plt.plot(v["ph2d%g" % th][128/2,:],Br_fit(th)[128/2,:],"k")
    #
    # Figure 2
    #
    plt.figure(2)
    plt.clf()
    # th = 0; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"r",lw=2)
    # th = 0; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"r:",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"g",lw=2)
    #th = 30; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"g:",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],Br_mhd_fit(th)[:,0],"g:",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],Br_vac_fit(th)[:,0],"g-.",lw=2)
    plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"g--",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"b",lw=2)
    #th = 60; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"b:",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],Br_mhd_fit(th)[:,0],"b:",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],Br_vac_fit(th)[:,0],"b-.",lw=2)
    plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"b--",lw=2)
    # th = 90; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"m",lw=2)
    # th = 90; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"m:",lw=2)
    #
    # Figure 3
    #
    plt.figure(3)
    plt.clf()
    #
    # th = 0; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"r",lw=2)
    # th = 0; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,0],"r:",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,128/4]/(v["psi%g" % th]/v["psi0"])/norm,"g",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],Br_mhd_fit(th)[:,128/4],"g:",lw=2)
    th = 30; plt.plot(v["th2d%g" % th][:,0],Br_vac_fit(th)[:,128/4],"g-.",lw=2)
    plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,128/4],"g--",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,128/4]/(v["psi%g" % th]/v["psi0"])/norm,"b",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],Br_mhd_fit(th)[:,128/4],"b:",lw=2)
    th = 60; plt.plot(v["th2d%g" % th][:,0],Br_vac_fit(th)[:,128/4],"b-.",lw=2)
    plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,128/4],"b--",lw=2)
    # th = 90; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,128/4]/(v["psi%g" % th]/v["psi0"])/norm,"m",lw=2)
    # th = 90; plt.plot(v["th2d%g" % th][:,0],Br_fit(th)[:,128/4],"m:",lw=2)
    #
    # Fig 4
    #
    plt.figure(4)
    plt.clf()
    colors = ["red", "green", "blue", "magenta", "black"]
    coliter = iter(colors)
    for th in [0, 30, 60, 75, 90]:
        col = next(coliter)
        plt.plot(v["th%g"%th]*180/np.pi,v["brsqavg%g"%th]/np.max(v["brsqavg0"])/(v["psi%g"%th]/v["psi0"])**2,color=col)
        l,=plt.plot(v["th%g"%th]*180/pi,Brsqavg_fit(th),":",color=col,lw=2)
        l.set_dashes([2,2])
    plt.ylim(0,1.5)
    plt.xlim(0,180)
    plt.grid(b=1)
    ax1=plt.gca()
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(20)
    plt.grid(b=1)
    ax1=plt.gca()
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(20)
    plt.xlabel(r"$\theta\ {\rm [^\circ]}$",fontsize=20)
    plt.ylabel(r"$\langle B_r^2\rangle$",fontsize=20)
    #
    # Figure 5
    #
    figure(5)
    clf()
    f1 = (v1["br_num_%g" % 60]**2).mean(-1)
    f2 = (v1["br_an_%g" % 90]**2).mean(-1)
    f = (v1["br_num_%g" % 60]*v1["br_an_%g" % 90]).mean(-1)
    plot(v["th2d60"][:,0],f,label=r"$f$")
    plot(v["th2d60"][:,0],f1,label=r"$f_1$")
    plot(v["th2d60"][:,0],f2,label=r"$f_2$")
    legend(loc="best")
    #
    # Figure 5
    #
    figure(5)
    clf()
    psigrid = []
    for th in thgrid:
        psigrid.append(psi_fit(th))
    plot(thgrid,psigrid)
    #
    v.close()
    
