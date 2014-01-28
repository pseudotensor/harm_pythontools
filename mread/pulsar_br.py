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
    for th in [0, 15, 30, 45, 60, 75, 90]:
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
    f = interp1d(th0[which],v["Br2d0"][:,0][which]/norm,bounds_error=0,fill_value=1)
    br0_num_func = f
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
    #Prepare to discretize these solutions on a grid of num^2 cells:
    # create grid
    num = 128
    phgrid = np.linspace(0,2*pi,2*num,endpoint=False)[None,:]
    thgrid = np.linspace(0,pi,num,endpoint=False)[:,None]+0*phgrid
    phgrid = phgrid + 0*thgrid
    # compute cell spacing
    dth = (thgrid[1,0]-thgrid[0,0])
    dph = (phgrid[0,1]-phgrid[0,0])
    # arrays for storing discretized solutions
    br_an_func_list = []
    br_num_func_list = []
    # arrays for storing the values of these solutions at theta = 0 and 90 degrees
    br_an_list = []
    br_num_list = []
    br0_an_list = []
    br0_num_list = []
    br90_an_list = []
    br90_num_list = []
    # well, actually, not 0 and 90 degrees but al1 = 5 and al2 = 87 degrees:
    al1 = 5./180.*pi
    al2 = 87./180.*pi
    v1 = {}
    # fitting functions
    for da in [0, 15, 30, 45, 60, 75, 90]:
        v1["br_an_%g" % da] = br_alpha_an_func(da/180.*pi,v["th2d%g" % da],v["ph2d%g" % da])
        v1["br_num_%g" % da] = br_alpha_num_func(da/180.*pi,v["th2d%g" % da],v["ph2d%g" % da])
    th = alpha
    plt.figure(1)
    plt.clf()
    plt.plot(v["ph2d%g" % th][128/2,:],v["Br2d%g" % th][128/2,:]/(v["psi%g" % th]/v["psi0"])/norm,"r")
    plt.plot(v["ph2d%g" % th][128/2,:],v1["br_an_%g" % th][128/2,:],"g")
    plt.plot(v["ph2d%g" % th][128/2,:],v1["br_num_%g" % th][128/2,:],"b")
    plt.plot(v["ph2d%g" % th][128/2,:],v1["br_an_%g" % th][128/2,:]*sin(th/180.*pi)+v1["br_num_%g" % th][128/2,:]*cos(th/180.*pi),"k")
    plt.figure(2)
    plt.clf()
    th = 0; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"r")
    th = 30; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"g")
    th = 60; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"b")
    th = 90; plt.plot(v["th2d%g" % th][:,0],v["Br2d%g" % th][:,0]/(v["psi%g" % th]/v["psi0"])/norm,"m")
    #
    # Plotting
    #
    # #
    # # Fig 1
    # #
    # plt.figure(1)
    # plt.clf()
    # plt.plot(alphas,brsq0_sol_list/(psis/psis[0])**2,"go-",label=r"$B_r(0)$")
    # plt.plot(alphas,brsq90_sol_list/(psis/psis[0])**2,"bo-",label=r"$B_r(90)$")
    # t = np.linspace(0,90,100)
    # plt.plot(t,cos(t/180.*pi)**2,"k:")
    # plt.plot(t,0.2+1-cos(t/180.*pi)**2,"k:")
    # plt.legend(loc="best")
    # plt.ylim(0,2)
    # plt.xlim(0,90)
    # plt.grid(b=1)
    # ax1=plt.gca()
    # for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    #     label.set_fontsize(20)
    # plt.xlabel(r"$\theta\ {\rm [^\circ]}$",fontsize=20)
    # plt.ylabel(r"$\langle B_r^2\rangle$",fontsize=20)
    # #plt.savefig("Br.pdf",bbox_inches='tight',pad_inches=0.02)
    #
    # Fig 10
    #
    plt.figure(10)
    plt.clf()
    plt.plot(alphas,w_an_list,"go-",label=r"$w_{\rm an}$")
    plt.plot(alphas,w_num_list,"bo-",label=r"$w_{\rm num}$")
    plt.plot(alphas,np.array(w_an_list)+np.array(w_num_list),"ro-",label=r"$w_{\rm an}+w_{\rm num}$")
    plt.legend(loc="best")
    plt.ylim(-4,4)
    plt.xlim(0,90)
    plt.grid(b=1)
    ax1=plt.gca()
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(20)
    plt.xlabel(r"$\theta\ {\rm [^\circ]}$",fontsize=20)
    plt.ylabel(r"$w$",fontsize=20)
    #plt.savefig("Br.pdf",bbox_inches='tight',pad_inches=0.02)
    #
    # Fig 2
    #
    plt.figure(2)
    plt.clf()
    sol = []
    th = np.linspace(0,pi,100)
    for i in xrange(len(thetas)):
        sol.append( (w_an_list[i]*brsq_an_func_list[i](th)**0.5+w_num_list[i]*brsq_num_func_list[i](th)**0.5)**2 )
    plt.plot(v["th0"]*180/np.pi,v["brsqavg0"]/np.max(v["brsqavg0"]),"r")
    plt.plot(th*180./pi,sol[0],"r:",lw=2)
    #plt.plot(th*180./pi,brsq_num_func_list[0](th),"r:",lw=2)
    plt.plot(v["th30"]*180/np.pi,v["brsqavg30"]/np.max(v["brsqavg0"])/(v["psi30"]/v["psi0"])**2,"g")
    plt.plot(th*180./pi,sol[2],"g:",lw=2)
    plt.plot(v["th60"]*180/np.pi,v["brsqavg60"]/np.max(v["brsqavg0"])/(v["psi60"]/v["psi0"])**2,"b")
    plt.plot(th*180./pi,sol[4],"b:",lw=2)
    plt.plot(v["th90"]*180/np.pi,v["brsqavg90"]/np.max(v["brsqavg0"])/(v["psi90"]/v["psi0"])**2,"m")
    #plt.plot(v["th90"]*180/np.pi,brsq_an_func_list[-1](v["th90"]),"m--",lw=2)
    plt.plot(th*180./pi,sol[6],"m:",lw=2)
    plt.ylim(0,2)
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
    v.close()
