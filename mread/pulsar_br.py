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
    
def plotbrsq(cachefname="psrangle.npz"):
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
    brsq0_an_func_unnorm = interp1d(th0,cos(th0)**2,bounds_error=0,fill_value=1)
    which = (th0<87./180.*pi)+(th0>93./180.*pi)
    f = interp1d(th0[which],brsqavg0[which]/np.max(brsqavg0),bounds_error=0,fill_value=1)
    brsq0_num_func = f
    #analytic flux: due to vacuum dipole
    anflux = (2*pi*brsq0_an_func_unnorm(th0)**0.5*sin(th0)*(th0[1]-th0[0])).sum(-1)
    #numerical flux: due to axisymmetric MHD dipole
    numflux = (2*pi*brsq0_num_func(th0)**0.5*sin(th0)*(th0[1]-th0[0])).sum(-1)
    #now rescale aligned vacuum dipole such that its open flux is the same as that of numerical solution
    brsq0_an_func = lambda th: brsq0_an_func_unnorm(th)*(numflux/anflux)**2
    #old theta in terms of new theta, phi, and the amount of rotation, alpha
    oldth = lambda al,th,ph: arccos(sin(th)*cos(ph)*sin(al)+cos(th)*cos(al))
    #tilt both solutions
    brsqalpha_an_func = lambda al,th,ph: brsq0_an_func(oldth(al,th,ph))
    brsqalpha_num_func = lambda al,th,ph: brsq0_num_func(oldth(al,th,ph))
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
    brsq_an_func_list = []
    brsq_num_func_list = []
    # arrays for storing the values of these solutions at theta = 0 and 90 degrees
    brsq0_an_list = []
    brsq0_num_list = []
    brsq90_an_list = []
    brsq90_num_list = []
    # well, actually, not 0 and 90 degrees but al1 = 5 and al2 = 87 degrees:
    al1 = 5./180.*pi
    al2 = 87./180.*pi
    # fitting functions
    for da in [0, 15, 30, 45, 60, 75, 90]:
        an = interp1d(thgrid[:,0],brsqalpha_an_func(da/180.*pi,thgrid,phgrid).mean(-1),bounds_error=0)
        num = interp1d(thgrid[:,0],brsqalpha_num_func(da/180.*pi,thgrid,phgrid).mean(-1),bounds_error=0)
        brsq_an_func_list.append(an)
        brsq_num_func_list.append(num)
        brsq0_an_list.append(an(al1))
        brsq0_num_list.append(num(al1))
        brsq90_an_list.append(an(al2))
        brsq90_num_list.append(num(al2))
    # numerical solutions
    brsq0_sol_list = []
    brsq90_sol_list = []
    for th in [0, 15, 30, 45, 60, 75, 90]:
        th_array = v["th%g" % th]
        brsq_sol = v["brsqavg%g" % th]/(v["psi%g"%th]/v["psi0"])**2
        brsq_sol_func = interp1d(th_array,brsq_sol)
        brsq0_sol_list.append(brsq_sol_func(al1)/np.max(v["brsqavg0"]))
        brsq90_sol_list.append(brsq_sol_func(al2)/np.max(v["brsqavg0"]))
    brsq90_sol_list = np.array(brsq90_sol_list)
    brsq0_sol_list = np.array(brsq0_sol_list)
    alphas = np.array(alphas)
    # find optical weights, w_an, w_num
    w_an_list = []
    w_num_list = []
    for i in xrange(len(alphas)):
        an1 = float(brsq0_an_list[i]**0.5)
        an2 = float(brsq90_an_list[i]**0.5)
        num1 = float(brsq0_num_list[i]**0.5)
        num2 = float(brsq90_num_list[i]**0.5)
        rhs1 = float(brsq0_sol_list[i]**0.5)
        rhs2 = float(brsq90_sol_list[i]**0.5)
        w_an, w_num = linsolve(np.array([[an1,num1],[an2,num2]]),np.array([rhs1,rhs2]))
        w_an_list.append(w_an)
        w_num_list.append(w_num)
        # print( "%g*%g + %g*%g = %g =?= %g" % (w_an,an1,w_num,num1,w_an*an1+w_num*num1,rhs1) )
        # print( "%g*%g + %g*%g = %g =?= %g" % (w_an,an2,w_num,num2,w_an*an2+w_num*num2,rhs2) )
        #[0, 15, 30, 45, 60, 75, 90]
    # if desired, override the weights
    # w_an_list = [0,  0,  -0.5,  0,  1,  0,  1]
    # w_num_list= [1,  0,  1.5,  0,  0,  0,  0]
    #w_num_list[4] = 1 - w_an_list[4]
    #
    # Plotting
    #
    #
    # Fig 1
    #
    plt.figure(1)
    plt.clf()
    plt.plot(alphas,brsq0_sol_list/(psis/psis[0])**2,"go-",label=r"$B_r(0)$")
    plt.plot(alphas,brsq90_sol_list/(psis/psis[0])**2,"bo-",label=r"$B_r(90)$")
    t = np.linspace(0,90,100)
    plt.plot(t,cos(t/180.*pi)**2,"k:")
    plt.plot(t,0.2+1-cos(t/180.*pi)**2,"k:")
    plt.legend(loc="best")
    plt.ylim(0,2)
    plt.xlim(0,90)
    plt.grid(b=1)
    ax1=plt.gca()
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(20)
    plt.xlabel(r"$\theta\ {\rm [^\circ]}$",fontsize=20)
    plt.ylabel(r"$\langle B_r^2\rangle$",fontsize=20)
    #plt.savefig("Br.pdf",bbox_inches='tight',pad_inches=0.02)
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
