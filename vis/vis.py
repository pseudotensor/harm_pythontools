import visit_writer
from mread import rfd, grid3d
import numpy as np

def getxyz():
    global r, h, ph
    x = r*np.sin(h)*cos(ph)
    y = r*np.sin(h)*sin(ph)
    z = r*np.cos(h)
    return x,y,z

def prime2cart(V):
    global dxdxp
    Vr = dxdxp[1,1]*V[1]+dxdxp[1,2]*V[2]
    Vh = dxdxp[2,1]*V[1]+dxdxp[2,2]*V[2]
    Vp = V[3]*dxdxp[3,3]
    #
    Vrnorm=Vr
    Vhnorm=Vh*np.abs(r)
    Vpnorm=Vp*np.abs(r*np.sin(h))
    #
    Vznorm=Vrnorm*np.cos(h)-Vhnorm*np.sin(h)
    VRnorm=Vrnorm*np.sin(h)+Vhnorm*np.cos(h)
    Vxnorm=VRnorm*np.cos(ph)-Vpnorm*np.sin(ph)
    Vynorm=VRnorm*np.sin(ph)+Vpnorm*np.cos(ph)
    return([0,Vxnorm,Vynorm,Vznorm])


def writedata(fnameformat="fieldline%04d.vtk",no=0):
    global Bcart, uucart
    fname = fnameformat % no
    Bcart = prime2cart(B)
    uucart = prime2cart(uu)
    x, y, z = getxyz()
    vars = (("ijk",3,1,np.array([ti,tj,tk]).transpose(3,2,1,0).ravel()),
            ("X",3,1,np.array([ti,tj,tk]).transpose(3,2,1,0).ravel()),
            ("V",3,1,np.array([r,h,ph]).transpose(3,2,1,0).ravel()),
            ("xyz",3,1,np.array([x,y,z]).transpose(3,2,1,0).ravel()),
            ("rho",1,1,rho.transpose(2,1,0).ravel()))
            

if __name__ == "__main__":
    return
