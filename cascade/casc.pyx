import numpy as np
#cimport numpy as np

#DTYPE = np.float
#ctypedef np.float_t DTYPE_t

class SeedPhoton:
    """our seed photon class"""
    def __init__(self,Emin,Emax,s):
        self.Emin = Emin
        self.Emax = Emax
        self.s = s
        #minimum energy gamma-ray to be able to pair produce
        self.Egmin = 2/Emax
        self.Nprefactor = (1-s)/(Emax**(1-s)-Emin**(1-s))

    def canPairProduce(self,E):
        return( E > self.Egmin )

    def f(self,E):
        return( self.Nprefactor*E**(-self.s)*(E >= self.Emin)*(E <= self.Emax) )

def fmagic( E ):
    return( E*(E>0)+1*(E<=0) )

def fg( Eg, Ee, seed):
    Eseed = Eg/(2*Ee*(fmagic(Ee-Eg)))
    fEseed = seed.f(fmagic(Eseed))
    fgval = fEseed / (2*(fmagic(Ee-Eg))**2)
    fgval *= (Ee-Eg>0)*(Eseed>0)
    # del Eseed
    # del fEseed
    # gc.collect()
    return( fgval )

def K( Enew, Eold, seed ):
    K = 4*fg(2*Enew,Eold,seed)*(2*Enew>=seed.Egmin)+fg(Eold-Enew,Eold,seed)
    return( K )

def flnew( Evec, flold, seed, nskip = 1 ):
    """Expect E and flold defined on a regular log grid, Evec"""
    dx = np.log(Evec[1]/Evec[0])
    x = np.log(Evec)
    flnew = np.empty_like(flold)
    # for i in xrange(0,int(len(Evec)/nskip)):
    #     flnew[i*nskip:(i+1)*nskip] = integr( K(Evec[i*nskip:(i+1)*nskip,None],Evec[None,:],seed)*(flold*Evec)[None,:], dx=dx,axis=-1 )  
    for i in xrange(0,len(Evec)):
        flnew[i] = integr( K(Evec[i],Evec,seed)*(flold*Evec), dx=dx,axis=-1 )          # gc.collect()
    return( flnew )

def integr( f, dx=1, axis=-1 ):
    return dx*f.sum(axis=axis)
