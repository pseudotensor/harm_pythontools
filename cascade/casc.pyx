from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

# cdef extern from "numpy/arrayobject.h":
#     ctypedef int intp
#     ctypedef extern class numpy.ndarray [object PyArrayObject]:
#         cdef char *data
#         cdef int nd
#         cdef intp *dimensions
#         cdef intp *strides
#         cdef int flags

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef public class SeedPhoton [object CSeedPhoton, type TSeedPhoton ]:
    """our seed photon class"""
    cdef public double Emin
    cdef public double Emax
    cdef public double s
    cdef public double Egmin
    cdef public double Nprefactor

    def __init__(self, double Emin, double Emax, double s):
        self.Emin = Emin
        self.Emax = Emax
        self.s = s
        #minimum energy gamma-ray to be able to pair produce
        self.Egmin = 2./Emax
        self.Nprefactor = (1.-s)/(Emax**(1.-s)-Emin**(1.-s))

    cpdef int canPairProduce(self, double E):
        return( E > self.Egmin )

    cpdef double f(self, double E):
        return( self.Nprefactor*E**(-self.s) if (E >= self.Emin and E <= self.Emax) else 0 )

cdef double fg( double Eg, double Ee, SeedPhoton seed):
    cdef double Ep = Ee-Eg
    cdef double fgval = ( (seed.f(Eg/(2*Ee*Ep))/(2*Ep**2)) if (Ep>0 and Ee>0 and Eg>0) else (0) )
    return( fgval )

cdef double K( double Enew, double Eold, SeedPhoton seed ):
    cdef double K = ( (4*fg(2*Enew,Eold,seed)) if (2*Enew>=seed.Egmin) else (0) ) + fg(Eold-Enew,Eold,seed)
    return( K )


def flnew( Evec not None, flold not None, seed not None ):
    return flnew_c( Evec, flold, seed )

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef public np.ndarray[double, ndim=1] flnew_c( np.ndarray[double, ndim=1] Evec, np.ndarray[double, ndim=1] flold, SeedPhoton seed ):
    """Expect E and flold defined on a regular log grid, Evec"""
    cdef double dx = log(Evec[1]/Evec[0])
    cdef np.ndarray[DTYPE_t, ndim=1] flnew = np.zeros_like(flold)
    cdef int i
    cdef int j
    cdef double *flnew_data = <double *>flnew.data
    cdef double *Evec_data = <double *>Evec.data
    cdef double *flold_data = <double *>flold.data
    cdef int dim = flnew.shape[0]

    # for i in xrange(0,int(len(Evec)/nskip)):
    #     flnew[i*nskip:(i+1)*nskip] = integr( K(Evec[i*nskip:(i+1)*nskip,None],Evec[None,:],seed)*(flold*Evec)[None,:], dx=dx,axis=-1 )  
    for i from 0 <= i < dim:
        #flnew_data[i] = 0
        for j from 0 <= j < dim:
            flnew_data[i] += K(Evec_data[i],Evec_data[j],seed)*(flold_data[j]*Evec_data[j])
        flnew_data[i] *= dx
    return( flnew )