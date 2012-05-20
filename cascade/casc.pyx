#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: cdivision_warnings=False
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport pow



DTYPE = np.float64
ctypedef np.float_t DTYPE_t

cdef double tiny = 1e-300

def fg_p( Eg not None, Ee not None, SeedPhoton seed not None):
    return fgvec( Eg, Ee, seed )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef np.ndarray[double, ndim=1] fgvec( np.ndarray[double, ndim=1] Eg, np.ndarray[double, ndim=1] Ee, SeedPhoton seed):
    cdef int i
    cdef int dim = Ee.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] Eg1 = np.zeros_like(Ee)
    for i from 0 <= i < dim:
        Eg1[i] = fg( Eg[i], Ee[i], seed )
    return( Eg1 )

cdef inline double fg( double Eg, double Ee, SeedPhoton seed):
    cdef double Ep = Ee-Eg
    cdef double fgval = ( (seed.f(Eg/(2*Ee*Ep))/(2*Ep*Ep)) if (Ep>0 and Ee>0 and Eg>0) else (0) )
    return( fgval )

cdef inline double K1( double Enew, double Eold, SeedPhoton seed ):
    cdef double K = (4*fg(2*Enew,Eold,seed)) if (2*Enew>=seed.Egmin) else (0)
    return( K )

cdef inline double K2( double Enew, double Eold, SeedPhoton seed ):
    cdef double K = fg(Eold-Enew,Eold,seed) if Eold > Enew else 0
    return( K )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef public double* get_data( np.ndarray[double, ndim=1] nparray ):
    return <double *>nparray.data

def flnew( flold not None, flnew not None, seed not None ):
    return flnew_c( flold, flnew, seed )

#@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef double flnew_c( Func flold_func, Func flnew_func, SeedPhoton seed ):
    """Expect E and flold defined on a regular log grid, Evec"""
    cdef int i
    cdef int j
    cdef double a, b, c, d, delta
    cdef Grid grid = flold_func
    cdef double *Evec_data = grid.Egrid_data
    cdef double *flnew_data = flnew_func.func_vec_data
    cdef double *lflnew_data = flnew_func.lfunc_vec_data
    cdef double *flold_data = flold_func.func_vec_data
    cdef int dim1 = flold_func.Ngrid
    cdef double temp1, temp2, temp1sum, temp2sum

    temp1sum = 0
    temp2sum = 0
    for i from 0 <= i < dim1:
        Eenew = Evec_data[i]
        temp1 = 0
        temp2 = 0
        for j from 0 <= j < dim1:
            temp1 += K1(Eenew,Evec_data[j],seed)*flold_data[j]*grid.dEdxgrid_data[j]*grid.dx
            temp2 += K2(Eenew,Eenew+Evec_data[j],seed)*flold_func.fofE(Eenew+Evec_data[j])*grid.dEdxgrid_data[j]*grid.dx
        temp1sum += temp1*grid.dEdxgrid_data[i]*grid.dx
        temp2sum += temp2*grid.dEdxgrid_data[i]*grid.dx
        flnew_func.set_funci_c(i,temp1+temp2)
    return(temp2sum)

###############################
#
#  CLASSES
#
###############################        
    

###############################
#
#  SEED PHOTON
#
###############################        

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
        self.Nprefactor = (1.-s)/(pow(Emax,1-s)-pow(Emin,1-s))

    cpdef int canPairProduce(self, double E):
        return( E > self.Egmin )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef np.ndarray[double, ndim=1] f_vec(self, np.ndarray[double, ndim=1] E):
        cdef int i
        cdef np.ndarray[double, ndim=1] f_out = np.zeros_like(E)
        for i from 0 <= i < E.shape[0]:
            f_out[i] = self.f(E[i])
        return f_out

    cdef double f(self, double E):
        return( self.Nprefactor*E**(-self.s) if (E >= self.Emin and E <= self.Emax) else 0 )

    cpdef double minEg(self, double Eenew, double grid_Emin):
        """ Returns minimum gamma-ray energy """
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double minEg_val
        if bottom > 0:
            minEg_val = 2*self.Emin*Eenew**2/bottom
            return minEg_val if minEg_val > grid_Emin else grid_Emin
        else:
            return grid_Emin

    cpdef double maxEg(self, double Eenew, double grid_Emax):
        """ Returns minimum gamma-ray energy """
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double maxEg_val
        if bottom > 0:
            maxEg_val = 2*self.Emax*Eenew**2/bottom
            return maxEg_val if maxEg_val < grid_Emax else grid_Emax
        else:
            return grid_Emax


    cpdef double minEg1(self, double Eenew, double grid_Emin):
        """ Returns minimum energy electron contributing to Eenew"""
        cdef double minEg_val
        minEg_val = Eenew*(1+sqrt(1+1/(self.Emax*Eenew)))
        return minEg_val if minEg_val > grid_Emin else grid_Emin

    cpdef double maxEg1(self, double Eenew, double grid_Emax):
        """ Returns maximum energy electron contributing to Eenew"""
        cdef double maxEg_val
        maxEg_val = Eenew*(1+sqrt(1+1/(self.Emin*Eenew)))
        return maxEg_val if maxEg_val < grid_Emax else grid_Emax

    cpdef double minEg2(self, double Eenew, double grid_Emin):
        """ Returns minimum energy electron contributing to Eenew"""
        cdef double bottom = 1-2*self.Emin*Eenew
        cdef double minEg_val
        if bottom > 0:
            minEg_val = Eenew/bottom
            return minEg_val if minEg_val > grid_Emin else grid_Emin
        else:
            return grid_Emin

    cpdef double maxEg2(self, double Eenew, double grid_Emax):
        """ Returns maximum energy electron contributing to Eenew"""
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef double maxEg_val
        if bottom > 0:
            maxEg_val = Eenew/bottom
            return maxEg_val if maxEg_val < grid_Emax else grid_Emax
        else:
            return grid_Emax

###############################
#
#  GRID
#
###############################        

cdef public class Grid [object CGrid, type TGrid ]:
    """grid class"""
    cdef  double Emin
    cdef  double Emax
    cdef  double E0
    cdef  double xmin
    cdef  double xmax
    cdef  int Ngrid
    cdef  double dx
    cdef public Egrid
    cdef public xgrid
    cdef public dEdxgrid
    cdef double *xgrid_data
    cdef double *Egrid_data
    cdef double *dEdxgrid_data
    cdef double di

    def __init__(self, double Emin, double Emax, double E0, int Ngrid, double di):
        """ Full constructor: allocates memory and generates the grid """
        self.Ngrid = Ngrid
        self.xgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.Egrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.dEdxgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.set_grid( Emin, Emax, E0, di )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid, grid.di )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 1, 2, 0.5, Ngrid, 0.5 )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef set_grid(self, double Emin, double Emax, double E0, double di ):
        """ Same as Grid() but without reallocation of memory """
        cdef int i
        cdef int dim = self.Ngrid
        self.Emin = Emin
        self.Emax = Emax
        self.E0 = E0
        self.di = di
        self.xmax = log( self.Emax-self.E0 )
        self.xmin = log( self.Emin-self.E0 )
        self.dx = (self.xmax - self.xmin) * 1.0 / dim
        #get direct C pointers to numpy arrays' data fields
        self.xgrid_data = get_data(self.xgrid)
        self.Egrid_data = get_data(self.Egrid)
        self.dEdxgrid_data = get_data(self.dEdxgrid)
        for i from 0 <= i < dim:
            self.xgrid_data[i] = self.xmin + self.dx*(i+self.di)
            self.Egrid_data[i] = self.E0 + exp( self.xgrid_data[i] )
            self.dEdxgrid_data[i] = self.Egrid_data[i] - self.E0

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef double get_dx(self):
        return self.dx



    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef int iofx(self, double xval):
        """ Returns the index of the cell containing xval """
        cdef int ival
        ival = int( (xval-self.xmin)/self.dx - self.di )
        return ival

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double xofE(self, double Eval):
        """ Returns the value of x corresponding to Eval """
        cdef double xval
        xval = log(Eval - self.E0)
        return xval

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef inline int iofE(self, double Eval):
        """ Returns the index of the cell containing Eval """
        return int( (log(Eval-self.E0)-self.xmin)/self.dx - self.di )


###############################
#
#  FUNCTION
#
###############################        

cdef public class Func(Grid)  [object CFunc, type TFunc ]:
    """ Function class derived from Grid class """
    
    cdef public func_vec
    cdef double *func_vec_data
    cdef public lfunc_vec
    cdef double *lfunc_vec_data

    def __init__(self, double Emin, double Emax, double E0, int Ngrid, func_vec = None):
        Grid.__init__(self, Emin, Emax, E0, Ngrid)
        if func_vec is None:
            self.func_vec = np.zeros((self.Ngrid),dtype=DTYPE)+tiny
            self.func_vec_data = get_data(self.func_vec)
            self.lfunc_vec = np.log(self.func_vec)
            self.lfunc_vec_data = get_data(self.lfunc_vec)
        else:
            self.func_vec = np.copy(func_vec)
            self.func_vec_data = get_data(self.func_vec)
            self.lfunc_vec = np.log(self.func_vec)
            self.lfunc_vec_data = get_data(self.lfunc_vec)

    cpdef set_grid(self, double Emin, double Emax, double E0, double di):
        """ Same as Grid() but without reallocation of memory """
        Grid.set_grid( self, Emin, Emax, E0, di )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid, grid.di )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 1, 2, 0.5, Ngrid, 0.5 )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef np.ndarray[double, ndim=1] fofE_vec(self, np.ndarray[double, ndim=1] Eval):
        cdef double *Eval_data = get_data(Eval)
        cdef Einterp = np.zeros_like(Eval)
        cdef double *Einterp_data = get_data(Einterp)
        cdef int len = Eval.shape[0]
        cdef int i
        for i from 0 <= i < len:
            Einterp_data[i] = self.fofE(Eval_data[i])
        return Einterp

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double fofE(self, double Eval):
        """ Linearly interpolates f(E) in log-log """
        cdef int i
        cdef double logfl, logfr, logf, f, invldiff
        cdef double x, dx
        if Eval < self.Egrid_data[0] or Eval > self.Egrid_data[self.Ngrid-1]:
            return 0
        i = int( (log(Eval-self.E0)-self.xmin)/self.dx - self.di )
        #i = self.iofE(Eval)
        #i = Grid.iofE( self, Eval )
        if i < 0 or i >= self.Ngrid-1:
            return 0
        #log-log
        x  = log(Eval-self.E0)
        dx = (x-self.xgrid_data[i])/self.dx
        logfl = self.lfunc_vec_data[i]
        logfr = self.lfunc_vec_data[i+1]
        return exp(logfr * dx + logfl * (1-dx))
        
    def set_func(self, func_vec):
        return self.set_func_c( get_data(func_vec) )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef double set_func_c(self, double *func_vec_data):
        cdef int i
        for i from 0 <= i < self.Ngrid:
            self.func_vec_data[i] = func_vec_data[i] #if func_vec_data[i] > tiny else tiny
            self.lfunc_vec_data[i] = log(func_vec_data[i]+tiny)
        return tiny

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef int set_funci_c(self, int i, double f):
        self.func_vec_data[i] = max(f,tiny)
        self.lfunc_vec_data[i] = log(self.func_vec_data[i])
        return 0


