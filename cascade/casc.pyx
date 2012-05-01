from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log
from libc.math cimport exp


DTYPE = np.float
ctypedef np.float_t DTYPE_t


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

cdef double fg( double Eg, double Ee, SeedPhoton seed):
    cdef double Ep = Ee-Eg
    cdef double fgval = ( (seed.f(Eg/(2*Ee*Ep))/(2*Ep**2)) if (Ep>0 and Ee>0 and Eg>0) else (0) )
    return( fgval )

cdef double K1( double Enew, double Eold, SeedPhoton seed ):
    cdef double K = (4*fg(2*Enew,Eold,seed)) if (2*Enew>=seed.Egmin) else (0)
    return( K )

cdef double K2( double Enew, double Eold, SeedPhoton seed ):
    cdef double K = fg(Eold-Enew,Eold,seed)
    return( K )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef public double* get_data( np.ndarray[double, ndim=1] nparray ):
    return <double *>nparray.data

def flnew( Evec not None, flold not None, seed not None ):
    return flnew_c( Evec, flold, seed )

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef public np.ndarray[double, ndim=1] flnew_c( Grid grid, np.ndarray[double, ndim=1] flold, SeedPhoton seed ):
    """Expect E and flold defined on a regular log grid, Evec"""
    cdef int i
    cdef int j
    cdef double a, b, c, d, delta
    cdef np.ndarray[double, ndim=1] flnew = np.zeros_like(flold)
    cdef double *flnew_data = get_data(flnew)
    cdef double *Evec_data = get_data(grid.Egrid)
    cdef double *Evec2_data
    cdef double *flold_data = get_data(flold)
    cdef int dim1 = flnew.shape[0]
    cdef double minEg, maxEg
    #use old grid as a start
    cdef int dim2 = 10000
    #cdef Grid grid2 = Grid.empty(dim2)
    cdef Grid grid2 = Grid.fromGrid(grid)
    cdef Func flold_func = Func.fromGrid(grid)
    flold_func.set_func_c(flold_data)

    for i from 0 <= i < dim1:
        Eenew = Evec_data[i]
        #new grid defined by Eenew
        minEg = seed.minEg(Eenew,grid.Emin)
        maxEg = seed.maxEg(Eenew,grid.Emax)
        if maxEg < grid.Emin or minEg > grid.Emax:
           continue
        grid2.set_grid(minEg,2*maxEg,0.9*minEg)
        #print i, grid2.Emin, grid2.Emax, grid2.E0
        #print i, grid2.Emin, grid2.Emax, grid2.E0
        Evec2_data = grid2.Egrid_data
        for j from 0 <= j < dim1:
            #integration on old grid
            flnew_data[i] += K1(Eenew,Evec_data[j],seed)*(flold_data[j]*grid.dEdxgrid_data[j])*grid.dx
        for j from 0 <= j < dim2:
            if True:
                #integration on new grid
                a = K2(Eenew,Evec2_data[j],seed)
                b = flold_func.fofE(Evec2_data[j])
                c = grid2.dEdxgrid_data[j]
                d = grid2.dx
                delta = a*b*c*d
                flnew_data[i] += delta
                #if delta != 0: print "***", i, j, a, b, delta
            else:
                flnew_data[i] += K2(Eenew,Evec_data[j],seed)*(flold_data[j]*Evec_data[j])*grid.dx
    return( flnew )


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
        self.Nprefactor = (1.-s)/(Emax**(1.-s)-Emin**(1.-s))

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
        cdef minEg_val
        if bottom > 0:
            minEg_val = 2*self.Emin*Eenew**2/bottom
            return minEg_val if minEg_val > grid_Emin else grid_Emin
        else:
            return grid_Emin

    cpdef double maxEg(self, double Eenew, double grid_Emax):
        """ Returns minimum gamma-ray energy """
        cdef double bottom = 1-2*self.Emax*Eenew
        cdef maxEg_val
        if bottom > 0:
            maxEg_val = 2*self.Emax*Eenew**2/bottom
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
    cdef public double Emin
    cdef public double Emax
    cdef public double E0
    cdef public double xmin
    cdef public double xmax
    cdef public int Ngrid
    cdef public double dx
    cdef public Egrid
    cdef public xgrid
    cdef public dEdxgrid
    cdef double *xgrid_data
    cdef double *Egrid_data
    cdef double *dEdxgrid_data

    def __init__(self, double Emin, double Emax, double E0, int Ngrid):
        """ Full constructor: allocates memory and generates the grid """
        self.Ngrid = Ngrid
        self.xgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.Egrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.dEdxgrid = np.zeros((self.Ngrid),dtype=DTYPE)
        self.set_grid( Emin, Emax, E0 )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 0, 1, 0.5, Ngrid )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef set_grid(self, double Emin, double Emax, double E0 ):
        """ Same as Grid() but without reallocation of memory """
        cdef int i
        cdef int dim = self.Ngrid
        self.Emin = Emin
        self.Emax = Emax
        self.E0 = E0
        self.xmax = log(self.Emax-self.E0)
        self.xmin = log(self.Emin-self.E0)
        self.dx = (self.xmax - self.xmin) * 1.0 / dim
        #get direct C pointers to numpy arrays' data fields
        self.xgrid_data = get_data(self.xgrid)
        self.Egrid_data = get_data(self.Egrid)
        self.dEdxgrid_data = get_data(self.dEdxgrid)
        for i from 0 <= i < dim:
            self.xgrid_data[i] = self.xmin + self.dx*(i+0.5)
            self.Egrid_data[i] = self.E0 + exp(self.xgrid_data[i])
            self.dEdxgrid_data[i] = self.Egrid_data[i] - self.E0

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef int iofx(self, double xval):
        """ Returns the index of the cell containing xval """
        cdef int ival
        ival = int( (xval-self.xmin)/self.dx - 0.5 )
        return ival

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef double xofE(self, double Eval):
        """ Returns the value of x corresponding to Eval """
        cdef double xval
        xval = log(Eval - self.E0)
        return xval

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cpdef int iofE(self, double Eval):
        """ Returns the index of the cell containing Eval """
        return self.iofx( self.xofE(Eval) )


###############################
#
#  FUNCTION
#
###############################        

cdef public class Func(Grid)  [object CFunc, type TFunc ]:
    """ Function class derived from Grid class """
    
    cdef public func_vec
    cdef double *func_vec_data

    def __init__(self, double Emin, double Emax, double E0, int Ngrid, func_vec = None):
        Grid.__init__(self, Emin, Emax, E0, Ngrid)
        if func_vec is None:
            self.func_vec = np.zeros((self.Ngrid),dtype=DTYPE)
            self.func_vec_data = get_data(self.func_vec)
        else:
            self.func_vec = np.copy(func_vec)
            self.func_vec_data = get_data(self.func_vec)

    cpdef set_grid(self, double Emin, double Emax, double E0):
        """ Same as Grid() but without reallocation of memory """
        Grid.set_grid( self, Emin, Emax, E0 )

    @classmethod
    def fromGrid(cls, Grid grid):
        return cls( grid.Emin, grid.Emax, grid.E0, grid.Ngrid )

    @classmethod
    def empty(cls, int Ngrid):
        return cls( 0, 1, 0.5, Ngrid )

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
    cpdef double fofE(self, double Eval):
        """ Linearly interpolates f(E) in log-log """
        cdef int i = Grid.iofE( self, Eval )
        cdef double logfl, logfr, logxl, logxr, logf, f
        cdef eps = 1e-300
        if i < 0:
            return self.func_vec_data[0]
        if i >= self.Ngrid-1:
            return self.func_vec_data[self.Ngrid-1]
        if True:
            logx  = log(Eval)
            logxl = log(self.Egrid[i])
            logxr = log(self.Egrid[i+1])
            logfl = log(self.func_vec_data[i]+eps)
            logfr = log(self.func_vec_data[i+1]+eps)
            logf  = (logfr * (logxl - logx) + logfl * (logx - logxr)) / (logxl - logxr)
            f = exp(logf)-eps
        else:
            f = self.func_vec_data[i]
        return( f )
        
    def set_func(self, func_vec):
        self.set_func_c( get_data(func_vec) )

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef set_func_c(self, double *func_vec_data):
        cdef int i
        for i from 0 <= i < self.Ngrid:
            self.func_vec_data[i] = func_vec_data[i]


