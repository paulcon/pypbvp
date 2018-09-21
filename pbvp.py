import numpy as np
from scipy.optimize import brentq

EPS = np.finfo(float).eps

class ParameterizedBoundaryValueProblem():
    '''
    A class for the following parameterized boundary value problem:
    
    ( kappa(t,x) u'(t,x) )' = 1, u(-1,x) = 0, u'(1,x) = 1, t \in [-1, 1],
    
    where x represents the vector of parameters that affect the variable
    coefficient function kappa(t,x). 
    
    Attributes
    ----------
        kappa : RandomField 
            class that defines the parameterized variable coefficients kappa
            
    Methods
    -------
        
            
    Notes
    -----
    
    '''

    kappa = []
    
    def __init__(self, kappa):
        '''
        Initialize the boundary value problem with a random field for the 
        variable coefficients.
        '''
        self.kappa = kappa
                
    def _integrand(self, Z, n):
        '''
        
        
        Parameters
        ----------
            Z : ndarray 
                array of inputs [REWRITE]
            n : int
                number of points [REWRITE]
                
        Notes
        -----
    
        '''
        kappa = self.kappa
        
        # evaluate the eigenfunctions
        tgrid = np.linspace(-1., 1., n+2)
        if len(kappa.eigenvecs) == 0 or kappa.eigenvecs.shape[0] != n+2:
            kappa.compute_eigenvecs(tgrid)
        
        # compute the integrands
        return tgrid, (1. - tgrid.reshape((n+2, 1))) / np.exp(kappa.evaluate(Z.transpose()))
                
    def quantity_of_interest(self, Z, n=32767, gradflag=False):
        '''
        
        
        Parameters
        ----------
            Z : ndarray 
                array of inputs [REWRITE]
            n : int
                number of points [REWRITE]
                
        Returns
        -------
            f : ndarray
                stuff [REWRITE]
            df : ndarray
                stuff [REWRITE]
                
        Notes
        -----
    
        '''
        kappa = self.kappa
        M = Z.shape[0]
        
        # evaluate the integrands
        tgrid, F = self._integrand(Z, n)
        
        # trapezoidal weights
        dt = tgrid[1] - tgrid[0]
        w = dt*np.ones((1, n+2))
        w[0,0], w[0,-1] = 0.5*dt, 0.5*dt
        
        # evaluate the integrals
        f = np.dot(w, F)
        
        if gradflag:
            B = kappa.eigenvecs * np.sqrt(kappa.eigenvals)
            df = -np.dot(w * F.transpose(), B)
        else:
            df = []
            
        return f.reshape((M,)), df

class RandomField():
    """
    Implementing 1d exponential kernal from Appendix A.2.2 from Fasshauer and 
    McCourt.
    
    K(x,z) = exp( - | x - z | / (2*cl) ), where x,z \in [-L, L]
    
    Attributes
    ----------
        kappa : RandomField 
            class that defines the parameterized variable coefficients kappa
            
    Methods
    -------
        
            
    Notes
    -----
    
    """
    N = []
    eps = []
    L = []
    w, v = [], []
    eigenvals = []
    eigenvecs = []
    
    def __init__(self, N, cl=0.5, sig2=1.):
        '''
        Initialize the boundary value problem with a random field for the 
        variable coefficients.
        '''
        
        # let's just make N even
        if np.mod(N, 2):
            raise Exception('Dimension N must be even.')
        
        self.N = N
        self.eps = 1./(2*cl)
        self.sig2 = sig2
        self.L = 1.
        self.w, self.v = self._ev_coeff(N)
        self.eigenvals = self.compute_eigenvals()
        self.eigenvecs = []
        
    def _ev_coeff(self, N):
        '''
        
        
        Parameters
        ----------
            N : int
                number of terms in the KL, must be even [REWRITE]
                
        Notes
        -----
    
        '''
        # assumes N is even
        w, v = np.zeros((N/2, )), np.zeros((N/2, ))
        
        # odd indices
        def odd_fun(w):
            return self.eps - w*np.tan( self.L*w )
        
        for i in range(N/2):
            n = 2*i + 1
            a = (2*n - 1)*np.pi/(2*self.L) + np.sqrt(EPS)
            b = (2*n + 1)*np.pi/(2*self.L) - np.sqrt(EPS)
            w0 = brentq(odd_fun, a, b, maxiter=2000)
            w[i] = w0
        
        # even indices
        def even_fun(v):
            return v + self.eps*np.tan( self.L*v )
            
        for i in range(N/2):
            n = 2*i + 2
            a = (2*n - 1)*np.pi/(2*self.L) + np.sqrt(EPS)
            b = (2*n + 1)*np.pi/(2*self.L) - np.sqrt(EPS)
            v0 = brentq(even_fun, a, b, maxiter=2000)
            v[i] = v0
        
        return w, v
        
    def compute_eigenvals(self):
        '''
        
        
                
        Notes
        -----
    
        '''
        eps, sig2, w, v, N = self.eps, self.sig2, self.w, self.v, self.N
        
        lamda = np.vstack( ( \
            (2*eps) / (eps**2 + np.power(w,2)), \
            (2*eps) / (eps**2 + np.power(v,2)) ) )
        return sig2*lamda.transpose().reshape((N, ))
        
    def compute_eigenvecs(self, t):
        '''
        
        
        Parameters
        ----------
            t : ndarray
                points at which to evaluate eigenvecs
                
        Notes
        -----
    
        '''
        if numgigs(len(t), self.N) > 6:
            raise Warning('Storing the eigenvectors will take more than 6 GB of memory.')
        
        t = t.reshape((len(t),1))
        
        # get these numerically generated coefficients
        L, w, v, N = self.L, self.w, self.v, self.N
        
        # eigenfunctions
        phi_odd = np.sqrt( (2.*w) / (2.*L*w - np.sin(2.*L*w)) ) * np.cos( np.outer(t, w) )
        phi_even = np.sqrt( (2.*v) / (2.*L*v - np.sin(2.*L*v)) ) * np.sin( np.outer(t, v) )
        phi = np.zeros((len(t), N))
        ind_odd, ind_even = np.arange(1, N+1, 2), np.arange(2, N+1, 2)
        phi[:,ind_odd-1] = phi_odd
        phi[:,ind_even-1] = phi_even
        self.eigenvecs = phi[:,:N]
        
    def evaluate(self, Z):
        '''
        
        
        Parameters
        ----------
            Z : ndarray 
                array of inputs [REWRITE]
                
        Notes
        -----
    
        '''
        N, eigenvals, eigenvecs = self.N, self.eigenvals, self.eigenvecs
        
        if Z.shape[0] != N:
            raise Exception('Dimension of random inputs is {:d} but should be {:d}.'.format(Z.shape[0], N))
        
        return np.dot( eigenvecs, np.sqrt(eigenvals).reshape((N, 1))*Z )
        
def numgigs(nrows, ncols):
    return nrows*ncols*8.0 / float(2**30)

    
    