import numpy as np
from scipy.optimize import brentq

EPS = np.finfo(float).eps

class ParameterizedBoundaryValueProblem():
    '''
    A class for the following parameterized boundary value problem:
    
    ( \exp(\kappa(t,x)) u'(t,x) )' = 1, u(-1,x) = 0, u'(1,x) = 0, t \in [-1, 1],
    
    where x represents the vector of parameters that affect the variable
    coefficient function \kappa(t,x). With elementary calculus, we can write the 
    BVP solution as
    
    u(t,x) = \int_{-1}^t (1-t) / \exp(\kappa(t,x)) dt.
    
    The class provides a method to estimate the quantity of interest u(1,x) and 
    its gradient with respect to x.
    
    Attributes
    ----------
        kappa : RandomField 
            class that defines the parameterized variable coefficients kappa
            
    Methods
    -------
        quantity_of_interest
            computes the quantity of interest from the BVP solution and its 
            gradient vector with respect to the parameters
    
    '''

    kappa = []
    
    def __init__(self, kappa):
        '''
        Initialize the boundary value problem with a random field for the 
        variable coefficients.

        Parameters
        ----------
            kappa : RandomField
                initialize the BVP instantiation with a RandomField for the 
                variable coefficients
        '''
        self.kappa = kappa
                
    def _integrand(self, Z, n):
        '''
        Compute the integrand 
        
        (1-t) / \exp(\kappa(t,x))
        
        for all t in a grid on the interval [-1,1] and for x equal to each row
        of the input matrix Z.
        
        Parameters
        ----------
            Z : ndarray 
                array of input parameters at which to evaluate the integrand
            n : int
                number of points in the uniform discretization of [-1,1]
                
        Notes
        -----
        The code uses lazy evaluation of the eigenfunctions associated with the
        correlation kernel for the variable coefficients kappa at points in the
        interval [-1,1].

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
        Compute the quantity of interest u(1,x) and (optionally) its gradient
        with respect to the parameters x at several points in the parameter
        space.
        
        Parameters
        ----------
            Z : ndarray 
                array of input parameters at which to evaluate the quantity of 
                interest and its gradient
            n : int, optional
                number of points in the uniform discretization of [-1,1]
            gradflag : boolean
                indicates whether or not to compute and return the gradients of
                the quantity of interest
                
        Returns
        -------
            f : ndarray
                contains evaluations of the quantity of interest at each row
                of the input Z
            df : ndarray
                contains evaluations of the gradient of the quantity of interest
                with respect to the parameters at each row of the input Z
        Notes
        -----
        The default discretization parameter n=32767 is in the asymptotic 
        regime for a wide range of problems defined by different points in the
        parameter space.
        
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
            # compute the gradient of the quantity of interest
            B = kappa.eigenvecs * np.sqrt(kappa.eigenvals)
            df = -np.dot(w * F.transpose(), B)
        else:
            df = []
            
        return f.reshape((M,)), df

class RandomField():
    """
    Implements the eigenpairs for the exponential correlation kernel
    
    K(s,t) = exp( - | s - t | / (2*cl) ), where s,t \in [-L, L]
    
    The eigenpairs have explicit forms which can be found in Appendix A.2.2 from
    Fasshauer and McCourt, Kernel-based Approximation Methods Using MATLAB,
    World Scientific, NJ, 2015. Once these pairs are calculated for the given
    variance and correlation length (scale) parameter, then generating 
    realizations of a Gaussian random field with the same correlation function
    is straightforward; see the 'evaluate' method.
    
    Attributes
    ----------
        N : int
            number of terms in Karhunen-Loeve, must be even
        eps : float
            scale parameter for correlation kernel, inversely proportional to 
            the correlation length
        L : float
            domain size [-L,L] for spatial variable
        w : ndarray
            intermediate values for computing eigenpairs
        v : ndarray
            intermediate values for computing eigenpairs
        eigenvals : ndarray
            (N, )-array of Karhunen-Loeve eigenvalues
        eigenvecs : ndarray
            (len(t), N)-array of Karhunen-Loeve eigenfunctions evaluated at a 
            discrete set of points in the spatial domain
            
            
    Methods
    -------
        compute_eigenvals
            compute the eigenvalues for the correlation kernel
        compute_eigenvecs
            compute the eigenfunctions for the correlation kernel evaluated at
            a grid in the interval [-1,1]
        evaluate
            computes several realizations of a Gaussian random field with the 
            chosen correlation function
         
    """
    N = []
    eps = []
    L = []
    w, v = [], []
    eigenvals = []
    eigenvecs = []
    
    def __init__(self, N, cl=0.5, sig2=1.):
        '''
        Initialize the random field. Compute the intermediate quantities needed
        to estimate the eigenpairs, and compute the top N eigenvalues.

        Parameters
        ----------
            N : int
                number of terms in the Karhunen-Loeve expansion, must be even
            cl : float, optional
                correlation length for the correlation kernel
            sig2 : float, optional
                total variance of the Gaussian random field

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
        Compute the intermediate quantities that define the eigenpairs for the 
        correlation function.
        
        Parameters
        ----------
            N : int
                number of terms in the Karhunen-Loeve expansion, must be even
                
        Notes
        -----
        This method uses the SciPy root finding method 'brentq' to solve the 
        root finding problems whose solution defines the intermediate 
        quantities.
        
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
        Compute the first N eigenvalues of the correlation kernel.
                
        Notes
        -----
        See Appendix A.2.2 from Fasshauer and McCourt, Kernel-based 
        Approximation Methods Using MATLAB, World Scientific, NJ, 2015

        '''
        eps, sig2, w, v, N = self.eps, self.sig2, self.w, self.v, self.N
        
        lamda = np.vstack( ( \
            (2*eps) / (eps**2 + np.power(w,2)), \
            (2*eps) / (eps**2 + np.power(v,2)) ) )
        return sig2*lamda.transpose().reshape((N, ))
        
    def compute_eigenvecs(self, t):
        '''
        Compute the eigenfunctions at the grid on the interval [-1,1].
        
        Parameters
        ----------
            t : ndarray
                points in [-1,1] at which to evaluate the eigenfunctions

        Notes
        -----
        See Appendix A.2.2 from Fasshauer and McCourt, Kernel-based 
        Approximation Methods Using MATLAB, World Scientific, NJ, 2015

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
        Compute several realizations of the Gaussian random field with mean zero
        and chosen correlation kernel.
        
        Parameters
        ----------
            Z : ndarray 
                array of points in the parameter space at which to compute 
                realizations of the random field
                
        Notes
        -----
        The points in the parameter space are coefficients of the Karhunen-
        Loeve expansion associated with the chosen correlation function.
        
        '''
        N, eigenvals, eigenvecs = self.N, self.eigenvals, self.eigenvecs
        
        if Z.shape[0] != N:
            raise Exception('Dimension of random inputs is {:d} but should be {:d}.'.format(Z.shape[0], N))
        
        return np.dot( eigenvecs, np.sqrt(eigenvals).reshape((N, 1))*Z )
        
def numgigs(nrows, ncols):
    '''
    Number of gigabytes of memory needed to store matrix of size nrows-by-ncols.
    '''
    return nrows*ncols*8.0 / float(2**30)

    
    