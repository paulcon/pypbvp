import numpy as np
from bvpnbc import BoundaryValueProblem, ExponentialKernel
import matplotlib.pyplot as plt
import sys

def error_study(cl, sig2, Z):
    
    m = Z.shape[1]
    
    ek = ExponentialKernel(m, cl=cl, sig2=sig2)
    bvp = BoundaryValueProblem(ek)
    
    ftrue = bvp.sol(Z, n=2**18-1)[0]
    
    maxk = 16
    errz = []
    nz = []
    
    for k in range(3, maxk):
        
        nx = 2**k - 1
        nz.append(nx)
        
        bvp.kappa.eigenvecs = []
        
        f = bvp.sol(Z, n=nx)[0]
        
        errz.append( np.fabs(f - ftrue) / np.fabs(ftrue) )
        
    return np.array(nz), np.array(errz)

if __name__ == '__main__':
    
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./32, 1./16, 1./8, 1./4, 1./2]
    m = 1000
    
    for sig2 in sig2z:

        for cl in clz:
            
            Nsamples = 500
            Z = np.random.uniform(-1., 1., size=(Nsamples, m))
            nz, errz = error_study(cl, sig2, Z)
            
            print 'sig2 {:4.2f}, cl {:4.2f}'.format(sig2, cl)
            
            np.savez('data/{:06d}.npz'.format(np.random.randint(low=0, high=100000)), sig2=sig2, cl=cl, nz=nz, errz=errz)




    
    