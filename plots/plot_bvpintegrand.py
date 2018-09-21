import numpy as np
from bvpnbc import BoundaryValueProblem, ExponentialKernel
import matplotlib.pyplot as plt
import sys

sig2z = [1., 4., 16., 64., 256.]
clz = [1./32, 1./16, 1./8, 1./4, 1./2]

def plot_integrands(cl, sig2, Z):
    
    ind_cl = clz.index(cl)
    ind_sig2 = sig2z.index(sig2)
    
    M, m = Z.shape
    n = 2**15-1
    
    ek = ExponentialKernel(m, cl=cl, sig2=sig2)
    
    tgrid = np.linspace(-1., 1., n+2)
    ek.compute_eigenvecs(tgrid)
    F = (1. - tgrid.reshape((n+2, 1))) / np.exp(ek.evaluate(Z.transpose()))
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(M):
        plt.plot(tgrid, F[:,k], '-')
    plt.grid(True)
    #plt.xlabel(r'$x^T u$')
    #plt.ylabel(r'$f(x)$')
    plt.xlim([-1., 1.])
    #plt.ylim([0.4, 3.5])
    #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
    plt.savefig('figs/integrand_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(M):
        plt.plot(tgrid, np.log10(F[:,k]), '-')
    plt.grid(True)
    #plt.xlabel(r'$x^T u$')
    #plt.ylabel(r'$f(x)$')
    plt.xlim([-1., 1.])
    #plt.ylim([0.4, 3.5])
    #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
    plt.savefig('figs/logintegrand_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
        
        

if __name__ == '__main__':
    
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./32, 1./16, 1./8, 1./4, 1./2]
    m = 1000
    
    for sig2 in sig2z:

        for cl in clz:
            
            Nsamples = 10
            Z = np.random.uniform(-1., 1., size=(Nsamples, m))
            plot_integrands(cl, sig2, Z)
            
            print 'sig2 {:4.2f}, cl {:4.2f}'.format(sig2, cl)
    




    
    