import numpy as np
from bvpnbc import BoundaryValueProblem, ExponentialKernel
    
def sweep(fun, p, u, M=100):
    """

    """
    m = len(p)
    
    # get the range of h
    ind = u > 0
    hmin = np.amax( np.concatenate(( (1 - p[~ind])/u[~ind], (-1 - p[ind])/u[ind] )) )
    hmax = np.amin( np.concatenate(( (1 - p[ind])/u[ind], (-1 - p[~ind])/u[~ind] )) )
    hz = np.linspace(hmin, hmax, M)
    
    # get the Z's
    Z = p.reshape((1, m)) + np.outer(hz, u)
    
    # evaluate the function
    f = fun(Z).reshape((M, 1))
    
    return np.hstack(( np.dot(Z, u), f ))

def sample(fun, m, M=10000):
    """

    """
    Z = np.random.uniform(-1., 1., size=(M, m))
    
    # evaluate the function
    f = fun(Z).reshape((M, 1))

    return np.hstack(( Z, f ))

if __name__ == '__main__':
    
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./32, 1./16, 1./8, 1./4, 1./2]
    m, M = 1000, 2000
    
    count = 0
    for sig2 in sig2z:

        for cl in clz:
            
            # set up the function
            ek = ExponentialKernel(m, cl=cl, sig2=sig2)
            bvp = BoundaryValueProblem(ek)
            
            # evaluate the function
            X = np.random.uniform(-1., 1., size=(M, m))
            f, df = bvp.sol(X, n=2**17-1, gradflag=True)
            
            np.savez('grads/{:06d}.npz'.format(np.random.randint(low=0, high=100000)), sig2=sig2, cl=cl, X=X, f=f, df=df)
            
            print 'count: {:d}, sig2 {:4.2f}, cl {:4.2f}'.format(count, sig2, cl)
            
            count += 1
    
    """
    plt.figure(figsize=(5,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.boxplot(np.log10(errz).transpose())
    plt.grid(True)
    plt.xlabel(r'$n$')
    plt.xticks([1, 5, 9, 13], [r'$2^{3}$', r'$2^{7}$', r'$2^{11}$', r'$2^{15}$' ], rotation='horizontal')
    plt.ylabel(r'$\log_{10}(\mbox{error})$')
    plt.xlim([0, 14])
    plt.ylim([-8, 2])
    plt.savefig('figs/err.eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    """



    
    