import numpy as np
from bvpnbc import ExponentialKernel, BoundaryValueProblem
import active_subspaces as ac

def sweep(fun, m, M=20):
    """

    """
    
    # random point in domain
    x = np.random.uniform(-1., 1., size=(m, 1))
    
    # random direction
    u = np.random.normal(size=(m, 1))
    u = u / np.linalg.norm(u)
    
    # get the range of h
    ind = u > 0
    hmin = np.amax( np.concatenate(( (1 - x[~ind])/u[~ind], (-1 - x[ind])/u[ind] )) )
    hmax = np.amin( np.concatenate(( (1 - x[ind])/u[ind], (-1 - x[~ind])/u[~ind] )) )
    hz = np.linspace(hmin, hmax, M)
    
    # get the Z's
    Z = x.reshape((1, m)) + np.outer(hz, u)
    
    # evaluate the function
    f = fun(Z).reshape((M, 1))
    
    return np.hstack(( np.dot(Z, u), f ))

def convergence_study(bvp, Z):
    """
    
    """
    ftrue = bvp.qoi(Z, n=2**18-1)[0]
    
    maxk = 16
    nz, errz = [], []
    
    for k in range(3, maxk):
        
        nx = 2**k - 1
        nz.append(nx)
        
        f = bvp.qoi(Z, n=nx)[0]
        errz.append( np.fabs(f - ftrue) / np.fabs(ftrue) )
        
    return np.array(nz), np.array(errz)

if __name__ == '__main__':
    
    np.random.seed(34) # Sweetness, baby! 
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./256, 1./64, 1./16, 1./4, 1./1]
    m = 1000
    
    for ii in range(len(sig2z)):
        sig2 = sig2z[ii]

        for jj in range(len(clz)):
            cl = clz[jj]
            
            print 'sig2 {:4.2f}, cl {:4.2f}'.format(sig2, cl)
            
            # save KL components
            print '\t KL components...'
            
            ek = ExponentialKernel(m, cl=cl, sig2=sig2)
            ind = np.arange(m) + 1
            n = 2**17-1
            tgrid = np.linspace(-1., 1., n+2)
            ek.compute_eigenvecs(tgrid)
            
            fname = 'data/kl_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, ind=ind, kl_evals=ek.eigenvals, tgrid=tgrid, kl_evecs=ek.eigenvecs[:,:10], sig2=sig2, cl=cl)
            
            # save M realizations of integrand
            print '\t integrands...'
            
            bvp = BoundaryValueProblem(ek)
            M = 50
            Z = np.random.uniform(-1., 1., size=(M, m))
            tgrid, F = bvp._integrand(Z, n)
            
            fname = 'data/integrands_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, tgrid=tgrid, F=F, sig2=sig2, cl=cl)
            
            # save M realizations of the qoi and gradient
            print '\t gradients...'
            
            f, df = bvp.qoi(Z, n, gradflag=True)
            
            fname = 'data/grads_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, f=f, ind=ind, df=df, sig2=sig2, cl=cl)
            
            # convergence study in t
            print '\t convergence study...'
            
            M = 500
            Z = np.random.uniform(-1., 1., size=(M, m))
            nz, errz = convergence_study(bvp, Z)
            
            fname = 'data/errz_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, nz=nz, errz=errz, sig2=sig2, cl=cl)
                
            # random sweeps
            print '\t random sweeps...'
            
            fun = lambda Z: bvp.qoi(Z, n=n)[0]
            
            sweepz = []
            Nsweeps = 10
            for i in range(Nsweeps):
                sweepz.append(sweep(fun, m))
                
            fname = 'data/sweepz_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, sweepz=sweepz, sig2=sig2, cl=cl)
            
            # 10k random samples with gradients
            print '\t 10k samples...'
            
            M = 10000
            Z = np.random.uniform(-1., 1., size=(M, m))
            f, df = bvp.qoi(Z, n, gradflag=True)
            f = f.reshape((M, 1))
            
            # OLS subspace
            print '\t OLS subspace...'
            
            ss = ac.subspaces.Subspaces()
            ss.compute(X=Z, f=f, sstype='OLS')
            v = ss.eigenvecs[:,0]
            y = np.dot(Z, v)
            
            fname = 'data/OLS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, ind=ind, w=v, y=y, f=f, sig2=sig2, cl=cl)
            
            # 10k random samples for AS
            print '\t active subspace...'
            
            ss.compute(df=df, sstype='AS')
            y = np.dot(Z, ss.eigenvecs[:,0])
            
            fname = 'data/AS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, evals=ss.eigenvals[:20], evecs=ss.eigenvecs[:,:5], y=y, f=f, sig2=sig2, cl=cl)
            
            # 10k random samples for NAS
            print '\t normalized active subspace...'
            
            ss.compute(df=df, sstype='NAS')
            y = np.dot(Z, ss.eigenvecs[:,0])
            
            fname = 'data/NAS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, evals=ss.eigenvals[:20], evecs=ss.eigenvecs[:,:5], y=y, f=f, sig2=sig2, cl=cl)
            
            # compute the DGSM
            print '\t compute DGSM...'
            
            dgsm1 = np.mean(np.fabs(df), axis=0)
            dgsm2 = np.sqrt(np.mean(df*df, axis=0))
            
            fname = 'data/DGSM_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            np.savez(fname, dgsm1=dgsm1, dgsm2=dgsm2, sig2=sig2, cl=cl)
            