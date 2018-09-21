import numpy as np
import matplotlib.pyplot as plt
import active_subspaces as ac
import os

if __name__ == '__main__':
    
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./32, 1./16, 1./8, 1./4, 1./2]
    
    for filename in os.listdir('samples'):

        data = np.load('samples/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        samplez = data['samplez']
        X, f = samplez[:,:1000], samplez[:,1000].reshape((samplez.shape[0], 1))
        
        ss = ac.subspaces.Subspaces()
        ss.compute(X=X, f=f, sstype='OLS')
        y = np.dot(X, ss.W1)

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.plot(y, f, '.')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        #plt.xlim([-1., 1.])
        #plt.ylim([0.4, 3.5])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/ssp1_OLS_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        
    for filename in os.listdir('grads'):

        data = np.load('grads/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        X, f, df = data['X'], data['f'], data['df']
        f = f.reshape((len(f), 1))
        
        ss = ac.subspaces.Subspaces()
        ss.compute(df=df, sstype='AS')
        y = np.dot(X, ss.W1)
        

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.plot(y, f, '.')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        #plt.xlim([-1., 1.])
        #plt.ylim([0.4, 3.5])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/ssp1_AS_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.semilogy(np.arange(20), ss.eigenvals[:20], 'ro')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        #plt.xlim([-1., 1.])
        plt.ylim([10**(-8), 10**11])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/eigs_AS_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    for filename in os.listdir('grads'):

        data = np.load('grads/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        X, f, df = data['X'], data['f'], data['df']
        f = f.reshape((len(f), 1))
        
        ss = ac.subspaces.Subspaces()
        ss.compute(df=df, sstype='NAS')
        y = np.dot(X, ss.W1)
        
        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.plot(y, f, '.')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        #plt.xlim([-1., 1.])
        #plt.ylim([0.4, 3.5])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/ssp1_NAS_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)
        plt.semilogy(np.arange(20), ss.eigenvals[:20], 'ro')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        #plt.xlim([-1., 1.])
        plt.ylim([10**(-8), 10**0])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/eigs_NAS_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
    