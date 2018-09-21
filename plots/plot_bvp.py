import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    
    sig2z = [1., 4., 16., 64., 256.]
    clz = [1./32, 1./16, 1./8, 1./4, 1./2]
    
    for filename in os.listdir('data'):
        
        data = np.load('data/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        cl = clz[ind_cl]
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        errz = data['errz']
        nz = data['nz']
        
        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=10)
        plt.boxplot(np.log10(errz).transpose())
        plt.grid(True)
        #plt.xlabel(r'$n$')
        plt.xticks([1, 5, 9, 13], [r'$2^{3}$', r'$2^{7}$', r'$2^{11}$', r'$2^{15}$' ], rotation='horizontal')
        #plt.ylabel(r'$\log_{10}(\mbox{error})$')
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.xlim([0, 14])
        plt.ylim([-11, 2])
        plt.savefig('figs/err_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    
    for filename in os.listdir('sweeps'):

        data = np.load('sweeps/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        sweepz = data['sweepz']

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=10)
        for k in range(10):
            s = sweepz[k,:,:]
            plt.plot(s[:,0], s[:,1], '-')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        plt.xlim([-1.5, 1.5])
        plt.ylim([0.4, 3.5])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/sweep_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
    
    
    for filename in os.listdir('sweeps'):

        data = np.load('sweeps/' + filename)
        cl = data['cl']
        ind_cl = clz.index(cl)
        sig2 = data['sig2']
        ind_sig2 = sig2z.index(sig2)
        sweepz = data['sweepz']

        plt.figure(figsize=(2.5,2.5))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=10)
        for k in range(10):
            s = sweepz[k,:,:]
            ss = np.linspace(-1., 1., len(s[:,0]))
            plt.plot(ss, s[:,1], '-')
        plt.grid(True)
        #plt.xlabel(r'$x^T u$')
        #plt.ylabel(r'$f(x)$')
        plt.xlim([-1., 1.])
        #plt.ylim([0.4, 3.5])
        #plt.title('cl: {:6.4f}, sig2: {:6.4f}'.format(cl, float(sig2)))
        plt.savefig('figs/sweep0_eps_{:02d}_sig2_{:02d}.eps'.format(ind_cl, ind_sig2), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()
