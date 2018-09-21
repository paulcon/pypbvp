import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def kl_plots(fname):
    data = np.load(fname)
    ind, kl_evals, tgrid, kl_evecs = data['ind'], data['kl_evals'], data['tgrid'], data['kl_evecs']
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.semilogy(ind, kl_evals, 'ro')
    plt.grid(True)
    plt.ylim([10**(-7), 10**2])
    plt.savefig('figs/klevals_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(3):
        plt.plot(tgrid, kl_evecs[:,k], '-')
    plt.grid(True)
    plt.xlim([-1., 1.])
    plt.ylim([-1.1, 1.1])
    plt.savefig('figs/klevecs_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

def integrand_plots(fname, sig2ind):
    data = np.load(fname)
    tgrid, F = data['tgrid'], data['F']
    
    ylimz = [[0.,8.],[0.,33.],[0.,680.],[0.,32000.],[0.,2e12]]
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(F.shape[1]):
        plt.plot(tgrid, F[:,k], '-')
    plt.grid(True)
    plt.xlim([-1., 1.])
    plt.ylim(ylimz[sig2ind])
    #plt.set_powerlimits([-20,3])
    plt.savefig('figs/integrand_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(F.shape[1]):
        plt.plot(tgrid, np.log10(F[:,k]), '-')
    plt.grid(True)
    plt.xlim([-1., 1.])
    plt.ylim([-10.,10.])
    plt.savefig('figs/log10integrand_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
            
def histf_plots(fname, sig2ind):
    data = np.load(fname)
    f = data['f']
    
    nbins = 40
    xlimz = [[1.97,3.03],[1.9,6.],[0.,60.],[0.,32000.],[0.,1e11]]
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.hist(f, bins=nbins)
    plt.xlim(xlimz[sig2ind])
    plt.grid(True)
    #plt.set_powerlimits([-20,3])
    plt.savefig('figs/histf_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.hist(np.log10(np.fabs(f)), bins=nbins)
    plt.xlim([0.,10.])
    plt.grid(True)
    plt.savefig('figs/histlog10f_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
            
def df_plots(fname):
    data = np.load(fname)
    ind, df = data['ind'], data['df']
    
    """
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.plot(ind, df.transpose(), '.')
    plt.grid(True)
    plt.savefig('figs/grads_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    """
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.plot(ind, np.log10(np.fabs(df[:20,:].transpose())), '.')
    plt.grid(True)
    plt.ylim([-10.,10])
    plt.savefig('figs/log10grads_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
            
def conv_plots(fname):
    data = np.load(fname)
    nz, errz = data['nz'], data['errz']
    
    maxerr, meanerr, minerr = np.amax(errz, axis=1), np.mean(errz, axis=1), np.amin(errz, axis=1)
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.loglog(nz, maxerr, 'r--')
    plt.loglog(nz, meanerr, 'bo-')
    plt.loglog(nz, minerr, 'r--')
    plt.grid(True)
    plt.ylim([10**(-11), 10**2])
    plt.savefig('figs/conv_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
                
def sweep_plots(fname, sig2ind):
    data = np.load(fname)
    sweepz = data['sweepz']
    
    ylimz = [[2.,2.3],[2.,3.8],[2.,22.],[0.,750.],[0.,0.9e7]]
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(sweepz.shape[0]):
        s = sweepz[k,:,:]
        plt.plot(s[:,0], s[:,1], '-')
    plt.grid(True)
    plt.xlim([-1.5, 1.5])
    plt.ylim(ylimz[sig2ind])
    plt.savefig('figs/sweep_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    for k in range(sweepz.shape[0]):
        s = sweepz[k,:,:]
        ss = np.linspace(-1., 1., len(s[:,0]))
        plt.plot(ss, s[:,1], '-')
    plt.grid(True)
    plt.xlim([-1., 1.])
    plt.ylim(ylimz[sig2ind])
    plt.savefig('figs/nsweep_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
        
def ols_plots(fname, sig2ind):
    data = np.load(fname)
    ind, w, y, f = data['ind'], data['w'], data['y'], data['f']
    
    ylimz = [[1.9,3.],[2.,6.],[0.,60.],[0.,40000.],[0.,1.05e11]]
    xlimz = [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-2.2, 10.], [-2.2, 12.]]
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.plot(y[:1000], f[:1000,0], '.')
    plt.ylim(ylimz[sig2ind])
    plt.xlim(xlimz[sig2ind])
    plt.grid(True)
    plt.savefig('figs/ssp_OLS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.plot(ind, w, 'ro')
    plt.ylim([-1., 1.])
    plt.grid(True)
    plt.savefig('figs/weight_OLS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
        
def as_plots(fname, normflag, sig2ind):
    data = np.load(fname)
    evals, evecs, y, f = data['evals'], data['evecs'], data['y'], data['f']
    
    ylimz = [[1.9,3.],[2.,6.],[0.,60.],[0.,40000.],[0.,1.05e11]]

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.plot(y[:1000], f[:1000,0], '.')
    plt.ylim(ylimz[sig2ind])
    plt.xlim([-2.2,2.2])
    plt.grid(True)
    if normflag:
        plt.savefig('figs/ssp_NAS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.savefig('figs/ssp_AS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.plot(y[:1000], np.log10(f[:1000,0]), '.')
    plt.ylim([-1., 12])
    plt.xlim([-2.2, 2.2])
    plt.grid(True)
    if normflag:
        plt.savefig('figs/log10ssp_NAS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.savefig('figs/log10ssp_AS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    plt.semilogy(np.arange(len(evals)), evals, 'ro')
    plt.grid(True)
    if normflag:
        plt.ylim([10**(-8), 10**1])
        plt.savefig('figs/evals_NAS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.ylim([10**(-8), 10**15])
        plt.savefig('figs/evals_AS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    legz = []
    for k in range(evecs.shape[1]):
        l, = plt.plot(np.arange(evecs.shape[0]), evecs[:,k], 'o', label='{:d}'.format(k))
        legz.append(l)
    #plt.legend(handles=legz)
    plt.grid(True)
    plt.ylim(-1., 1.)
    if normflag:
        plt.savefig('figs/evecs_NAS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.savefig('figs/evecs_AS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    legz = []
    for k in range(evecs.shape[1]):
        l, = plt.plot(np.arange(evecs.shape[0]), np.log10(np.fabs(evecs[:,k])), 'o', label='{:d}'.format(k))
        legz.append(l)
    #plt.legend(handles=legz)
    plt.grid(True)
    plt.ylim(-10., 0.2)
    if normflag:
        plt.savefig('figs/log10evecs_NAS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.savefig('figs/log10evecs_AS_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

def dgsm_plots(fname, sig2ind):
    data = np.load(fname)
    dgsm1, dgsm2 = data['dgsm1'], data['dgsm2']
    
    ylimz = [[0.,0.3],[0.,1.3],[0.,12.],[0.,2100.],[0.,0.8e10]]
    
    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    l1, = plt.plot(np.arange(len(dgsm1)), dgsm1, 'o', label='DGSM1')
    l2, = plt.plot(np.arange(len(dgsm2)), dgsm2, 'o', label='DGSM2')
    plt.legend(handles=[l1, l2])
    plt.ylim(ylimz[sig2ind])
    plt.grid(True)
    plt.savefig('figs/DGSM_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.figure(figsize=(2.5,2.5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    l1, = plt.plot(np.arange(len(dgsm1)), np.log10(dgsm1), 'o', label='DGSM1')
    l2, = plt.plot(np.arange(len(dgsm2)), np.log10(dgsm2), 'o', label='DGSM2')
    plt.legend(handles=[l1, l2])
    plt.ylim([-8,11])
    plt.grid(True)
    plt.savefig('figs/log10DGSM_sig2_{:02d}_cl_{:02d}.eps'.format(ii, jj), dpi=300, bbox_inches='tight', pad_inches=0.01)


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

            # plot KL components
            fname = 'data/kl_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            kl_plots(fname)
            
            # plot integrands
            fname = 'data/integrands_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            integrand_plots(fname, ii)
            
            # plot hist of f, hist of log f
            fname = 'data/OLS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            histf_plots(fname, ii)
            
            # plot grad samples, log grad samples
            fname = 'data/grads_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            df_plots(fname)
            
            # plot convergence study
            fname = 'data/errz_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            conv_plots(fname)
            
            # random sweeps
            fname = 'data/sweepz_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            sweep_plots(fname, ii)
            
            # OLS subspace
            fname = 'data/OLS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            ols_plots(fname, ii)

            # Active subspace, normalized active subspace
            fname = 'data/AS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            as_plots(fname, False, ii)
            
            fname = 'data/NAS_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            as_plots(fname, True, ii)

            # plot the DGSM
            fname = 'data/DGSM_sig2_{:03d}_cl_{:03d}.npz'.format(ii,jj)
            dgsm_plots(fname, ii)
