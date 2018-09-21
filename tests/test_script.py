import numpy as np
from bvpnbc import ExponentialKernel, BoundaryValueProblem
import matplotlib.pyplot as plt

np.random.seed(44)

m, M = 1000, 10
ek = ExponentialKernel(m, cl=1./32, sig2=256.)
bvp = BoundaryValueProblem(ek)
Z = np.random.normal(size=(M, m))
f, df = bvp.sol(Z, n=2**15-1, gradflag=True)

plt.figure()
plt.semilogy(np.arange(M), f, 'bo')
plt.show()


"""
plt.figure()
plt.semilogy(np.arange(m), np.fabs(df).transpose(), '.')
plt.show()

# test finite diff gradient
h = 1e-6
for i in range(10):
    
    z = np.random.normal(size=(1, m))
    fz, dfz = bvp.sol(z, gradflag=True)
    
    zp = z + h*np.eye(m, m)
    fzp = bvp.sol(zp)[0]
    
    df_fd = (fzp - fz) / h
    
    print 'Grad error: {:6.4e}'.format( np.linalg.norm(df_fd - dfz) / np.linalg.norm(dfz) )
"""

"""
plt.figure(figsize=(7,7))
plt.plot(xgrid, u, '-')
plt.grid(True)
plt.title('BVP')

plt.show()
"""