import numpy as np
from pbvp import ParameterizedBoundaryValueProblem, RandomField

d = 10                  # number of terms in the KL
cl = 0.5                # correlation length
sig2 = 1.               # variance
n = 32767               # number of points in spatial domain

# instantiate the random field
kappa = RandomField(d, cl, sig2)

# instantiate the boundary value problem
bvp = ParameterizedBoundaryValueProblem(kappa)

# draw 100 samples from the parameter space
X = np.random.uniform(-1., 1., (100, d))

# evaluate the function and its gradient at each input sample
f, df = bvp.quantity_of_interest(X, n, gradflag=True)
