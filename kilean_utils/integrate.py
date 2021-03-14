from scipy.special import roots_legendre
import numpy as np
  

def gaussian_quad(func,a,b,n_sub_interval=8,n_order=32):
  '''
  1D function Gaussian qaudrature integration from a to b
  '''
  L = b-a
  dL = L/n_sub_interval
  X0,W0 = roots_legendre(n_order)
  X = dL*(X0+1.)/2. 
  y = 0.
  for i in range(n_sub_interval):
    x_min = a + i*dL
    y = y + np.sum(W0*func(X+x_min), axis=-1)
  return y*dL/2.0