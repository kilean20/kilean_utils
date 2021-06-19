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




def gaussian_quad2D(func,xmin,xmax,ymin,ymax,n_sub_interval_x=8, n_sub_interval_y=8, n_order=32):
  '''
  2D function Gaussian qaudrature integration
  '''
  Lx = xmax-xmin
  Ly = ymax-ymin
  dLx = Lx/n_sub_interval_x
  dLy = Ly/n_sub_interval_y
  X0,W0 = roots_legendre(n_order)
  x = dLx*(X0+1.)/2. 
  y = dLy*(X0+1.)/2. 
  z = 0.
  for i in range(n_sub_interval_x):
    x_min = xmin + i*dLx
    tmp = 0
    for j in range(n_sub_interval_y):
        y_min = ymin + j*dLy
        tmp += np.sum(W0*func(x+x_min,y+y_min), axis=-1)
    z += np.sum(W0*tmp, axis=-1)
  return z*dLx*dLy/4.0