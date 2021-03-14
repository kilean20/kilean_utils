import numpy as np


def getEmittance(X):
  '''
  input:
    X: 2-dim array representing (x,p) data
  '''
  sigx = np.std(X[:,0])
  sigp = np.std(X[:,1])
  sigxp = np.sum((X[:,0]-X[:,0].mean())*(X[:,1]-X[:,1].mean()))/len(X)
  return np.sqrt(sigx*sigx*sigp*sigp - sigxp*sigxp)