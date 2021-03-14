import numpy as np
import scipy.optimize as opt
from copy import deepcopy as copy
pi = np.pi


def naff(nmode,signal,window_id=1):
  """
  tunes,amps,substracted_signals = naff(nmode,signal)
  t=[0,T-1]
  amp = signal*np.exp(-2j*pi*tune*np.arange(T))
  """
  T = len(signal)
  window = (1.0+np.cos(np.pi*(-1.0+2.0/(T+1.0)*np.arange(1,T+1))))**window_id
  window = window/np.sum(window)


  def getPeakInfo(signal):
    T = len(signal)
    def loss(tune):
      return -np.abs(np.sum(signal*window*np.exp(-2j*pi*tune*np.arange(T))))
    tune = np.argmax(np.abs(np.fft.fft(signal)))/T
    result = opt.differential_evolution(loss,((tune-2.2/T,tune+2.2/T),),popsize=9)
    return result

  tunes = []
  amps = []
  subtracted_signals = []

  X = copy(signal)
  for i in range(nmode):
    result = getPeakInfo(X)
    if result.message!='Optimization terminated successfully.':
      print('Optimization failed at '+str(i+1)+'-th mode')
      break
    tunes.append(copy(result.x))
    amps.append(np.sum(X*np.exp(-2j*pi*tunes[-1]*np.arange(T)))/T)

    X = X - amps[-1]*np.exp(2j*pi*tunes[-1]*np.arange(T))
    subtracted_signals.append(copy(X))

  return tunes,amps,subtracted_signals