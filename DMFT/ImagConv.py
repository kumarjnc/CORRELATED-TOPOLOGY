'''
This module provides various functions for defining imaginary time grids or 
Matsubara frequencies, as well as Fourier transformation of functions between 
the two.
'''
from __future__ import print_function
from std_imports import *

def FreqToImagTime(omega,orig,T,beta,leading_order_coeff=1.):
    '''This function converts the function ``orig``, given in Matsubara 
    frequencies ``omega`` (which should be the real :math:`\omega_n` values), 
    into imaginary time at locations ``T``. The inverse temperature ``beta`` is 
    a required argument.

    The optional keyword ``leading_order_coeff`` provides a better Fourier 
    transform for Green's functions (or any function with a leading 
    :math:`1/(i\omega_n)` component. For Green's functions, the default of 
    ``leading_order_coeff = 1`` is always true, however for other functions, 
    such as self-energies, this should be given explicitly.
    '''

    from numpy import pi,array,exp
    #omega = OmegaN()

    #imagtime = 1/beta2 * \
    #           sum(\
    #               orig[n] * exp(-1j*omega[n]*T) + orig[n].conj() * exp(1j*omega[n]*T) \
    #           for n in range(omega_max))

    #imagtime = 1/beta2 * \
    #           sum(\
    #               (orig[n] + 1/(1j*omega[n])) * exp(+1j*omega[n]*T) \
    #            + ((orig[n] + 1/(1j*omega[n])) * exp(+1j*omega[n]*T)).conj() \
    #           for n in range(omega_max))
    #imagtime -= 1/2.

    #imagtime = 1/beta2 * \
    #           sum(\
    #               (orig[n]) * exp(+1j*omega[n]*T) \
    #            + ((orig[n]) * exp(+1j*omega[n]*T)).conj() \
    #           for n in range(omega_max))
    
    #imagtime = 1/beta2 * \
    #           sum(\
    #               (orig[n]) * exp(-1j*omega[n]*T) \
    #            + ((orig[n]) * exp(-1j*omega[n]*T)).conj() \
    #           for n in range(omega_max))
    
    #imagtime = 1/beta * \
    #           sum(\
    #               (orig[n] + 1/(1j*omega[n])) * exp(-1j*omega[n]*T) \
    #            + ((orig[n] + 1/(1j*omega[n])) * exp(-1j*omega[n]*T)).conj() \
    #           for n in range(len(omega)))
    #imagtime += 1/2.

    imagtime = 1./beta * \
               sum(\
                   (orig[n] - leading_order_coeff/(1j*omega[n])) * exp(-1j*omega[n]*T) \
                + ((orig[n] - leading_order_coeff/(1j*omega[n])) * exp(-1j*omega[n]*T)).conj() \
               for n in range(len(omega)))
    imagtime -= leading_order_coeff/2.

    return imagtime

def FreqToImagTimeComplex(omega,orig,T,beta,leading_order_coeff=1.):
    from numpy import exp
    imagtime = 1./beta * \
               sum((orig[n] - leading_order_coeff/(1j*omega[n])) * exp(-1j*omega[n]*T) \
               for n in range(len(omega)))
    imagtime -= leading_order_coeff/2.

    return imagtime

def ImagTimeToFreq(T,orig,omega,beta):
    '''Fourier transform the function ``orig`` on the imaginary time grid ``T`` 
    to Matsubara frequencies that are given by the :math:`\omega_n` values of 
    ``omega`` (i.e. these are real). The inverse temperature ``beta`` is a 
    required argument.
    '''
    from numpy import exp,linspace
    from scipy.integrate import simps
    from scipy.interpolate import UnivariateSpline
    interp = UnivariateSpline(T,orig,s=0)
    intT = linspace(0,beta,20000) # Works for a orig=1/2. test up to omega_max=5000
    #from pylab import plot
    #plot(intT,interp(intT),label='interped')
    freq = array([simps((interp(intT) - 1/2.)*exp(1j*omega_n*intT),intT) for omega_n in omega])
    #freq = array([simps((orig - 1/2.)*exp(1j*omega_n*T),T) for omega_n in omega])
    freq -= 1/(1j*omega)
    return freq

def ImagTimeToFreqUnitary(T,orig,omega,beta):
    '''Perform the Fourier transformation using the exact unitary opposite of 
    the forwards transformation (faster and should be more reliable).
    Note: This will not work with non-linear spacing of T.'''
    from numpy import exp
    #freq = sum((orig[i] - 1/2.) * exp(1j*omega*T[i]) for i in range(len(T)))
    freq = sum(orig[i] * exp(1j*omega*T[i]) for i in range(len(T)))
    #freq -= 1./(1j*omega)
    return freq

def GetT(numT,beta,unevenT=False):
    '''Return the imaginary time grid with ``numT`` points for an inverse 
    temperature of ``beta``.

    If ``unevenT`` is ``False`` then this grid is simply linearly spaced 
    between 0 and ``beta``. For ``unevenT = True`` however, the grid is 
    logarithmically spaced and symmetric about ``beta/2``.
    '''
    from numpy import linspace,log10,logspace,r_
    if unevenT:
        assert numT%2 == 0
        c = 0.5
        temp = logspace(log10(c),log10(beta/2.+c),numT/2 + 1)
        temp -= c
        temp = temp[:-1]
        temp = r_[temp, beta - temp[::-1]]
        return temp
    else:
        return linspace(0,beta,numT)

def OmegaN(numw,beta):
    '''Return a vector of the real value of :math:`\omega_n` for the 
    first ``numw`` Matsubara frequencies at an inverse temperature of ``beta``.
    '''
    from numpy import array,pi
    return array([(2*n+1)*pi/beta for n in range(numw)])


def WeissDiag(omega,mu,V,eps):
    '''This function calculates the inverse Weiss Green's function for the 
    Anderson impurity model using the given ``V`` and ``eps``. The inverse 
    temperature is ``beta`` and the chemical potential ``mu``. Note that this 
    should always be inverted before use.'''
    return 1j*omega + mu - sum(abs(V[l])**2 / (1j*omega - eps[l]) for l in range(len(V)))
