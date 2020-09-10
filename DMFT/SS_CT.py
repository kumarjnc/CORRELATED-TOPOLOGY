'''
This module implements DMFT for the CT-AUX solver with some various support functions for analysis of the results.
'''
from __future__ import print_function
from std_imports import *

def MonteCarloComplete(U,mu,beta,K=5.,delta_tol=1e-6,init_weiss_w=None,get_covar=True,numT=400,numw=400,init_weiss_T=None,max_iters=20,num_sweeps=1e7,match_weiss=False,allow_AF=False,fit_large_SE=False,global_updates=True,t=1.,unevenT=False,DoS='Bethe'):
    r'''This is function that performs the DMFT self-consistency loops.

    The main arguments are given by ``U``, ``mu`` and ``beta``. The initial 
    state may be initialised by either a Matsubara frequency Green's function 
    (``init_weiss_w``) or an imaginary time function (``init_weiss_T``). In 
    either case, the size of the imaginary time grid and the number of 
    frequencies are given by ``numT`` and ``numw`` respectively.

    There is no convergency check performed here, and the code simply runs for 
    ``max_iters`` (an obsolete variable ``delta_tol`` originally provided this 
    functionality). Information is saved to a file ``MC_info.dat`` in the 
    current directory.

    To allow for antiferromagnetic order, enable ``allow_AF``. To fit the 
    self-energy to its analytical asymptotic form, enable ``fit_large_SE`` 
    (this is currently fixed to fit values past :math:`\omega_n > 30`). The 
    density of states used can be changed with ``DoS``, which can be set to 
    'Bethe', 'cubic' or 'square' (note that the cubic and square densities are 
    simply read from a file).

    Other variables that are specific to the CT-AUX code are: ``get_covar``, ``num_sweeps``, ``global_updates``, ``unevenT``, ``t``, ``K``. The documentation for these is described in the CT-AUX C++ code.

    **Returns:** ``weiss_w,self_energy,(delta_avg,delta_min,delta_max),weiss_T,covar,GT,T``

        ``weiss_w`` (complex vector)
            The weiss Green's function in Matsubara frequency.
        ``self_energy`` (complex vector)
            The self-energy in Matsubara frequency.
        ``delta_avg, delta_min, delta_max`` (reals)
            Relative difference of the Weiss Green's functions of the last two 
            iterations (average, minimum and maximum respectively).
        ``weiss_T`` (real vector)
            The Weiss Green's function in imaginary time.
        ``covar`` (real matrxi)
            The covariance matrix of the Green's function.
        ``GT`` (real vector)
            The Green's function in imaginary time.
        ``T`` (real vector)
            The imaginary time grid used.
    '''

    mu = mu - U/2.
    from numpy import ones_like

    import ImagConv
    T = ImagConv.GetT(numT,beta,unevenT)
    omega = ImagConv.OmegaN(numw,beta)

    if fit_large_SE:
        # Make sure we are using enough points
        while omega[-1] < 400.:
            numw += 100
            omega = ImagConv.OmegaN(numw,beta)

    #weiss_input = ones_like(T)
    if init_weiss_T == None:
        if init_weiss_w == None:
            V,eps = [1,-1,1],[1,-1,1]
            cur_weiss_w = 1./ImagConv.WeissDiag(omega,mu,V,eps)
            cur_weiss_T = ImagConv.FreqToImagTime(omega,cur_weiss_w,T,beta)

            cur_weiss_w = [cur_weiss_w,cur_weiss_w]
            cur_weiss_T = [cur_weiss_T,cur_weiss_T]
        else:
            cur_weiss_w = init_weiss_w
            cur_weiss_T = [ImagConv.FreqToImagTime(omega,cur_weiss_w[0],T,beta),ImagConv.FreqToImagTime(omega,cur_weiss_w[1],T,beta)]
    else:
        cur_weiss_T = init_weiss_T
        cur_weiss_w = [ImagConv.ImagTimeToFreq(T,cur_weiss_T[0],omega,beta), ImagConv.ImagTimeToFreq(T,cur_weiss_T[1],omega,beta)]

    import os
    info_filename = 'MC_info.dat'
    file = open(info_filename,'w')
    file.write("U {0}\n".format(U))
    file.write("mu {0}\n".format(mu))
    file.write("beta {0}\n".format(beta))
    file.write("K {0}\n".format(K))
    file.write("numT {0}\n".format(numT))
    file.write("numw {0}\n".format(numw))
    file.write("num_sweeps {0}\n".format(num_sweeps))
    file.write("global_updates {0}\n".format(global_updates))
    file.write("t {0}\n".format(t))
    file.write("allow_AF {0}\n".format(allow_AF))
    file.write("fit_large_SE {0}\n".format(fit_large_SE))
    file.write("iter | Nf_up | Greens err | Ndbl | Order | Sign\n")
    file.close()

    greens_T = [None,None]
    for i in range(max_iters):
        try:
            old_weiss = [cur_weiss_T[0].copy(), cur_weiss_T[1].copy()]
            old_greens = greens_T

            from dancommon import CreatePool,FakePool
            pool_size,pool = CreatePool()
            #pool = FakePool()
            indiv_sweeps = num_sweeps / pool_size
            
            from CT_interface import EquilibriumIter
            res = [pool.apply_async(EquilibriumIter,(U,beta,cur_weiss_T),{'max_sweeps':indiv_sweeps,'global_updates':global_updates, 'unevenT':unevenT}) for z in range(pool_size)]
            res = [z.get() for z in res]
            pool.close()
            pool.join()

            greens_T = [avg(z[0][0] for z in res), avg(z[0][1] for z in res)]
            greens_w = [avg(z[1][0] for z in res), avg(z[1][1] for z in res)]
            covar = [avg(z[2][0] for z in res), avg(z[2][1] for z in res)]
            avg_order = avg(z[3] for z in res)
            avg_sign = avg(z[4] for z in res)

            # For comparison of spins even in allow_AF=False case.
            solver_gT = greens_T
            solver_gw = greens_w

            if allow_AF:
                # Switch them around based on spin
                #spins = [ (+1 if z[4] > 0 else -1) for z in res]
                #greens_T = {-1: avg([z[0][-1*spin] for z,spin in zip(*(res,spins))]),
                #            +1: avg([z[0][+1*spin] for z,spin in zip(*(res,spins))])}
                #greens_w = {-1: avg([z[1][-1*spin] for z,spin in zip(*(res,spins))]),
                #            +1: avg([z[1][+1*spin] for z,spin in zip(*(res,spins))])}
                #covar = {-1: avg([z[2][-1*spin] for z,spin in zip(*(res,spins))]),
                #         +1: avg([z[2][+1*spin] for z,spin in zip(*(res,spins))])}
                pass
            else:
                save_greens_T = greens_T
                greens_T = avg(greens_T)
                greens_w = avg(greens_w)
                covar = avg(covar)
                
                greens_T = [greens_T,greens_T]
                greens_w = [greens_w,greens_w]
                covar = [covar,covar]

            self_energy = [None,None]
            for spin in [0,1]:
                # Using the T version only
                greens_w2 = ImagConv.ImagTimeToFreq(T,greens_T[spin],omega,beta)

                #cur_weiss_w[spin] = ImagConv.ImagTimeToFreq(T,cur_weiss_T[spin],omega,beta)
                #self_energy = 1./cur_weiss_w - 1./greens_w
                self_energy[spin] = 1./cur_weiss_w[spin] - 1./greens_w2

                if fit_large_SE:
                    # Fit only past omega = 20
                    fit_ind = omega > 30
                    # Use formula from Gull et al review
                    N = -greens_T[-spin][-1]
                    self_energy[spin][fit_ind] = U*(N-0.5) + U**2/(1j*omega[fit_ind]) * N*(1-N)

            for spin in [0,1]:

                from numpy import empty,trapz,linspace,pi,sqrt
                num_eps = 20000
                if DoS == 'Bethe':
                    density = lambda E: sqrt(4*t**2 - E**2) / (2*pi*t**2)
                    E = linspace(-2*t,2*t,num_eps)
                    #from pylab import plot;plot(E,density(E))
                elif DoS == 'cubic':
                    from numpy import load
                    from scipy.interpolate import interp1d
                    import os
                    data = load(os.path.expanduser('~/work3/python/cubic_dos_norm.npz'))
                    density = interp1d(data['omega'],data['dos'],bounds_error=False,fill_value=0.)
                    E = linspace(3*-2*t,3*2*t,num_eps)
                    #from pylab import plot;plot(E,density(E))

                greens2_diag = empty(omega.shape,complex)
                for n in range(len(omega)):
                    if allow_AF:
                        zeta = 1j*omega[n] + mu - self_energy[spin][n]
                        zeta_opp = 1j*omega[n] + mu - self_energy[1-spin][n]
                        func = zeta_opp * density(E) / (zeta*zeta_opp - E)
                    else:
                        func = density(E) / (1j*omega[n] + mu - E - self_energy[spin][n])
                    #func = density(E) / (-1j*omega[n] - mu + E - self_energy[n])
                    greens2_diag[n] = trapz(func,E)

                cur_weiss_w[spin] = 1./(1./greens2_diag + self_energy[spin])
                # Testing the exact Bethe lattice form
                #cur_weiss_w[spin] = 1./(1j*omega + mu - greens_w2)

                if match_weiss:
                    V,eps = MatchWeiss(cur_weiss_w[spin],max_l=20)
                    matchweiss_test = cur_weiss_w[spin]
                    cur_weiss_w[spin] = -1./WeissDiag(V,eps)
                    diff_from_matchweiss = abs(matchweiss_test - cur_weiss_w[spin])
                    logger.info("Difference in weiss fit: avg:{0}, largest:{1}".format(avg(diff_from_matchweiss),diff_from_matchweiss.max()))

                cur_weiss_T[spin] = ImagConv.FreqToImagTime(omega,cur_weiss_w[spin],T,beta)

            from dancommon import TestConvergence
            delta_avg,delta_min,delta_max,ind = TestConvergence(old_weiss[0],cur_weiss_T[0])
            logger.warning("Weiss avg is {0}, min {1}, max {2} at {3}".format(delta_avg,delta_min,delta_max,ind))
            if i >= 1:
                delta_avg,delta_min,delta_max,ind = TestConvergence(old_greens[0],greens_T[0])
                logger.warning("Greens avg is {0}, min {1}, max {2} at {3}".format(delta_avg,delta_min,delta_max,ind))
    
            N_up = -greens_T[0][-1]
            N_down = -greens_T[1][-1]
            logger.info("Nf is {0},{1}".format(N_up,N_down))
            if U == 0:
                Ndbl = -1.
            else:
                Ndbl = (K - avg_order)/float(beta*U) + (N_up + N_down) / 2.
            logger.info("Ndbl is {0}".format(Ndbl))
            file = open('MC_info.dat','a+')
            file.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(i,greens_T[0][-1],delta_avg,Ndbl,avg_order,avg_sign))
            file.close()
            
            # Output the current info:
            import pickle,gzip
            file = gzip.open('data_iter{0}.pickle.gz'.format(i),'wb')
            #data = (cur_weiss_w,self_energy,(delta_avg,delta_min,delta_max),cur_weiss_T,covar,greens_T,T)
            data = (cur_weiss_w,self_energy,(delta_avg,delta_min,delta_max),cur_weiss_T,covar,solver_gT,T)
            pickle.dump(data,file)
            file.close()

            if delta_avg < delta_tol and delta_max < 10*delta_tol:
                #break
                # Disable breaking for the moment
                pass

        except KeyboardInterrupt:
            print("Keep raising?")
            ans = raw_input()
            if ans in 'yY':
                raise
            else:
                break

    return cur_weiss_w,self_energy,(delta_avg,delta_min,delta_max),cur_weiss_T,covar,greens_T,T


def CalcKineticEnergyFromFile(first_iter,last_iter,beta,U=None):

    data = [DataFromPGZ('data_iter{0}.pickle.gz'.format(i)) for i in range(first_iter,last_iter)]

    #T = data[-1][-1]
    #greens = avg(z[-2][0] for z in data)
    SE = avg(z[1][0] for z in data)

    # Shortening SE for speed - it should make no difference in accuracy
    SE = SE[:200]

    return CalcKineticEnergy2(SE,beta)


def CalcKineticEnergy(SE,beta,U=None):
    r'''An old form to calculate the kinetic energy. Do not use.'''
    from numpy import linspace,sqrt,trapz,pi
    import ImagConv
    omega = ImagConv.OmegaN(len(SE),beta)

    raise Exception("This version of CalcKineticEnergy should no longer be used")

    # Compensate large range omega with asymptotic form (assume half-filling)
    if U != None:
        from numpy import concatenate
        # Extend the SE - but truncate before omega_n = 400
        SE = concatenate((SE,SE))
        SE = concatenate((SE,SE))
        omega = ImagConv.OmegaN(len(SE),beta)
        SE = SE[omega < 400.]
        omega = omega[omega < 400.]
        SE[omega>40.] = -U**2 / omega[omega>40.] * 0.25
        
    t = 1.
    num_eps = 20000
    density = lambda E: sqrt(4*t**2 - E**2) / (2*pi*t**2)
    E = linspace(-2*t,2*t,num_eps)

    T = 1./beta

    Ek = T * sum([trapz(E*density(E) * 1./(1j*omega[n] - E - SE[n]),E) for n in range(len(SE))])

    # Also the opposite sign for omega
    Ek += T * sum([trapz(E*density(E) * 1./(-1j*omega[n] - E - SE[n].conj()),E) for n in range(len(SE))])

    # For both spins
    Ek *= 2

    return Ek

def CalcKineticEnergy2(SE,beta,U=None):
    r'''This function calculates the kinetic energy corresponding to a Hubbard 
    model for a given self-energy ``SE`` at an inverse temperature ``beta``. If 
    the interaction is given in ``U``, then the self-energy is first extended 
    to very large omega to assist in the reaching convergence in the Mastubara 
    sum.

    Note that this function uses the form of the calculation that factorises 
    out the non-interacting part, which is then summed over exactly.

    **Returns:** ``Ek``
    '''
    from numpy import linspace,sqrt,trapz,pi,exp
    import ImagConv
    omega = ImagConv.OmegaN(len(SE),beta)

    # Compensate large range omega with asymptotic form (assume half-filling)
    if U != None:
        from numpy import concatenate
        # Extend the SE - but truncate before omega_n = 400
        SE = concatenate((SE,SE))
        SE = concatenate((SE,SE))
        omega = ImagConv.OmegaN(len(SE),beta)
        SE = SE[omega < 400.]
        omega = omega[omega < 400.]
        SE[omega>40.] = -U**2 / omega[omega>40.] * 0.25
        
    t = 1.
    num_eps = 20000
    density = lambda E: sqrt(4*t**2 - E**2) / (2*pi*t**2)
    E = linspace(-2*t,2*t,num_eps)

    T = 1./beta

    # This calculation method works better with truncated self-energies (i.e.  
    # no extrapolation is required here).
    
    const = trapz(E*density(E) / (exp(beta*E) + 1),E)

    greens_term = T * sum([trapz(E*density(E) * (1./(1j*omega[n] - E - SE[n]) - 1./(1j*omega[n] - E)),E) for n in range(len(SE))])

    # Also the opposite sign for omega
    greens_term += T * sum([trapz(E*density(E) * (1./(-1j*omega[n] - E - SE[n].conj()) - 1./(-1j*omega[n] - E)),E) for n in range(len(SE))])

    Ek = const + greens_term

    # For both spins
    Ek *= 2

    return Ek

def BetheGreensFromSE(omega,SE):
    r'''A convenience function to determine the Green's function from a given 
    self-energy ``SE`` on the given Matsubara frequencies ``omega``.

    Do not use this function - it is implemented more rigourously in another 
    module.
    '''
    from numpy import sqrt,imag,pi,empty,trapz,linspace
    t = 1.
    density = lambda E: sqrt(4*t**2 - E**2) / (2*pi*t**2)
    greens[spin] = empty(omega.shape,complex)
    #import pdb; pdb.set_trace()
    if NRG_broadening == 0:
        raise NotImplementedError
        # Do analytical version
        imag_part = -pi * density(omega)
        greens[spin]
    num_eps = 20000
    E = linspace(-2*t,2*t,num_eps)
    for n in range(len(omega)):
        func = density(E) / (omega[n] + mu - E - SE[spin][n])
        greens[spin][n] = trapz(func,E)
