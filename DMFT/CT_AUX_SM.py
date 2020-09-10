r'''
This module interfaces to the C++ code for the CT-AUX solver, that includes possible 
spin-mixing processes. In addition, some DMFT routines are included.

By default, the code will look in the following path for the C++ program::

    ~/work3/ccode/Michael_TISM_CT/Spin-mixin/danny_interface

If one would like to specify a different path, then one can modify the global variable 
of the module ``program_path`` to point to a different location, or one can set the 
environment variable ``CTAUX_CODE_PATH`` before starting python.

By default, all the output from the CT-AUX code is suppressed so that the messages 
from the python code can be identified. If one wants to make this available, set the 
global variable ``global_print_output = True``.
'''

from __future__ import print_function
from std_imports import *

import os
program_path = os.environ.get('CTAUX_CODE_PATH',None)
if program_path == None:
    program_path = '~/work3/ccode/Michael_TISM_CT/Spin-mixin/danny_interface'
program_path = os.path.expanduser(program_path)
global_print_output = False

def CT_AUX_SM(U,beta,weiss_input,num_sweeps,use_tmpdir=True,combine_Gw=False,K=5.):
    r'''Interface with the C++ code to perform the CT-AUX calculation.

    ``U`` and ``beta`` are the interaction and inverse temperature 
    respectively. The hybridisation is speficied through the Weiss Green's 
    functions in ``weiss_input``.  This should be a tuple of three functions, 
    ``(weiss_up,weiss_down,weiss_cross)``. 

    Note that the CT-AUX solver requires a shift of the global chemical 
    potential by U/2. However, as this solver does not take a parameter ``mu``, 
    it is necessary to pass an appropriate ``weiss_input`` that takes this into 
    account.

    The number of sweeps is given by ``num_sweeps`` and the ``K`` parameter 
    influences the expansion order of the solver, which can improve the speed 
    of the solver in the case of low interaction. However, the default value of 
    ``K=5`` is recommended.

    The option ``combine_Gw`` exists to allow a mixing of the imaginary time 
    and Matsubara frequency measurements. However, it is recommended to leave 
    this off, as the Matsubara frequency measurements currently behave well for 
    all frequencies.

    By default, this function will make a temporary directory to run in, so 
    that no pollution of working directories occurs. This can be disabled with 
    the ``use_tmpdir`` option.

    As it is theoretically possible in the spin-mixing case, for the sign 
    problem to arise, this function raises an exception if the sign difference 
    to 1 becomes great than 0.01.

    **Returns:** ``G,GT,pert_order,dbl_occ,WT,G_orig,Gw``

        ``G`` (tuple of three complex double arrays)
            The mixed Matsubara frequency Green's functions (equiv to ``Gw`` with ``combine_Gw==False``).
        ``GT`` (tuple of three complex double arrays)
            The imaginary time Green's functions.
        ``pert_order`` (double)
            The average perturbation order of the CT-AUX.
        ``dbl_occ`` (double)
            The double occupancy calculated from the filling and the perturbation order.
        ``WT`` (tuple of three complex double arrays)
            The imaginary time Weiss Green's functions.
        ``G_orig`` (tuple of three complex double arrays)
            The Matsubara Green's functions Fourier transformed from the imaginary time outputs.
        ``Gw`` (tuple of three complex double arrays)
            The Matsubara Green's functions measurements.
    '''

    import os

    from tempfile import mkdtemp
    cwd = os.getcwd()
    if use_tmpdir:
        tmpdir = mkdtemp()
        #print("Using temp directory " + tmpdir)
        os.chdir(tmpdir)

    try:
        g0_up,g0_down,f0 = weiss_input
        # Convert to the unitless Green's functions (also creating a new copy at 
        # the same time)
        g0_up = g0_up / beta
        g0_down = g0_down / beta
        f0 = f0 / beta
        assert len(g0_up) == len(g0_down) == len(f0)
        assert g0_up.dtype == g0_down.dtype == f0.dtype == complex
        # The divide 2 is here because of pos/neg frequencies.
        assert len(g0_up) % 2 == 0
        num_omega = len(g0_up) / 2

        args = [program_path, '-', '-', repr(U*beta), repr(num_omega), repr(num_sweeps), '--binary','--K',str(K)]

        import subprocess
        proc = subprocess.Popen(args,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        g0_up.tofile(proc.stdin)
        g0_down.tofile(proc.stdin)
        f0.tofile(proc.stdin)
        proc.stdin.close()

        # Wait for the signal text
        summary = False
        summary_text = ''
        print_text = False
        if global_print_output:
            print_text = True
        while True:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() != None:
                    stderr = proc.stderr.read()
                    logger.info('STDERR: {0}'.format(stderr))
                    raise Exception("Process finished early")
                import time
                time.sleep(1)
            if print_text:
                #print(line.strip())
                logger.info(line.strip())
            if summary:
                summary_text += line
            if line == 'Writing data\n':
                break
            elif line == 'Writing summary\n':
                summary = True
                print_text = True
        else:
            raise Exception("Did not find sentinel string")

        from numpy import fromfile
        pert_order = fromfile(proc.stdout,'float64',1)[0]
        sign = fromfile(proc.stdout,'complex128',1)[0]
        G_up = fromfile(proc.stdout,'complex128',num_omega*2)
        G_down = fromfile(proc.stdout,'complex128',num_omega*2)
        F = fromfile(proc.stdout,'complex128',num_omega*2)
        Gw_up = fromfile(proc.stdout,'complex128',num_omega*2)
        Gw_down = fromfile(proc.stdout,'complex128',num_omega*2)
        Fw = fromfile(proc.stdout,'complex128',num_omega*2)
        # Convert back from the unitless Green's functions
        G_up *= beta
        G_down *= beta
        F *= beta
        Gw_up *= beta
        Gw_down *= beta
        Fw *= beta

        # Also grab the imaginary time functions
        from numpy import genfromtxt
        GT_up = -genfromtxt('GreenTauup0.txt')
        GT_up = GT_up[:,0] + 1j*GT_up[:,1]
        GT_down = -genfromtxt('GreenTaudown0.txt')
        GT_down = GT_down[:,0] + 1j*GT_down[:,1]
        FT = -genfromtxt('FTau0.txt')
        FT = FT[:,0] + 1j*FT[:,1]
        WT = -genfromtxt('nigreen.txt')
        WT = WT[:,0] + 1j*WT[:,1]
    finally:
        if use_tmpdir:
            for filename in ["FTau0.txt","GreenTaudown0.txt","GreenTauup0.txt","nigreen.txt"]:
                if os.path.exists(filename):
                    os.unlink(filename)
                    pass
            os.chdir(cwd)
            os.rmdir(tmpdir)

    N_up = -GT_up[-1].real
    N_down = -GT_down[-1].real
    #logger.info("Pert order is {0}".format(pert_order))
    dbl_occ = (K - pert_order)/float(beta*U) + (N_up + N_down) / 2.
    #print(dbl_occ,pert_order,beta,U,N_up,N_down)

    #logger.info("WT[-1] is {0}, N_up,N_down is {1},{2}".format(WT[-1],N_up,N_down))

    G_orig = array([G_up.copy(),G_down.copy(),F.copy()])

    if combine_Gw:
        # Combining the best of both measurements.
        # This is a bit of an abitrary choice here.
        mix = 10
        G_up[mix:] = Gw_up[mix:]
        G_down[mix:] = Gw_down[mix:]
        F[mix:] = Fw[mix:]
    else:
        G_up = Gw_up
        G_down = Gw_down
        F = Fw

    # Check for sign.
    # If this ever becomes different to one, then the solver must be adjusted 
    # in its sampling procedure.
    if abs(sign - 1.) > 0.01:
        #raise Exception("Sign of solver was definitely not one! Sign = {0}".format(sign))
        logger.warning("Sign of solver was definitely not one! Sign = {0}".format(sign))
    #logger.info("Sign was {0}".format(sign))

    from numpy import c_,conj

    # Check the real nature of the up and down Green's functions
    if avg(abs(G_up[:num_omega] - conj(G_up[num_omega:][::-1]))) > 0.01 or \
       avg(abs(G_down[:num_omega] - conj(G_down[num_omega:][::-1]))) > 0.01:
        #raise Exception("Up or down Green's function was no longer purely real")
        logger.warning("Up or down Green's function was no longer purely real")
    if avg(abs(Gw_up[:num_omega] - conj(Gw_up[num_omega:][::-1]))) > 0.01 or \
       avg(abs(Gw_down[:num_omega] - conj(Gw_down[num_omega:][::-1]))) > 0.01:
        #raise Exception("Up or down Green's function was no longer purely real")
        logger.warning("Up or down Green's function was no longer purely real")

    # Force the realness to be exact.
    G_up[:num_omega] = conj(G_up[num_omega:][::-1])
    G_down[:num_omega] = conj(G_down[num_omega:][::-1])
    Gw_up[:num_omega] = conj(Gw_up[num_omega:][::-1])
    Gw_down[:num_omega] = conj(Gw_down[num_omega:][::-1])

    return array([G_up,G_down,F]),array([GT_up,GT_down,FT]),pert_order,dbl_occ,WT,G_orig,array([Gw_up,Gw_down,Fw])

def CT_AUX_SM_Parallel(*args,**kwds):
    '''An extension of the :func:`CT_AUX_SM` to run the solver over many cores. 
    
    All parameters are the same as :func:`CT_AUX_SM`. The iterations will be split evenly over the different cores.

    **Returns:**

        Same as :func:`CT_AUX_SM`.

    '''
    
    count = kwds.pop('force_proc_count',False)

    from dancommon import CreatePool
    num,pool = CreatePool(count)

    # Divide the sweeps evenly between the processors
    if 'num_sweeps' in kwds:
        kwds['num_sweeps'] /= num
    elif len(args) >= 4:
        #args[3] /= num
        # Tuples are a pain in the ass
        args = args[:3] + (args[3]/num,) + args[4:]

    res = []
    for i in range(num):
        res += [pool.apply_async(CT_AUX_SM,args,kwds)]
    for i in range(num):
        res[i] = res[i].get()
        #print("Done process",i)

    pool.close()
    pool.join()

    # Average everything
    output = []
    for i in range(len(res[0])):
        output += [avg(z[i] for z in res)]

    return output

def DMFT(U,mu,beta,num_sweeps=1e7,Bethe_eqn=False,fit_large_SE=True,allow_AF=True,DoS='Bethe',max_iters=20,K=5.):
    '''Perform a basic DMFT iteration, which considers only a single unit cell.

    Given the parameters ``U,mu,beta``, the self-consistency starts from a 
    random Weiss Green's function and runs for ``max_iters`` with 
    ``num_sweeps``, and ``K`` passed to :func:`CT_AUX_SM_Parallel`. The type of 
    density of states can be chosen with ``DoS`` to be one of:

        * 'Bethe'
        * 'square'
        * 'cubic'

    In the case that ``DoS=='Bethe'`` then specifying ``Bethe_eqn=True`` will 
    use the Bethe equation, rather than the Hilbert transform of the DoS.

    ``allow_AF`` allows for asymmetric up and down components, but this does 
    not correctly lead to AF order. Please use :func:`DMFT_TwoSubLattice` 
    instead.

    **Returns:** ``weiss_w,greens_w,GT,dbl_occ,SE``
    '''
    from numpy import array,linspace,sqrt,pi,trapz,r_

    num_omega = 200
    mu = mu - U/2.

    import ImagConv
    omega = ImagConv.OmegaN(num_omega,beta)
    #omega = ImagConv.OmegaN(num_omega,1.)
    omega = r_[-omega[::-1],omega]

    #V,eps = [1,-1,10],[1,-3,5]
    V,eps = [1,-1,1],[1,-1,1]
    #V,eps = [0,0,0],[0,0,0]
    weiss_w = 1./ImagConv.WeissDiag(omega,mu,V,eps)
    #weiss_w = 1./(1j*omega + mu)
    #weiss_w = (weiss_w,weiss_w,weiss_w*0)
    weiss_w = [weiss_w,weiss_w]

    initial = weiss_w

    file = open('info.dat','wb')
    file.write('Info and stuff\n')
    for key in ['U','mu','beta','num_sweeps','Bethe_eqn','fit_large_SE','allow_AF','DoS','max_iters','K']:
        file.write('{0} {1}\n'.format(key,locals()[key]))
    #file.write('U {0}\n'.format(U))
    #file.write('mu(adjusted) {0}\n'.format(mu))
    #file.write('beta {0}\n'.format(beta))
    #file.write('num_sweeps {0}\n'.format(num_sweeps))
    #file.write('allow_AF {0}\n'.format(allow_AF))
    #file.write('fit_large_SE {0}\n'
    #file.write('DoS {0}\n'.format(DoS))
    file.write('-------------\n')
    file.write('iter Nf dbl_occ pert_order\n')
    file.close()

    old = None
    for i in range(max_iters):
        #weiss_w = real(weiss_w)
        Gw,GT,pert_order,dbl_occ,WT,Gworig,Gwmatsu = CT_AUX_SM_Parallel(U,beta,(weiss_w[0],weiss_w[1],0*weiss_w[0]),num_sweeps,K=K)
        # Use Matsu measurements
        #Gw = Gw2

        new = (GT[0]+GT[1])/2.
        if old != None:
            from numpy import trapz
            logger.info('error is ' + str(trapz(abs(new-old))))

        old = new

        #Nf = -new[-1]
        Nf_up = -GT[0][-1]
        Nf_down = -GT[1][-1]
        #logger.info('new Nf is '+str(1 - new[0]))
        #logger.info('new Nf is '+str(new[-1]))
        logger.info('new Nf_up is '+str(Nf_up))
        logger.info('new Nf_down is '+str(Nf_down))
        logger.info('dbl_occ is '+str(dbl_occ))

        if allow_AF:
            greens_w = Gw[:2]
        else:
            greens_w = (Gw[0] + Gw[1])/2.
            greens_w = (greens_w,greens_w)

        #greens_w = -greens_w.conj()

        #if i % 2 == 0:
        #if True:
        if False:
            import pylab
            pylab.figure()
            pylab.plot(GT[0],marker='.')
            pylab.figure()
            pylab.plot(omega,weiss_w)
            pylab.plot(omega,greens_w)
            pylab.figure()
            pylab.plot(omega,weiss_w.imag)
            pylab.plot(omega,greens_w.imag)
            #return

        if False:
            greens_w = weiss_w

        if Bethe_eqn:
            for spin in range(2):
                if allow_AF:
                    # Allowing for AF order here is simple.
                    weiss_w[spin] = 1./(1j*omega + mu - greens_w[1-spin])
                else:
                    weiss_w[spin] = 1./(1j*omega + mu - greens_w[spin])
                    #weiss_w = 1./(-1j*omega - mu - greens_w)
        else:
            SE = [None,None]
            for spin in range(2):
                SE[spin] = 1./weiss_w[spin] - 1./greens_w[spin]

                if fit_large_SE:
                    # Fit only past omega = 20
                    fit_ind = omega > 20
                    # Use formula from Gull et al review
                    # Note that Nf must be of the opposite spin
                    if spin == 0:
                        Nf = Nf_down
                    else:
                        Nf = Nf_up
                    SE[spin][fit_ind] = U*(Nf-0.5) + U**2/(1j*omega[fit_ind]) * Nf*(1-Nf)

            for spin in range(2):
                t = 1.
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
                elif DoS == 'square':
                    from numpy import load
                    from scipy.interpolate import interp1d
                    import os
                    data = load(os.path.expanduser('~/work3/python/square_dos_norm.npz'))
                    density = interp1d(data['omega'],data['dos'],bounds_error=False,fill_value=0.)
                    E = linspace(2*-2*t,2*2*t,num_eps)

                if allow_AF:
                    from numpy import zeros
                    new_greens_w = zeros(len(omega),complex)
                    for n in range(len(omega)):
                        zeta_up = 1j*omega[n] + mu - SE[spin][n]
                        zeta_down = 1j*omega[n] + mu - SE[1-spin][n]
                        new_greens_w[n] = zeta_down * trapz(density(E) / (zeta_up*zeta_down - E**2),E)
                else:
                    new_greens_w = array([trapz(density(E) / (1j*omega[n] + mu - E - SE[spin][n]),E) for n in range(len(omega))])

                weiss_w[spin] = 1./(1./new_greens_w + SE[spin])

            if fit_large_SE:
                weiss_w[spin] = weiss_w[spin].imag * 1j

        # Save data each iter
        from numpy import savez
        if Bethe_eqn:
            savez('data_iter{0}.npz'.format(i),weiss=weiss_w,greens=greens_w,GT=GT,dbl_occ=dbl_occ,Gwmatsu=Gwmatsu,Gworig=Gworig)
        else:
            savez('data_iter{0}.npz'.format(i),weiss=weiss_w,greens=greens_w,GT=GT,dbl_occ=dbl_occ,SE=SE,Gworig=Gworig,Gwmatsu=Gwmatsu)
        
        file = open('info.dat','ab')
        file.write('{i} {Nf_up} {Nf_down} {dbl_occ} {pert_order}\n'.format(**locals()))
        file.close()

    if Bethe_eqn:
        return weiss_w,greens_w,GT,dbl_occ
    else:
        return weiss_w,greens_w,GT,dbl_occ,SE

def LoadAverage(first_iter,last_iter):
    import numpy
    list = []
    for i in irange(first_iter,last_iter):
        list += [numpy.load('data_iter{0}.npz'.format(i))]

    GT = avg(z['GT'] for z in list)
    weiss = avg(z['weiss'] for z in list)
    dbl_occ = avg(z['dbl_occ'] for z in list)
    greens = avg(z['greens'] for z in list)
    if 'SE' in list[0].files:
        SE = avg(z['SE'] for z in list)
        return greens,weiss,dbl_occ,GT,SE
    else:
        return greens,weiss,dbl_occ,GT

def CalcAllKEs(first_iter,last_iter,beta):
    import numpy
    KE = []
    for i in irange(first_iter,last_iter):
        SE = numpy.load('data_iter{0}.npz'.format(i))['SE']
        import SS_CT
        KE += [SS_CT.CalcKineticEnergy2(SE,beta)]
        print("KE for iter {0} is {1}".format(i,KE[-1]))

    return KE

def IntEnergy(greens,SE,beta,eta=0.01):
    from numpy import exp
    import ImagConv
    omega = ImagConv.OmegaN(len(SE),beta)
    return 1./beta * sum(greens*SE*exp(1j*omega*eta)).real * 2





def DMFT_TwoSubLattice(U,mu,beta,num_sweeps=1e7,Bethe_eqn=False,fit_large_SE=True,DoS='Bethe',max_iters=20):
    '''This is a two sub-lattice version of :func:`DMFT`. The parameters are 
    identical to that function.

    This function differs in that it doesn't return any values, but instead 
    saves each iteration to a file ``data_iter<iter>.npz``. See 
    :mod:`RDMFT_TISM` for how to open this type of file.
    '''
    from numpy import array,linspace,sqrt,pi,trapz,zeros,ones,r_

    num_omega = 200
    mu = mu - U/2.

    import ImagConv
    omega = ImagConv.OmegaN(num_omega,beta)
    #omega = ImagConv.OmegaN(num_omega,1.)
    omega = r_[-omega[::-1],omega]

    # This setup of the Weiss functions is no longer used, because the SE is 
    # taken preferentially in the sub-lattice structure.
    #V,eps = [1,-1,10],[1,-3,5]
    V,eps = [1,-1,1],[1,-1,1]
    #V,eps = [0,0,0],[0,0,0]
    weiss_w1 = 1./ImagConv.WeissDiag(omega,mu,V,eps)
    V,eps = [1,-3,2],[2,-1,1]
    weiss_w2 = 1./ImagConv.WeissDiag(omega,mu,V,eps)
    #weiss_w = 1./(1j*omega + mu)
    #weiss_w = (weiss_w,weiss_w,weiss_w*0)
    weiss_w_A = [weiss_w1,weiss_w2]
    weiss_w_B = [weiss_w2,weiss_w1]
    #SE_A = [zeros(len(omega),complex),zeros(len(omega),complex)]
    #SE_B = [zeros(len(omega),complex),zeros(len(omega),complex)]
    SE_A = [ones(len(omega),complex)*0.1,ones(len(omega),complex)*0.2]
    SE_B = [ones(len(omega),complex)*0.2,ones(len(omega),complex)*0.1]
    greens_w = [None,None]

    file = open('info.dat','wb')
    file.write('Info and stuff for sublattice 2\n')
    for key in ['U','mu','beta','num_sweeps','Bethe_eqn','fit_large_SE','DoS','max_iters']:
        file.write('{0} {1}\n'.format(key,locals()[key]))
    file.write('-------------\n')
    file.write('iter Nf dbl_occ pert_order\n')
    file.close()

    old = None
    for i in range(max_iters):
        if i % 2 == 0:
            cur_weiss = weiss_w_A
            other_weiss = weiss_w_B
            cur_SE = SE_A
            other_SE = SE_B
        else:
            cur_weiss = weiss_w_B
            other_weiss = weiss_w_A
            cur_SE = SE_B
            other_SE = SE_A

        for spin in range(2):
            t = 1.
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
            elif DoS == 'square':
                from numpy import load
                from scipy.interpolate import interp1d
                import os
                data = load(os.path.expanduser('~/work3/python/square_dos_norm.npz'))
                density = interp1d(data['omega'],data['dos'],bounds_error=False,fill_value=0.)
                E = linspace(2*-2*t,2*2*t,num_eps)

            greens_w[spin] = zeros(len(omega),complex)
            for n in range(len(omega)):
                cur_zeta = 1j*omega[n] + mu - cur_SE[spin][n]
                other_zeta = 1j*omega[n] + mu - other_SE[spin][n]
                greens_w[spin][n] = other_zeta * trapz(density(E) / (cur_zeta*other_zeta - E**2),E)

            cur_weiss[spin] = 1./(1./greens_w[spin] + cur_SE[spin])

        Gw,GT,pert_order,dbl_occ,WT,Gworig,Gwmatsu = CT_AUX_SM_Parallel(U,beta,(cur_weiss[0],cur_weiss[1],0*cur_weiss[0]),num_sweeps)

        for spin in [0,1]:
            cur_SE[spin] = 1./cur_weiss[spin] - 1./Gw[spin]

        new = (GT[0]+GT[1])/2.
        if old != None:
            from numpy import trapz
            logger.info('error is ' + str(trapz(abs(new-old))))

        old = new

        Nf_up = -GT[0][-1]
        Nf_down = -GT[1][-1]
        #logger.info('new Nf is '+str(1 - new[0]))
        #logger.info('new Nf is '+str(new[-1]))
        logger.info('iteration is {0}'.format('A' if i % 2 == 0 else 'B'))
        logger.info('new Nf_up is '+str(-GT[0][-1]))
        logger.info('new Nf_down is '+str(-GT[1][-1]))
        logger.info('dbl_occ is '+str(dbl_occ))

        greens_w = Gw[:2]

        # Save data each iter
        from numpy import savez
        if Bethe_eqn:
            savez('data_iter{0}.npz'.format(i),weiss=cur_weiss,greens=greens_w,GT=GT,dbl_occ=dbl_occ)
        else:
            savez('data_iter{0}.npz'.format(i),weiss=cur_weiss,greens=greens_w,GT=GT,dbl_occ=dbl_occ,SE=(SE_A,SE_B),iteration=i)
        
        file = open('info.dat','ab')
        file.write('{i} {Nf_up} {Nf_down} {dbl_occ} {pert_order}\n'.format(**locals()))
        file.close()

    #if Bethe_eqn:
    #    return weiss_w,greens_w,GT,dbl_occ
    #else:
    #    return weiss_w,greens_w,GT,dbl_occ,SE
    return

def CompareWithED(U=1.,beta=1.,num_omega=100,show_figures=True,num_sweeps=1e5):
    '''Perform a comparison with the ED solver given a set of Anderson 
    parameters. For large enough sweeps, these values should agree exactly.
    '''

    import ED_new
    import ImagConv
    from numpy import array,r_

    num_l = 3
    mu = 0.

    omega = ImagConv.OmegaN(num_omega,beta)*1j
    omega = r_[-omega[::-1],omega]

    ed = ED_new.ED_SM(num_l,U,mu,beta)
    ed.Vup = [1.,2.3,5.]
    ed.Vdown = [2.,0.1,3.]
    ed.eps = [4.,2.,5.]
    ed.Wup = [1j+2.,5j+3.,1.]
    ed.Wdown = [2j+3.,1j+9.,3.]
    #ed.Vup = [1.,2.,3.]
    #ed.Vdown = [1.,2.,3.]
    #ed.eps = [3.,2.,1.]
    #ed.Wup = [0,0,0]
    #ed.Wdown = [0,0,0]

    weiss = array(ED_new.Weiss(omega,mu-U/2.,ed))

    G_ED = ed.CalcGreens(omega)

    global global_print_output
    global_print_output = True
    G_CT,GT,pert_order,dbl_occ,WT,G_orig,Gw = CT_AUX_SM(U,beta,weiss,num_sweeps)

    print("GT max imag: ",max(abs(GT.flatten().imag)))
    print("GT avg imag: ",avg(abs(GT.flatten().imag)))

    import pylab
    G_CT = array(G_CT)
    G_ED = array(G_ED)

    err = abs(G_CT.flatten() - G_ED.flatten())
    print("Worst error: ",max(err))
    print("Average error: ",avg(err))

    if show_figures:
        #pylab.figure()
        #pylab.plot(weiss.T.imag)
        pylab.figure()
        pylab.plot(G_ED.T.imag)
        pylab.plot(G_CT.T.imag,'.')
        pylab.savefig('EDCT_compare.png')

    return G_ED,G_CT

def CompareWithOldCT(U=1.,beta=1.,num_omega=100,show_figures=True):
    '''This code is used to compare the solver to Danny's implementation in C++ 
    for the case of no spin-mixing.'''

    # Only using ED_new here to get a test Weiss function
    import ED_new
    import ImagConv
    from numpy import array,r_

    num_l = 3
    mu = 0.

    num_sweeps = 1e5

    omega_half = ImagConv.OmegaN(num_omega,beta)*1j
    omega_full = r_[-omega_half[::-1],omega_half]
    T = ImagConv.GetT(num_omega*2+1,beta)

    ed = ED_new.ED_SM(num_l,U,mu,beta)
    ed.Vup = [1.,2.,3.]
    ed.Vdown = [1.,2.,3.]
    ed.eps = [3.,2.,1.]
    ed.Wup = [0,0,0]
    ed.Wdown = [0,0,0]

    weiss = array(ED_new.Weiss(omega_full,mu-U/2.,ed))

    global global_print_output
    global_print_output = True
    Gw_new,GT_new,pert_order,dbl_occ,WT,G_orig,Gw = CT_AUX_SM(U,beta,weiss,num_sweeps)

    weiss_w = [weiss[0][num_omega:],weiss[1][num_omega:]]
    weiss_input = [ImagConv.FreqToImagTime(omega_half.imag,weiss_w[0],T,beta),
                   ImagConv.FreqToImagTime(omega_half.imag,weiss_w[1],T,beta)]

    import CT_interface
    CT_interface.global_print_output = True
    GT_old,Gw_old,covar,avg_order,avg_sign = CT_interface.EquilibriumIter(U,beta,weiss_input,max_sweeps=num_sweeps,weiss_w=weiss_w,num_omega=num_omega,global_updates=True)

    GT_diff = GT_new[0] - GT_old[0]

    print("GT max: ",max(abs(GT_diff)))
    print("GT avg: ",avg(abs(GT_diff)))

    import pylab
    Gw_diff = Gw_new[0][num_omega:] - Gw_old[0]

    print("Worst error: ",max(abs(Gw_diff)))
    print("Average error: ",avg(abs(Gw_diff)))

    if show_figures:
        #pylab.figure()
        #pylab.plot(weiss.T.imag)
        pylab.figure()
        pylab.plot(omega_full.imag,Gw_new[0].imag)
        pylab.plot(omega_half.imag,Gw_old[0].imag)
        pylab.savefig('CTCT_compare_Gw.png')
        pylab.figure()
        pylab.plot(T,GT_new[0].real,'.')
        pylab.plot(T,GT_old[0].real,'.')
        pylab.savefig('CTCT_compare_GT.png')

    return GT_old,GT_new,Gw_old,Gw_new
