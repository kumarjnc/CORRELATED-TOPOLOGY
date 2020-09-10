'''
This module provides functions to analytically continue Green's functions (or
other data) given in imaginary time to real-frequency spectra. So far, only a 
Maximum Entropy method is implemented.

**Notes:**

 1. Data must be transformed to imaginary time. That is, Matsubara frequency 
    data cannot be given to the functions.
 2. At the present time this code relies on a couple of other helper libraries 
    for multiprocessing.  If you are to use the code that requires this, please 
    contact Danny.

The major function of interest to the end-user here is :func:`ClassicalMaxEnt`. 
The function :func:`FitAlpha` also describes some of the process behind the 
method.

**Changelog** (partial listing):

    13/10/2011: Implemented the option to increase the number of search vectors 
                used in the algorithm

---------------------------

'''

from __future__ import print_function
from std_imports import *

def IterativeModel(first_iter,last_iter,steps,beta,omega,**keys):
    '''Temporary function to test an iterative model-update for a batched set of 
    data.
    
    There is a danger that this function does not converge when the model 
    becomes too much like the result.
    '''
    Alist = []
    for i in range(5):
        omega,A = BatchedCTData(first_iter,last_iter,steps,beta,omega,**keys)

        keys['model'] = A
        Alist += [A]

    return omega,Alist

def BatchedCTData(first_iter,last_iter,steps,beta,omega,**keys):
    '''Temporary function to continue a set of data individually, and then 
    combined the results. The idea is to smooth over spurious fluctuations in 
    the resultant spectra.
    '''
    from dancommon import CreatePool
    from numpy import sqrt

    import pylab
    pylab.figure()

    data = [DataFromPGZ('data_iter{0}.pickle.gz'.format(i)) for i in range(first_iter,last_iter)]

    out = []
    num,pool = CreatePool()
    for item in data:
        T = item[-1]
        greens = -item[-2][0]
        covar = item[-3][0] / steps

        # Need this for perfect particle-hole symmetric cases.
        T2 = T[::2]
        greens2 = greens[::2]
        covar2 = covar[::2,::2]

        out += [pool.apply_async(ClassicalMaxEnt,(T2,greens2,covar2,beta,omega),keys)]

    out = [z.get() for z in out]
    pool.close()
    pool.join()

    for omega,A,alpha in out:
        pylab.plot(omega,A)
        print("Final alpha:",alpha)


    return omega,avg(z[1] for z in out)

def LoadCTData(first_iter,last_iter,steps):
    r'''Convenience function to load data from the ``*.pickle.gz`` DMFT output.'''
    data = [DataFromPGZ('data_iter{0}.pickle.gz'.format(i)) for i in range(first_iter,last_iter)]
    T = data[-1][-1]
    greens = -avg(z[-2][0] for z in data)
    covar = avg(z[-3][0] for z in data)

    T2 = T[::2]
    greens2 = greens[::2]
    covar2 = covar[::2,::2] / steps
    
    return T2,greens2,covar2

def BryansMaxEnt(T,G,covar,beta,omega,guess=None,**keys):
    r'''Perform the MaxEnt procedure, but instead of finding the spectra that 
    maximises the probability function with respect to one alpha, return the 
    spectra that is averaged from many values of alpha around the peak. I.e. 
    this function returns 

    .. math:: \langle A \rangle = \int d\alpha A_\alpha \mathrm{Pr}[\alpha | \bar{G}]

    where :math:`A_\alpha` is the spectra that maximises the probability :math:`\mathrm{Pr}[A | \bar{G},\alpha]`.

    **Notes**:
     - The name of this function is a bit of a misnomer. The Bryan's method 
       itself refers to a search procedure in maximising the probability 
       function. However, it has also been used to refer to the averaging 
       procedure.
     - Internally this calls :func:`ClassicalMaxEnt` and passes all addition 
       keywords to it.
     - This function is not recommended: the results are nearly always the same 
       to the naked eye as from the plain :func:`ClassicalMaxEnt` call, and take 
       much more time to run.
    '''

    from numpy.linalg import inv
    from numpy import array,exp

    # First get the classical point
    omega,A,alpha_peak = ClassicalMaxEnt(T,G,covar,beta,omega,guess,**keys)

    # Now work around it to determine the structure
    covar_inv = inv(covar)
    omega,A_peak,alpha_peak,res_peak = FitAlpha(T,G,covar_inv,beta,omega,A,alpha_peak,**keys)
    store = [(alpha_peak,res_peak,A)]

    cur = alpha_peak
    min_comp = 0.01
    A = A_peak
    while True:
        cur *= 0.95
        omega,A,alpha,res = FitAlpha(T,G,covar_inv,beta,omega,A_peak,cur,**keys)
        store += [(cur,res,A)]
        print(cur,res)
        if res - res_peak < log(min_comp):
            break
    cur = alpha_peak
    A = A_peak
    while True:
        cur *= 1.05
        omega,A,alpha,res = FitAlpha(T,G,covar_inv,beta,omega,A,cur,**keys)
        store += [(cur,res,A)]
        print(cur,res)
        if res - res_peak < log(min_comp):
            break

    store.sort(key=lambda x: x[0])
    
    alpha = array([z[0] for z in store])
    res = array([z[1] for z in store])
    A = array([z[2] for z in store])

    # Integrate the spectra up (trapezoidal)
    spacing = alpha[1:] - alpha[:-1]
    weight = (spacing[1:] + spacing[:-1])/2
    byrans = sum(exp(res[i+1])*weight[i]*A[i+1] for i in range(len(weight)))
    norm = sum(exp(res[i+1])*weight[i] for i in range(len(weight)))
    byrans /= norm

    return omega,byrans,A_peak

def ClassicalSmoothedMaxEnt(T,G,covar,beta,omega,guess=None,smoothing_factor=1.,**keys):
    '''This function does two normal classical MaxEnt runs with a smoothing step 
    in between. The smoothing is applied as a convolution with a Gaussian of 
    width ``smoothing_factor``.
    
    It turns out that this is not such a good idea. If the smoothing makes a big 
    difference, then in all likelihood the functions have not actually converged 
    correctly. However, it is possible that the smoothing can avoid a local 
    maxima, and allow the second MaxEnt run to converge to the global maxima.'''

    omega,A,alpha = ClassicalMaxEnt(T,G,covar,beta,omega,guess,**keys)

    from numpy import exp,trapz,array
    smoothed = [trapz(A*exp(-smoothing_factor*(omega[i] - omega)**2),omega) for i in range(len(omega))]
    smoothed = array(smoothed)
    smoothed /= trapz(smoothed,omega)

    # Also use the final alpha from before here
    omega,A,alpha = ClassicalMaxEnt(T,G,covar,beta,omega,smoothed,alpha,**keys)

    return omega,A,alpha

def ClassicalMaxEnt(T,G,covar,beta,omega,guess=None,start_alpha=100.,**keys):
    r'''This is the main function. It performs the search for the alpha that 
    maximises the probability function :math:`\mathrm{Pr}[A|\bar{G}]`.
    
    The search combines both a bisection method and an iterative search based on the equation for a maximum of the probability function (see :func:`FitAlpha`). The bisection method takes priority in the search, but new values of alpha are chosen from a substitution into the equation of the maximum where possible.

    **Returns:** ``omega,A,alpha``

        ``omega`` (real vector)
            The omega spacing of the spectra.
        ``A`` (vector)
            The spectra.
        ``alpha`` (scalar)
            The value of alpha that was reached for convergence.

    **Notes:**
     - It is assumed that the ``alpha_diff`` results from :func:`FitAlpha` are 
       monotonically increasing.  Each call to :func:`FitAlpha` is assumed to 
       have converged to the maxima for that value of alpha.
    
     - This function allows for any optional keywords that can be passed onto 
       the :func:`FitAlpha` function.
    
     - One can uncomment a portion of this function that allows for gradual 
       increase of accuracy in the calls to :func:`FitAlpha`.
       
     - This function takes the covar matrix instead of the inverse.'''
    
    from numpy.linalg import inv
    from numpy import diag
    alpha = start_alpha
    A = guess
    covar_inv = inv(covar)
    #covar_inv = diag(1./diag(covar))

    # FIXME: Disabled increasing iters
    increased_iter = True
    alpha_bound_max = 1e10
    alpha_bound_min = 0.
    for iter in range(25):
        # Gradually get more accurate with the A fitting.
        # Do this in cycles.
        #if iter%30 < 10:
        #    max_iters = 40
        #elif iter%30 < 20:
        #    max_iters = 100
        #else:
        #    max_iters = 1000
        #max_iters = 10000

        omega,A,new_alpha,res,converged,alpha_diff = FitAlpha(T,G,covar_inv,beta,omega,A,alpha,**keys)

        #import pylab
        #pylab.plot(omega,A,label=str(iter))

        if alpha > 9e2 and new_alpha > 9e2:
            logger.info("Stopping because alpha is too big")
            break
        elif alpha < 5e-3:
            logger.info("Stopping because alpha is too small")
            break

        # This makes a (possibly) bad assumption about the monoticity of the 
        # alpha max condition function.
        if alpha_diff < 0 and alpha < alpha_bound_max:
            alpha_bound_max = alpha
        elif alpha_diff > 0 and alpha > alpha_bound_min:
            alpha_bound_min = alpha

        if iter > 4:
            # Reset the min max bounds.
            alpha_bound_max = 1e10
            alpha_bound_min = 0.

        # Take over choosing a new alpha if we are going out of bounds
        # NOTE: If this continually happens, then perhaps the convergence will 
        # not be to the global minima. One should always check this thoroughly.
        if new_alpha >= alpha_bound_max or new_alpha <= alpha_bound_min:
            logger.info("Rechoosing alpha between {0} and {1}, because new_alpha is {2}".format(alpha_bound_min,alpha_bound_max,new_alpha))
            new_alpha = (alpha_bound_max + alpha_bound_min)/2.

        diff = new_alpha - alpha
        alpha = new_alpha
        logger.info("Alpha is now {0}".format(alpha))
        if abs(diff)/new_alpha < 1e-2:
            if converged:
                break
            # Since we are within tolerance, yet not converging, turn up the 
            # max_iters for the FitAlpha func
            if increased_iter:
                break
            elif 'max_iters' in keys:
                keys['max_iters'] *= 10
                increased_iter = True
                pass
            else:
                # FIXME: This has been disabled above.
                logger.info("Increasing max iterations")
                increased_iter = True
                keys['max_iters'] = 5000
    else:
        logger.info("Warning: reached end without converging")

    return omega,A,alpha


def FitAlpha(T,G,covar_inv,beta,omega,guess,alpha,model=1,disp_info=0,prior='inverse',max_iters=500,covar_diag=False,tolerance=1e-4):
    r'''Find the spectra that maximises the probability function 
    :math:`\mathrm{Pr}[A|\bar{G},\alpha]` for one particular value of alpha.

    This is base worker function of the MaxEnt procedure. It takes as input the Green's function via ``T`` and ``G`` for a system at inverse temperature ``beta``, with the covariance information as an inverted matrix in ``covar_inv``.

    The continued spectra is found on a grid ``omega`` using the specified value of ``alpha``. The procedure maximises the probabilty function for the spectra, starting from an initial spectra of ``guess``. The model used is taken from the type specified by the argument ``model``:

        ``model = 0``
            A normalised flat model, :math:`m(\omega) = 1/(\omega_N - \omega_0)`.
        ``model = 1``
            A Gaussian model, :math:`m(\omega) = e^{-\omega^2 / \sigma^2} / (\sqrt{2\pi} \sigma)`. Currently with :math:`\sigma = 2`
        ``model = 2``
            A flat model, normalised to pi for testing purposes.

    The bias for the likelihood of alpha values is given in ``prior``. It can 
    be either "none" for no bias or "inverse" for a :math:`1/\alpha` bias. This 
    seems to be useful if the maximisation procedure keeps trying to make alpha 
    go to infinity.

    ``max_iters`` sets a limit to the maximisation iterations.

    At the end of this function, a new guess for alpha is calculated from the 
    maximisation equation :math:`-2S\alpha = \sum_n \lambda_n / (\alpha + 
    \lambda_n)` where :math:`\lambda_n` are the eigenvalues of the Hessian 
    matrix (second derivatives with additional A factors).

    **Returns:** ``omega,A,new_alpha,res,converged,alpha_diff``

        ``omega`` (real vector)
            The omega spacing of the spectra.
        ``A`` (vector)
            The spectra.
        ``new_alpha`` (scalar)
            A guess for the next alpha value to try, using the above-mentioned 
            maximisation equation.
        ``res`` (scalar)
            The exact value of :math:`\mathrm{Pr}[A|\bar{G},\alpha]` (up to a 
            proporitionality on :math:`\mathrm{Pr}[\bar{G}]`).
        ``converged`` (boolean)
            Whether the function converged to within the tolerance before reaching 
            the maximum number of iterations.
        ``alpha_diff`` (scalar)
            The difference of the LHS and RHS in the maximum equation as 
            described above. Used in :func:`ClassicalMaxEnt` to implemented the 
            bisection method.
        ``tolerance`` (scalar)
            This is the numerical tolerance for the stopping condition of the 
            maximisation loop. The summed difference of the spectra between at 
            two points, 20 iterations apart, are compared as well as the 
            relative difference of the Q value.
    
    **Notes:**
     - If ``guess`` is ``None`` then the starting guess will be a spectra very 
       close to the model.

     - The model chosen should not significantly affect the results. If it does 
       then there is a larger problem than the choice of model.
    
    .. todo::
        Create an optimised version for when the covariance information is 
        diagonal (e.g. for faked variance information on self-energies). At the 
        moment, ``covar_diag`` is allowed, but this still extracts the diagonal 
        elements from the matrix which might be a bit slow. Try extracting once 
        at the beginning only.
    
    .. todo::
        An attempt at implementing the Bryan's search method is included here, 
        however it has not been fully successful. If this function gets too slow 
        this should be looked at again.
    '''
    from numpy import linspace,r_,ones_like,dot,exp,fromfunction,sqrt,pi,diag,log,identity,zeros_like,trapz
    from numpy.linalg import norm,eigvals,eigvalsh
    
    if omega == None:
        omega = linspace(-10,10,100)
    elif type(omega) == int:
        omega = linspace(-10,10,omega)
    else:
        # Make sure it is monotonically increasing
        assert(all(omega[1:] - omega[:-1]) > 0)

    if type(model) == int:
        if model == 0:
            print("Using model 0!!!")
            m = ones_like(omega) / (omega[-1] - omega[0])
        elif model == 1:
            sigma = 2. 
            m = 1./(sqrt(2*pi)*sigma) * exp(-omega**2/(2*sigma**2)) 
        elif model == 2:
            m = ones_like(omega) / (omega[-1] - omega[0]) * pi
        elif model == 3:
            # Same as 1 but with sigma = 8.
            print("Using model 3!!!")
            sigma = 8.
            m = 1./(sqrt(2*pi)*sigma) * exp(-omega**2/(2*sigma**2)) 
        elif model == 4:
            # Same as 1 but with sigma = 3.
            #print("Using model 4!!!")
            sigma = 3.
            m = 1./(sqrt(2*pi)*sigma) * exp(-omega**2/(2*sigma**2)) 
        else:
            raise ValueError("Unknown model type")

        # Always normalise these models
        m /= trapz(m,omega)
    else:
        m = model

    if guess == None:
        guess = m * 0.9

    omega_weight = omega[1:] - omega[:-1]
    omega_weight = (omega_weight[1:] + omega_weight[:-1])/2.
    omega_weight = r_[(omega[1] - omega[0])/2., omega_weight, (omega[-1] - omega[-2])/2.]

    m = m * omega_weight
    A = guess * omega_weight

    #K = exp(-T[:,None]*omega[None,:]) / (1+exp(-beta*omega[None,:]))
    K = 1. / (exp(T[:,None]*omega[None,:]) + exp((T[:,None] - beta)*omega))
    #from numpy.linalg import svd
    #V,S,U = svd(K)
    #search_space = U[:,S > S[0]/10.].T

    normalisation_list = []

    oldA = zeros_like(A)
    oldQ = 0.
    for quad_iter in range(max_iters):
        S = sum(A - m - A*log(A/m))
        dS = -log(A/m)
        ddS = diag(-1./A)

        Gapprox = dot(K,A)
        Gdiff = G - Gapprox

        if covar_diag:
            L = 1/2. * dot(Gdiff.T, diag(covar_inv)*Gdiff)# / len(T)
            dL = dot( -K.T, diag(covar_inv)*Gdiff)# / len(T)
            ddL = dot(K.T,diag(covar_inv)[:,None]*K)# / len(T)
        else:
            L = 1/2. * dot(Gdiff.T, dot(covar_inv,Gdiff))# / len(T)
            dL = dot( -K.T, dot(covar_inv, Gdiff))# / len(T)
            ddL = dot(K.T,dot(covar_inv,K))# / len(T)

        Q = alpha*S - L
        dQ = alpha*dS - dL
        ddQ = alpha*ddS - ddL

        # An old test of using the complete space to optimise - does not 
        # complete in any reasonable time.
        if False:
            from scipy.optimize import fmin_ncg
            S = lambda A: sum(A - m - A*log(A/m))
            L = lambda A: 1/2. * dot( (G - dot(K,A)).T, dot(covar_inv, (G - dot(K,A))))
            Q = lambda A: alpha*S(A) - L(A)
            dS = lambda A: -log(A/m)
            dL = lambda A: dot( -K.T, dot(covar_inv, (G - dot(K,A))))
            dQ = lambda A: alpha*dS(A) - dL(A)
            ddS = lambda A: diag(-1./A)
            ddL = lambda A: dot( K.T, dot(covar_inv, K))
            ddQ = lambda A: alpha*ddS(A) - ddL(A)

            #return A,L,dL,ddL,S,dS,ddS

            
            def callback(A):
                import pylab
                pylab.figure()
                pylab.plot(A)
                pylab.show()
                waiting = raw_input('Waiting for keystroke')

            out = fmin_ncg(Q,A,dQ,fhess=ddQ,full_output=True,callback=callback)
            newA = out[0]

        if True:
            num_vectors = 3
            e = [None]*num_vectors
            e[0] = A * dS
            e[1] = A * dL
            dL_norm = dL / norm(dL)
            if norm(dS) != 0.:
                dS_norm = dS / norm(dS)
            else:
                dS_norm = dS*0
            #e[2] = A * dot(ddL,A*dS_norm - A*dL_norm)
            #e[2] = A * dot(ddL,alpha*A*dS/norm(dS) - A*dL/norm(dL))
            #e[2] = A * dot(ddL,alpha*dS/norm(dS) - dL/norm(dL))
            ddL_pow = identity(len(ddL))
            temp_vec = alpha*dS - dL
            temp_vec /= norm(temp_vec)
            for i in range(2,num_vectors):
                ddL_pow = dot(ddL,ddL_pow)
                ddL_pow /= norm(ddL_pow)
                #e[i] = A*dot(ddL_pow,alpha*A*dS_norm - A*dL_norm)
                #e[i] = A*dot(ddL_pow,alpha*dS_norm - dL_norm)
                e[i] = A*dot(ddL_pow,temp_vec)
            #e = [e1/norm(e1),e2/norm(e2),e3/norm(e3)]
            e = [z/norm(z) for z in e]

            coeff0 = Q
            coeffs1 = array( [dot(ei,dQ) for ei in e] )
            coeffs2 = array( [[1/2.*dot(ei.T, dot(ddQ,ej)) for ej in e] for ei in e] )
            try:
                movement = MaxQuad(coeff0,coeffs1,coeffs2)
            except SingularMatrixError:
                # At this point we cannot improve the solution. This is bad, 
                # but we can ignore this and hope that it improves in the next 
                # loop given by controlling function
                converged = True
                break

            if any(movement != movement):
                print(ddL_pow)
                print(norm(ddL_pow))
                print(alpha*dS - dL)
                raise Exception("Got some NaNs in FitAlpha!",e)
            # The 0.9 is here to help with oscillating convergence
            A = A + sum(movement[i] * 0.9 * e[i] for i in range(num_vectors))

            # Make sure this is well behaved - any negatives appearing in newA 
            # will cause problems with NaNs. Additionally, any very small 
            # values will also cause problems, so avoid them as well.
            #A[A < 0] = 1e-8
            A[A < 1e-8] = 1e-8

            # Compare to exit the loops only once in a while (also good for 
            # error comparison)
            if quad_iter % 20 == 0:
                diff = sum(abs(oldA - A))
                if diff < tolerance and (Q - oldQ)/(Q + oldQ) < tolerance:
                    converged = True
                    break
                oldA = A
                oldQ = Q

                if disp_info >= 2:
                    print("Q:",Q)
                    print("diff:",diff)
                    print("New normalisation:",sum(A))

            # Normalisation does not seem to help very much and can in fact 
            # hinder the procedure.
            #A /= sum(A)
            A = A.real
        if False:
            e = [A * dS]
            for col in search_space:
                e += [col]

            coeff0 = Q
            coeffs1 = array( [dot(ei,dQ) for ei in e] )
            coeffs2 = array( [[1/2.*dot(ei.T, dot(ddQ,ej)) for ej in e] for ei in e] )
            movement = MaxQuad(coeff0,coeffs1,coeffs2)

            A = A + sum(movement[i] * 1e-1 * e[i] for i in range(3))

            # Make sure this is well behaved - any negatives appearing in newA 
            # will cause problems with NaNs
            A[A < 0] = 1e-8

            # Compare to exit the loops only once in a while (also good for 
            # error comparison)
            if quad_iter % 20 == 0:
                diff = sum(abs(oldA - A))
                if diff < 1e-6 and (Q - oldQ)/(Q + oldQ) < 1e-6:
                    converged = True
                    break
                oldA = A
                oldQ = Q

                if disp_info >= 2:
                    print("Q:",Q)
                    print("diff:",diff)
                    print("New normalisation:",sum(A))

            # Renormalisation does not seem to help very much and can in fact 
            # hinder the procedure.
            #A /= sum(A)
            A = A.real

    else:
        #logger.info("Warning: reached end without converging. Q and oldQ are {0} and {1}. A_diff is {2}".format(Q,oldQ,diff))
        converged = False


    if disp_info >= 1:
        print("At end:")
        print("Normalisation:",sum(A))
        print("Q:",Q)
        print("S:",S)
        print("L:",L)


    # Attempt to guess a better alpha.
    Lambda = sqrt(A)[:,None] * ddL * sqrt(A)[None,:]
    evals = eigvalsh(Lambda)

    if prior == 'none':
        new_alpha = ( (sum(evals/(alpha + evals))) / (-2*S)).real
        alpha_diff = sum(evals)/(alpha + evals) + 2*S*alpha
    elif prior == 'inverse':
        new_alpha = ( (sum(evals/(alpha + evals)) + 2*-1) / (-2*S)).real
        alpha_diff = sum(evals/(alpha + evals)) + 2*-1 + 2*S*alpha
    #if abs(new_alpha - alpha) > 10:
    #    from numpy.random import rand
    #    new_alpha = alpha + (new_alpha - alpha)/abs(new_alpha - alpha) * 10 + rand(1)
    if new_alpha < alpha and new_alpha / alpha < 0.1:
        new_alpha = alpha * 0.1
    elif new_alpha > alpha and new_alpha / alpha > 10:
        new_alpha = alpha * 10


    # Work out the alpha probability exactly
    if prior == 'none':
        Pr_alpha = 1.
    elif prior == 'inverse':
        Pr_alpha = 1./alpha

    res = log(Pr_alpha) + Q - 0.5*sum(log(1 + evals/alpha)) 

    if disp_info >= 1:
        print("New alpha guess at:",new_alpha.real)
        good_data = sum(evals/(alpha + evals))
        print("Good data guess at:",good_data)
        print("Alpha probability is:",res)

    return omega,A/omega_weight,new_alpha,res,converged,alpha_diff


class SingularMatrixError(Exception):
    pass

def MaxQuad(coeff0,coeffs1,coeffs2):
    '''Simple function to solve the basic gradient descent equations. Also 
    handles singular matrix exceptions.'''
    from numpy import dot,zeros_like
    from numpy.linalg import inv,LinAlgError

    try:
        solution = dot(inv(coeffs2), -coeffs1)
    except LinAlgError as exc:
        if 'Singular matrix' in str(exc):
            logger.warning("Got a singular matrix! Bailing")
            #print(coeffs1,coeffs2)
            #raise SingularMatrixError
            return zeros_like(coeffs1)
        else:
            print("LinAlgError has the str",str(LinAlgError))
            raise


    #def func(x):
    #    return coeff0 + dot(coeffs1,x) + dot(x.T,dot(coeffs2,x))

    #print(func(solution), func(solution + [0,0.01,0.01]), func(solution + [0.01]))

    return solution





