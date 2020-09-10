r'''
This module implements real-space DMFT code for the spinful Hofstadter model, 
as proposed in Goldman et al, PRL 105, 255302 (2010). The non-interacting Hamiltonian is:

.. math::

    H = & -\sum_{j}  \Bigl\{ t_x c^\dag_{j+\hat{\bf x}} \,e^{- i 2 \pi \gamma \sigma^x} c_{j} + t_y c^\dag_{j + \hat{\bf y}} \,e^{i 2 \pi \alpha x \sigma^z} c_j + \text{h.c.} \Bigr\} + \lambda_x \sum_{j} (-1)^x c^\dag_j c_j

where the creation/annihiliation operators are :math:`c_j = (c_{j,\uparrow}, c_{j,\downarrow})^T` and the
Peierl's phases result from the Landau gauge :math:`\boldsymbol{A} = (0,Bx,0)`.

The main calculation is controlled by :func:`TI_CTAUX` and this should be the 
entry point for any code. The parameters for the self-consistency will be set up 
here, and the appropriate symmetries and potential will be created by 
the :func:`TI_Symmetry` function. :func:`TI_CTAUX` then calls :func:`ConvergenceSpin` 
which glues the individual self-consistency loops together. It manages the 
dampening and convergence to a target filling. The individual self-consistency 
loops themselves are implemented by :func:`SelfConsistencySpin`, which generates 
the Green's functions and calls the impurity solver. Further detail about the 
process is given below.

A lot of the information that is output from the solver is saved into files with 
three extension types:

    * ``*.dat``: These files are text files, and are meant for human 
      readability.
    * ``*.npy``: These files are numpy saved files which contain one array.
    * ``*.npz``: These files are numpy groups which contain many arrays.

In order to look at the data in the ``*.np?`` files, please see the section :ref:`reading-formats`.

There are many analysis functions, however an important convenience function 
exists to access general information about an individual run: 
:func:`ReadInfo`. It will parse the ``info.dat`` file, which summarizes a run 
and return a dictionary of useful information.

This module depends on :mod:`ImagConv`, :mod:`CT_AUX_SM`, :mod:`ED_new`, :mod:`BlockMatrixInv` and :mod:`Continuation`. 

Summary of functions:

    **RDMFT Calculations:** (with more high-level functions first)

        :func:`TI_CTAUX`
            Sets up parameters for the calculation.
        :func:`ConvergenceSpin`
            Manages the self-consistency loops.
        :func:`SelfConsistencySpin`
            Performs one single self-consistency loop with parallelization.
        :func:`GreensMatSinglePython`
            Calculates the Green's matrix for a single omega.
        :func:`TI_Symmetry`
            Generates appropriate symmetry and potential arrays.

    **Continuation and Fourier transforming:**
        :func:`TI_FourierCTAUX`
            Interface for :func:`FourierTransformSpin` for Matsubara frequencies.
        :func:`TI_FourierCTAUXRealFreq`
            Interface for :func:`FourierTransformSpin` for real frequencies.
        :func:`FourierTransformSpin`
            Calculates the various types of Fourier transformed Green's matrices.
        :func:`FourierTransformSpinSingle`
            Calculates the Fourier transformed Green's matrix for a single omega.
        :func:`TI_ContinueCTAUX_SE`
            Continues the Matsubara self-energies to real-space.
        :func:`TI_ContinueCTAUX_Gk`
            Continues the Fourier transformed Green's matrix to real-space.
        :func:`ContinueSelfEnergy`
            Handles the continuation of a single function to real-space.

    **Analysis:**

        :func:`ReadInfo`
            Parse the ``info.dat`` file into a convenient dictionary.
        :func:`ReadGkData`
            A convenience function to read the data output from :func:`TI_FourierCTAUX` and :func:`TI_FourierCTAUXRealFreq`.
        :func:`AverageData`
            A convenience function to average the last few self-consistency iterations.
        :func:`PlotGkData`
            Create some basic plots from the G(x,y) Green's function data.
        :func:`PlotPeaksMap`
            Create some basic plots from the G(x,k_y) Green's function data.
        :func:`PlotOmega0Data`
            Show detailed plots of the spectra about omega=0.
        :func:`SavePeaksTI`
            Collects all the peaks from each spectra.
        :func:`FindPeaksWithWeights`
            Identifies peaks within one spectra.
        :func:`PlotPeaksWithWeights`
            Plots the peaks calculated with :func:`SavePeaksTI`
        :func:`FitMagExp`
            Extrapolates the magnetisation to infinite self-consistency loops.
        :func:`CalcNZInvariant`
            Calculate the integer QH invariant for S_z conserving systems.
        :func:`GenerateAllDataCTAUX`
            Run various other functions to generate the common data from each run.
        :func:`GenerateAllDataForU0`
            Setup a fake U=0 run and generate data like :func:`GenerateAllDataCTAUX`.
        :func:`DisplayFillingFromNup`
            Display the filling and spin dependence over the grid.
        :func:`JudgeValidity`
            Give a rough estimate of whether the run has converged and if it has encountered any issues.

    **Additional:**

        These functions are used only for specific purposes and have not been documented.

        :func:`TI_AnotherIter`
            Performs an additional self-consistency loop at larger CT sweeps.

        * :func:`FitMagExp`
        * :func:`LookForGaps`
        * :func:`PlotLastMags`
        * :func:`PlotGkLargestAtZero`
        * :func:`DoNiceMagPlot`



Typical Run of the Program
==========================

A typical calculation will perform something similar to the code beneath. The steps 
are basically

    #. Setup the parameters.
    #. Determine if a previous run has occurred, and if so from which iteration to resume.
    #. Run the calculation with :func:`TI_CTAUX`.
    #. Run the continuation with :func:`TI_ContinueCTAUX_SE`.
    #. Create all data with :func:`GenerateAllDataCTAUX`.
    #. Try to catch obvious errors with :func:`JudgeValidity`.

This code currently is designed to run on the Loewe cluster which has nodes with
24 cores. If you would like to run this on a local machine, then one should indicate
how many cores are available with ``export NSLOTS=x`` in a shell prompt, followed by
``bash script.sh``. The code is:

.. literalinclude:: example_figures/RDMFT_TISM/cluster_template.sh
    :language: python

.. _reading-formats:

Reading .npy and .npz files
===========================

To open ``.npy`` and ``.npz`` files, one must use the python function 
``load`` from the module ``numpy``. ``.npy`` files contain a single array 
of data, whereas ``.npz`` files contain many sets of data. The following 
code snippet illustrates how to load these::

    from numpy import load
    SE = load('SE_iter20.npy')
    SE[0,:,0] # <-- This is \Sigma_{i=0,upup}(omega)

    data = load('everything_iter20.npz')
    data['SE'][0,:,0] # As above
    data['N_dbl'][0] # Double occupancy at site i=0

With ``.npz`` files one can view all of the data contained within, with 
the following method::

    data = load('everything_iter20.npz')
    print(data.files)
    ['GT', 'old_SE', 'G', 'weiss', 'WT', 'SE', 'test_mat']

It is also possible to convert these to Matlab files using the following type of code::

    from numpy import load
    from scipy.io import savemat
    data = load('file.npz')
    savemat('newfile.mat',data)

Be aware that the ``None`` type cannot be interpreted into Matlab, so one may have to 
manually delete these before saving.

Member functions
================

'''

from __future__ import print_function
from std_imports import *
import matplotlib
#matplotlib.use('Agg')
#import pylab
#pylab.rcParams['text.usetex'] = True

__all__ = ['TI_CTAUX',
           'ConvergenceSpin',
           'SelfConsistencySpin',
           'GreensMatSinglePython',
           'TI_Symmetry',
           'TI_FourierCTAUX',
           'TI_FourierCTAUXRealFreq',
           'FourierTransformSpin',
           'FourierTransformSpinSingle',
           'TI_ContinueCTAUX_SE',
           'TI_ContinueCTAUX_Gk',
           'ContinueSelfEnergy',
           'ReadInfo',
           'ReadGkData',
           'AverageData',
           'PlotGkData',
           'PlotPeaksMap',
           'PlotOmega0Data',
           'SavePeaksTI',
           'FindPeaksWithWeights',
           'PlotPeaksWithWeights',
           'FitMagExp',
           'CalcNZInvariant',
           'GenerateAllDataCTAUX',
           'GenerateAllDataForU0',
           'DisplayFillingFromNup',
           'JudgeValidity',
           'TI_AnotherIter',
           'FitMagExp',
           'LookForGaps',
           'PlotLastMags',
           'PlotGkLargestAtZero',
           'DoNiceMagPlot']
file_version = 10

# This is the path to the C-code for the RDMFT inversion.
# It is no longer used by the program.
import os
Ccode_location = os.path.expanduser('~/work3/ccode/RDMFT/')

def FindPeaksWithWeights(omega,f):
    '''This function takes a spectra ``f`` with frequencies ``omega`` and looks 
    for peaks. Peaks are defined to be at turning points in the spectra, 
    calcualted from the sign change of the first derivative.

    Also returned is a value indicating the 'strength' of the peak. This weight is larger for 
    taller and sharper peaks. The function also returns the locations of minima in the spectra 
    for which the weight is negative.

    **Returns:** ``points,weights``

        ``points`` (double array)
            Frequencies of peaks.
        ``weights`` (double array)
            Weights of peaks.
    '''

    from numpy import where
    diff = (f[1:] - f[:-1])
    #ind = where(((diff[1:]*diff[:-1]) <= 0))[0] * (diff[:-1]>=0) * (f[1:-1]>=cutoff) * (diff[:-1] - diff[1:]>cutoff_diff))[0]
    ind = where(((diff[1:]*diff[:-1]) <= 0))[0]

    # leave off indices right near the edge (because of weight_deriv)
    while len(ind)>0 and ind[0] < 3:
        ind = ind[1:]
    while len(ind)>0 and ind[-1] > len(diff)-4:
        ind = ind[:-1]

    if len(ind) == 0:
        return [],[]

    # Define a weight based on the height of the peak, and the steepness of the 
    # peak. Currently using 3 points either side to generate a more accurate 
    # derivative
    omega_mid = (omega[1:] + omega[:-1])/2.
    f_mid = (f[1:] + f[:-1])/2.

    weight_deriv = ( (f_mid[ind] - f_mid[ind-3])/(omega_mid[ind] - omega_mid[ind-3]) \
            - (f_mid[ind+3] - f_mid[ind])/(omega_mid[ind+3] - omega_mid[ind]) ) 
            #/ f_mid[ind]
    weight_value = f_mid[ind]
    # Make any minima show up as negatives (the deriv is already negative)
    weight_value[diff[ind] < 0] *= -1

    weight = weight_value + weight_deriv
    
    return omega_mid[ind],weight

def GetkValues(N,boundary):
    from numpy import pi,arange
    if isiterable(N):
        assert isiterable(boundary)
        assert len(N) == 2
        kx = GetkValues(N[0],boundary[0])
        ky = GetkValues(N[1],boundary[1])
        return kx,ky

    if boundary == 'P':
        k_list = 2*pi/N * arange(N)
    elif boundary == 'O':
        k_list = 2*pi/(N+1) * arange(1,N+1)

    return k_list

def SavePeaksTI(spin):
    '''This function iterates through all of the spectra in x,ky-space and 
    saves the peaks and weights found to the file 'peaks<spin>.pickle.gz'
    where <spin> is the given ``spin``.

    These files are saved in pickle format, with the entries:
        ``file_version``:
        ``x,ky,peaks`` with

            ``x``: (double array) list of x points.
            ``ky``: (double array) list of ky points.
            ``peaks``: (list of lists of [double array,double array]) ``peaks[x][ky][0][peak_i]`` is the peak with weight ``peaks[x][ky][1][peak_i]``

    **Returns:**
        ``x,ky,peaks``: These values are as is saved above.
    '''
    assert spin in ['up','down']

    info = ReadInfo()
    N = info['N']; M = info['M']; boundaries = info['boundaries']

    kx,ky = GetkValues((N,M),boundaries)

    peaks = []

    omega,Gk = ReadGkData(spin,real_freq=True)
    for kx_i in range(N):
        peaks.append([])
        for ky_i in range(M):
            peaks[kx_i] += [FindPeaksWithWeights(omega,abs(Gk[kx_i,ky_i].imag))]

    import gzip,pickle
    file = gzip.open('peaks'+spin+'.pickle.gz','wb')
    global file_version
    pickle.dump(file_version,file)
    pickle.dump((kx,ky,peaks),file)
    file.close()

    return kx,ky,peaks

def PlotPeaksWithWeights(input=None,color=None,min_weight=None,max_weight=None,spin='up',x=None):
    '''This function takes the peaks calculated by function 
    :func:`FindPeaksWithWeights` and saved by :func:`SavePeaksTI` and display them as a 
    scatter plot using colours to indicate the weights.
    
    One can safely call this function with no arguments, if :func:`SavePeaksTI` has already 
    been called.

    For additional configuration, one can pass a set of ``kx,ky,peaks`` calculated from 
    :func:`FindPeaksWithWeights` as ``input``. The colour of each peak can be set 
    via an array ``color``. Peaks can be omitted based on ``min_weight`` and ``max_weight``. 
    And the peaks corresponding to only one value of ``x`` can also be specified.

    An example of a plot from this function is below:

    .. image:: example_figures/RDMFT_TISM/peaks_up.png

    '''

    if input == None:
        import pickle,gzip
        file = gzip.open('peaks'+spin+'.pickle.gz')
        file_version = pickle.load(file)
        kx,ky,peaks = pickle.load(file)
        file.close()
    else:
        kx,ky,peaks = input

    #spoints = array([ (ky[j], p, kx[i]) for i in range(len(kx)) for j in range(len(ky)) for p in peaks[i][j][0]])
    if x == None:
        spoints = array([ (ky[j], peaks[i][j][0][p_i], peaks[i][j][1][p_i]) for i in range(len(kx)) for j in range(len(ky)) for p_i in range(len(peaks[i][j][0]))])
    else:
        spoints = array([ (ky[j], peaks[x][j][0][p_i], peaks[x][j][1][p_i]) for j in range(len(ky)) for p_i in range(len(peaks[x][j][0]))])
    spoints = spoints.real

    if len(spoints) == 0:
        return
    # TODO: temp removal of negative weights
    spoints = spoints[spoints[:,2] > 0]
    
    # TODO: trying different weights
    spoints[:,2] = log(spoints[:,2])

    if min_weight:
        spoints = spoints[spoints[:,2] > min_weight]
    if max_weight:
        spoints = spoints[spoints[:,2] < max_weight]

    # Sort by the weight to make sure the high weights get plotted on top
    spoints = spoints[spoints[:,2].argsort()]

    from pylab import scatter,colorbar
    if color == None:
        scatter(spoints[:,0],spoints[:,1],c=spoints[:,2])
        colorbar()
    else:
        scatter(spoints[:,0],spoints[:,1],c=color)


global_print_output = False
def GreensMatSingleOLD(SE,potential,(N,M),boundaries,omega,(p,q),mu,gamma,no_invert,complete_mat,t=1.):
    '''Note: this function is deprecated! Please use 
    :func:`GreensMatSinglePython` instead.
    
    This function interfaces with the C code to generate a single matrix of the 
    single-particle Green's function including self-energy.

    Normally, the matrix returned is the matrix of Green's functions, but when 
    ``no_invert = True`` then it is the matrix of the single-particle 
    Hamiltonian with omega subtracted.

    As well, the function normally returns the diagonal G_rr' \delta{r,r'} 
    terms, however one can return the complete matrix by setting ``complete_mat 
    = True``.

    Note that the self-energy must be provided in the variable ``SE`` as a 
    tuple of three vectors, i.e. ``SE = (SE_up,SE_down,SE_cross)``.
    '''
    assert no_invert in [False,True]
    assert complete_mat in [False,True]
    assert len(SE) == 3
    for SE_i in SE:
        assert len(SE_i) == N*M
        assert SE_i.dtype == 'complex128'
    assert len(potential) == N*M
    assert potential.dtype == 'float64'

    no_invert_param = 'Y' if no_invert else 'N'
    return_diag_param = 'N' if complete_mat else 'Y'

    import subprocess
    cmdline = [Ccode_location + 'RDMFT_TISM_greensinv','-',boundaries,str(N),str(M),repr(omega.real),repr(omega.imag),repr(t),str(p),str(q),repr(mu),repr(gamma),'-','-',no_invert_param,return_diag_param]
    proc = subprocess.Popen(cmdline,stdout=subprocess.PIPE,stdin=subprocess.PIPE)

    SE[0].tofile(proc.stdin)
    SE[1].tofile(proc.stdin)
    SE[2].tofile(proc.stdin)
    potential.real.tofile(proc.stdin)
    proc.stdin.close()

    # Wait for the signal text
    while True:
        line = proc.stdout.readline()
        if line == '':
            if proc.poll() != None:
                raise Exception("Process finished early")
            import time
            time.sleep(1)
        if global_print_output:
            logger.info(line.strip())
        if line == 'Writing data\n':
            break
    else:
        raise Exception("Did not find sentinel string")

    from numpy import fromfile
    if complete_mat:
        mat = fromfile(proc.stdout,'complex128',(N*M*2)**2)
        #mat = mat.reshape(N,M,2,N,M,2)
        mat = mat.reshape(M,N,2,M,N,2)
        mat = mat.transpose(1,0,2,4,3,5)
    else:
        mat = fromfile(proc.stdout,'complex128',N*M*3)
        #mat = mat.reshape(N,M,3)
        mat = mat.reshape(M,N,3)
        mat = mat.transpose(1,0,2)

    return mat

def GreensMatSinglePython(SE,potential,(N,M),boundaries,omega,(p,q),mu,gamma,no_invert,complete_mat,t=1.):
    r'''This function calculates the inverted Green's matrix at frequency ``omega`` (real or Matsubara) 
    from a given self-energy ``SE`` and system parameters:

        * Flux \alpha = ``p/q`` ``gamma``
        * Global chemical potential ``mu``
        * Spin-mixing ``gamma``
        * Lattice dimensions ``N``x``M``
        * Hopping ``t``.

    A site-dependent potential can be specified as an array in ``potential`` and the 
    boundary conditions are specified as a string of two characters ('P' or 'O' for periodic 
    and open respectively) in ``boundaries``. (E.g. ``boundaries='OP'`` for open boundary conditions in x and periodic in y)

    The matrix is defined from its inverse as:

    .. math::

        (G^{-1})_{ij} &= t \exp(\pm 2\pi i x p/q) \delta_{i_x,i_x\pm1} \\
                &{}+ t \exp(\mp 2\pi i \gamma) \delta_{i_y,i_y\pm1} \\
                &{} + (\omega + \mu - V_i) \delta_{ij} - \Sigma_{i,j_\sigma} \delta_{i_x,j_x}\delta_{i_y,j_y}

    where :math:`i\equiv i_x,i_j,i_\sigma`. Note the reversal of sign in the \gamma term.
    
    Normally, the matrix returned is the matrix of Green's functions, but when 
    ``no_invert == True`` then the uninverted matrix is returned (i.e. the single-particle 
    Hamiltonian with omega subtracted).

    As well, the function normally returns the diagonal :math:`G_{ii} \delta_{i,j}` 
    terms, however one can return the complete matrix by setting
    ``complete_mat == True``.

    Note that the self-energy must be provided in the variable ``SE`` as a 
    tuple of four vectors, i.e. ``SE = (SE_up,SE_down,SE_updown,SE_downup)``. This is necessary 
    for the Matsubara frequency case, where this function cannot calculate 
    ``SE_downup(i omega) = SE_updown(-i omega)``. Each vector must contain ``NxM`` elements which 
    are stored in row-major ordering (i.e. the first ``N`` elements are one row of the system).
    
    This function uses the :mod:`BlockMatrixInv` module for quick inversion.

    **Returns:**
        ``mat`` (complex double array)
            * With ``complete_mat==False``, this is only the diagonal. 
              The array dimensions are: x-coord,y-coord,func, where func is 0,1,2,3 for 
              G_upup, G_downdown, G_updown, G_downup. I.e. mat[1,2,0] is 
              G_upup(x=1,y=2)

            * With ``complete_mat==True``, the array dimensions are: 
              i_x,i_y,i_spin,j_x,j_y,j_spin such that mat[1,2,0,5,6,1] = G_updown(x=1,x'=5,y=2,y'=6)
    '''
    from numpy import zeros,sin,cos,pi,conj,exp,exp,diag
    assert len(SE) == 4
    for SE_i in SE:
        assert len(SE_i) == N*M
    assert len(potential) == N*M

    assert boundaries[0] in 'OP'
    assert boundaries[1] in 'OP'
    xperiodic = True if boundaries[0] == 'P' else False
    yperiodic = True if boundaries[1] == 'P' else False

    if (yperiodic and M % q != 0) or \
        (xperiodic and N % q != 0):
        raise ValueError("Bad N,M for p,q")

    alpha = p/float(q)

    B = zeros((M,2,M,2),complex)
    x_sin_precalc = sin(2*pi*gamma)
    x_cos_precalc = cos(2*pi*gamma)
    for i in xrange(M):
        B[i,0,i,0] = -t * x_cos_precalc;
        B[i,1,i,1] = -t * x_cos_precalc;
        B[i,0,i,1] = -t * 1j*x_sin_precalc;
        B[i,1,i,0] = -t * 1j*x_sin_precalc;

    B = B.reshape(M*2,M*2)

    A = [None] * N
    for n in xrange(N):
        A[n] = zeros((M,2,M,2),complex)
        exp_precalc = conj(exp(2j*pi * (n+1) * alpha))
        exp_precalc_conj = conj(exp_precalc)

        for i in xrange(M):
            A[n][i,0,i,0] = omega + mu - SE[0][i*N + n] - potential[i*N + n]
            A[n][i,1,i,1] = omega + mu - SE[1][i*N + n] - potential[i*N + n]
            A[n][i,0,i,1] = -SE[2][i*N + n]
            A[n][i,1,i,0] = -SE[3][i*N + n]

            if i < M-1:
                A[n][i,0,i+1,0] = -t * exp_precalc
                A[n][i,1,i+1,1] = -t * exp_precalc_conj
            elif yperiodic:
                A[n][-1,0,0,0] = -t * exp_precalc
                A[n][-1,1,0,1] = -t * exp_precalc_conj
            if i > 0:
                A[n][i,0,i-1,0] = -t * exp_precalc_conj
                A[n][i,1,i-1,1] = -t * exp_precalc
            elif yperiodic:
                A[n][0,0,-1,0] = -t * exp_precalc_conj
                A[n][0,1,-1,1] = -t * exp_precalc

        A[n] = A[n].reshape(M*2,M*2)


    if no_invert:
        assert complete_mat
        mat = zeros((N,M*2,N,M*2),complex)
        for n in xrange(N):
            mat[n,:,n,:] = A[n]
            if n < N-1:
                mat[n,:,n+1,:] = B
            elif xperiodic:
                mat[n,:,0,:] = B
            if n > 0:
                mat[n,:,n-1,:] = conj(B)
            elif xperiodic:
                mat[n,:,-1,:] = conj(B)
        mat = mat.reshape(N,M,2,N,M,2)
        return mat
    else:
        import BlockMatrixInv
        if xperiodic:
            G = BlockMatrixInv.BlockedInversePeriodic(A,B,complete_mat)
        else:
            G = BlockMatrixInv.BlockedInverseOpen(A,B,complete_mat)

        from numpy import fromfile
        if complete_mat:
            mat = G.reshape(N,M,2,N,M,2)
            #mat = mat.reshape(M,N,2,M,N,2)
            #mat = mat.transpose(1,0,2,4,3,5)
        else:
            mat = zeros((N,M,4),complex)
            for n in xrange(N):
                G[n] = G[n].reshape(M,2,M,2)
                mat[n,:,0] = diag(G[n][:,0,:,0])
                mat[n,:,1] = diag(G[n][:,1,:,1])
                mat[n,:,2] = diag(G[n][:,0,:,1])
                mat[n,:,3] = diag(G[n][:,1,:,0])

            #mat = mat.reshape(N,M,3)
            #mat = mat.reshape(M,N,3)
            #mat = mat.transpose(1,0,2)

        return mat


def FourierTransformSpinSingle(SE,potential,(N,M),boundaries,omega,(p,q),mu,gamma,no_invert=False,y_only=True,complete_mat=False,t=1.):
    '''Calculate the Green's function in real-space and then Fourier transform 
    it to k-space.

    See :func:`GreensMatSinglePython` for details about the arguments ``SE,potential,(N,M),boundaries,omega,(p,q),mu,gamma,no_invert``.

    With ``y_only = True`` (the default) this will only transform y->ky and leave x untouched.

    The Fourier transform is :math:`\int dxdx' G_{x,x',...} exp^{-ikx + ik'x'}` 
    of which only the diagonal part, ``k=k'`` is returned.
    
    **Returns:**
    
        ``mat`` (complex double array)
            * With ``complete_mat==False``, this is only the diagonal elements
    '''

    mat = GreensMatSinglePython(SE,potential,(N,M),boundaries,omega,(p,q),mu,gamma,no_invert,complete_mat=True,t=t)

    from numpy.fft import fft,ifft
    from numpy import sqrt
    # Do the y transform
    mat = fft(mat,axis=1)
    # The conjugations here are to effect the opposite transform.
    mat = mat.conj()
    mat = fft(mat,axis=4)
    mat = mat.conj()
    mat /= M

    if not y_only:
        mat = fft(mat,axis=0)
        mat = mat.conj()
        mat = fft(mat,axis=3)
        mat = mat.conj()
        mat /= N

    if not complete_mat:
        from numpy import diagonal
        mat = diagonal(mat,axis1=0,axis2=3)
        mat = diagonal(mat,axis1=0,axis2=2)
        mat = mat.transpose(2,3,0,1)
        # Select just the three parts only, in the order up,down,cross
        # NO! Select all
        mat = mat.reshape(N,M,4)[:,:,[0,3,1,2]]

    return mat


def SelfConsistencySpin(U,mu,beta,gamma,SE,potential,omega,(N,M),boundaries,t,(p,q),symmetry=None,CT_sweeps=None,parallelize_CT=True,fit_large_SE=False,solver='CT-AUX',ED_params=None):
    r'''Perform one self-consistency loop of the RDMFT.

    This function takes a set of self-energies ``SE`` and performs the self-consistency steps to generate 
    a new set of self-energies. These steps are:

        #. Determine which sites need to be calculated in the impurity solver.
        #. Calculate Green's functions for the old self-energy (using :func:`GreensMatSinglePython`).
        #. Determine the local Weiss Green's function to give to the impurity solver.
        #. Run the impurity solver.
        #. Take the resultant Green's functions and calculate a new self-energy.
        #. Return this new self-energy, along with other various local quantities.

    The parameters for the system are given with ``U,mu,beta,gamma,N,M,boundaries,t,p,q``. 
    See :func:`GreensMatSinglePython` for more details about these.

    The array ``omega`` specifies the Matsubara frequencies to use in the calculations. Note that 
    this function currently assumes that this grid is mirrored about zero such that 
    ``omega[i] == -omega[len(omega)-1-i]``. ``omega`` should be purely imaginary.

    The self-energy is given as either ``SE==None``, for which the self-energy 
    is assumed to be zero, or of dimensions ``(3,num_calc_sites,len(omega))``, 
    where the first dimension corresponds to the individual self-energy funtcions, 
    :math:`\Sigma_{\uparrow\uparrow}, \Sigma_{\downarrow\downarrow},\Sigma_{\uparrow\downarrow}`, and 
    ``num_calc_sites`` is described below.

    Similarly, ``potential`` is an 1D array of size ``num_calc_sites`` that gives the local 
    potential at each site. If ``potential==None`` then it is assumed to be zero at each 
    site.

    **Symmetries:**

        If ``symmetry==None`` then all sites will be calculated in the impurity 
        solver step and the ``SE`` array will be of dimensions 
        ``(3,NxM,len(omega))``. To specify a symmetry, such that only 
        ``num_calc_sites`` need be calculated, one can pass a list via 
        ``symmetry``. This list must contain ``num_calc_sites`` ``None``'s, 
        which designate which sites must be calculated in the impurity solver. 
        Call this list ``calc_sites``. The remaining elements of the list must 
        be integers, which indicate that the site is identical to one within 
        ``calc_sites``. I.e. the integer is an index to ``calc_sites``. An 
        example better describes this. If we have::

            symmetry = [None,None,0,1,None,None,2,3]

        then only four sites of the eight will be solved in the impurity step (those with ``None``'s). 
        The Green's functions will then be copied into the other sites, such that::

            [0,1,0,1,2,3,2,3]

        indicates the symmetry, where entries with the same integer have the same Green's function.

        Note that the ``SE`` array is expanded from the input to the complete 
        grid by copying these entries as described above, and so the 
        ``symmetry`` array must be in row-major ordering as expected by 
        :func:`GreensMatSinglePython`.  The above example, for ``(N,M)=(2,2)`` 
        would then be a calculation on the left half of a grid.

        In addition, one can specify a negative integer for the ``symmetry`` elements. This then 
        indicates an anti-ferromagnetic symmetry, which is given by

        .. math ::

            G_{A,\sigma\sigma}(i\omega_n) &= G_{B,\bar\sigma\bar\sigma}(i\omega_n) \\
            G_{A,\sigma\bar\sigma}(i\omega_n) &= -G_{B,\sigma\bar\sigma}(i\omega_n) \\

        The negative integer is also an index directly to the array ``calc_list`` in python. 
        I.e. an index ``-1`` will access the last element of the calculation list and an 
        index ``-num_calc_sites`` is the same as ``0``, but with AF symmetry instead. 
        For example, one can specify::

            symmetry = [None,None,0,1,0,1,0,1,-2,-1,-2,-1,-2,-1,-2,-1]

        which indicates AF order on a ``(N,M)=(8,2)`` grid, with a two-site symmetry that is repeated 
        in the lower row, but flipped for AF order in the upper row.


    The solver is chosen from the ``solver`` argument. For ``solver=='CT-AUX'`` 
    there are some additional options: ``CT_sweeps`` sets the number of 
    iterations in the solver for a single impurity problem, and 
    ``parallelize_CT`` sets whether the solver itself is parallelized (i.e. 
    sweeps are split over different cores) or the solver is parallelized over 
    many sites. Note that ``mu`` should be provided without the CT correction.
    I.e. :math:`\mu=U/2` corresponds to half-filling for CT and ED. For more information 
    about the CT solver, see :mod:`CT_AUX_SM`.

    For ``solver=='EDx'`` the ED solver is chosen with x orbitals. To speed the solver's minimization of 
    Anderson parameters, one can also provide a list in ``ED_params`` where each element of the list 
    corresponds to the impurity calculation on one site and has an array of Anderson parameters. See 
    :mod:`ED_new` for further information.

    This function is parallelized with the ``multiprocessing`` package. The Green's function calculations 
    will be split onto different cores for different omega values. For the ED solver, the calculations will 
    be parallelized over each site. For the CT-AUX solver, either the solver itself is parallelized (as given 
    in the ``parallelize_CT`` input argument) without parallelizing over the sites, or the code is parallelized 
    over the sites without parallelizing the solver itself. If you would like to disable multiprocessing 
    entirely, you can do so by::

        import dancommon
        dancommon.disable_mp = True

    **Additional notes:**

        * The diagonal spin self-energies are enforced to be real in imaginary 
          time.
        * The cross-spin terms fit the relation 
          :math:`f_{\sigma\bar\sigma}(i\omega) = 
          f^*_{\bar\sigma\sigma}(-i\omega_n)` where f stands for either the Green's functions 
          or the self-energy.
        * There exists a ``fit_large_SE`` option that is only valid for the 
          case without spin-mixing. It turned out that this only hides problems, rather than 
          help with the self-consistency loops.

    '''
    from numpy import fromfile,zeros,diag,zeros_like,pi,imag,abs,empty
    from numpy.linalg import inv
    import multiprocessing, subprocess, tempfile, time, os

    if CT_sweeps == None:
        CT_sweeps = 1e6
    size = N*M
    maxomega = len(omega)

    ########
    # Determine how many sites must be calculated
    if symmetry == None:
        symmetry = [None] * N*M

    # Make a local copy to modify
    symmetry = list(symmetry)

    assert len(symmetry) == N*M
    num_calc_sites = symmetry.count(None)
    assert num_calc_sites > 0
    assert all(z == None or -num_calc_sites <= z < num_calc_sites for z in symmetry)

    # Turn the Nones into indexes as well
    for i in xrange(num_calc_sites):
        symmetry[symmetry.index(None)] = i
    assert None not in symmetry


    assert solver == 'CT-AUX' or solver.startswith('ED')

    if solver == 'CT-AUX':
        mu = mu - U/2.
    elif solver.startswith('ED'):
        # The number of orbitals is passed with the solver string
        # e.g. ED4 for a four-orbital ED solver.
        num_l = int(solver[2:])
        solver = 'ED'
        assert len(ED_params) == num_calc_sites

    if SE == None:
        SE_up = zeros((maxomega,num_calc_sites),complex)
        SE_down = zeros((maxomega,num_calc_sites),complex)
        SE_cross = zeros((maxomega,num_calc_sites),complex)
    else:
        SE_up,SE_down,SE_cross = SE

    if potential == None:
        potential = zeros((maxomega,num_calc_sites),complex)

    SE_up = SE_up.astype(complex)
    SE_down = SE_down.astype(complex)
    SE_cross = SE_cross.astype(complex)
    potential = potential.real.astype(float)

    assert SE_up.shape == (maxomega,num_calc_sites)
    assert SE_down.shape == (maxomega,num_calc_sites)
    assert SE_cross.shape == (maxomega,num_calc_sites)
    assert potential.shape == (num_calc_sites,)


    #################
    # Calculate the Green's function at each site

    # Multiprocessing
    from dancommon import CreatePool,FakePool
    num,pool = CreatePool()

    res = []
    start_greensinv = time.time()

    # Only calculate for negative frequencies, because we can build the rest up 
    # from symmetry.
    for i in range(maxomega/2):
        # For each frequency, we must generate the complete grid of self-energy 
        # from the symmetries and also with the block of four SE functions.
        SE_temp = empty((4,N*M),complex)
        pot_temp = empty((N*M),float)
        for j,index in enumerate(symmetry):
            pot_temp[j] = potential[index]
            if index < 0:
                SE_temp[0][j] = SE_down[i][index]
                SE_temp[1][j] = SE_up[i][index]
                #SE_temp[2][j] = SE_cross[-i-1][index].conj()
                #SE_temp[3][j] = SE_cross[i][index]
                SE_temp[2][j] = -SE_cross[i][index]
                SE_temp[3][j] = -SE_cross[-i-1][index].conj()
            else:
                SE_temp[0][j] = SE_up[i][index]
                SE_temp[1][j] = SE_down[i][index]
                SE_temp[2][j] = SE_cross[i][index]
                SE_temp[3][j] = SE_cross[-i-1][index].conj()

        #cmdline = [Ccode_location + 'RDMFT_TISM_greensinv',output_filename,boundaries,str(N),str(M),repr(omega[i].real),repr(omega[i].imag),repr(t),str(p),str(q),repr(mu),repr(gamma),SE_filename,potential_filename,'N','Y']
        #res += [pool.apply_async(subprocess.check_call,(cmdline,))]

        # Changed to using the new python code.
        res += [pool.apply_async(GreensMatSinglePython,(SE_temp,pot_temp,(N,M),boundaries,omega[i],(p,q),mu,gamma,False,False))]

    # Check for exceptions
    logger.info("Waiting for RDMFT_greensinv to finish...")
    pool.close()
    pool.join()
    for i in range(maxomega/2):
        res[i] = res[i].get()
    logger.info("... finished in {0:.5} seconds".format(time.time() - start_greensinv))

    ##############
    # Calculate the Weiss Green's functions from the results.
    weiss_up = zeros((maxomega,num_calc_sites),complex)
    weiss_down = zeros((maxomega,num_calc_sites),complex)
    weiss_cross = zeros((maxomega,num_calc_sites),complex)

    G_up = zeros((maxomega,num_calc_sites),complex)
    G_down = zeros((maxomega,num_calc_sites),complex)
    G_cross = zeros((maxomega,num_calc_sites),complex)

    # Only do negative frequencies, because we can build the rest up from 
    # symmetry
    for i in range(maxomega/2):
        # Only look at symmetry sites.
        for index in range(num_calc_sites):
            # Find an appropriate index into the complete grid to retrieve the 
            # Green's functions.
            site = symmetry.index(index)

            # Need to convert to the 2x2 matrix forms to apply the Dyson 
            # equation.
            G_mat = empty((2,2),complex)
            G_mat[0,0] = res[i][:,:,0].T.flat[site]
            G_mat[0,1] = res[i][:,:,2].T.flat[site]
            G_mat[1,0] = res[i][:,:,3].T.flat[site]
            G_mat[1,1] = res[i][:,:,1].T.flat[site]
            SE_mat = array([[SE_up[i][index], SE_cross[i][index]],[SE_cross[-i-1][index].conj(),SE_down[i][index]]])

            G0_mat = inv(inv(G_mat) + SE_mat)
            weiss_up[i][index] = G0_mat[0,0]
            weiss_down[i][index] = G0_mat[1,1]
            weiss_cross[i][index] = G0_mat[0,1]

            G_up[i][index] = G_mat[0,0]
            G_down[i][index] = G_mat[1,1]
            G_cross[i][index] = G_mat[0,1]

            # Now we can also fill in the positive omega terms, respecting 
            # symmetry.
            weiss_up[-i-1][index] = weiss_up[i][index].conj()
            weiss_down[-i-1][index] = weiss_down[i][index].conj()
            weiss_cross[-i-1][index] = G0_mat[1,0].conj()
            G_up[-i-1][index] = G_up[i][index].conj()
            G_down[-i-1][index] = G_down[i][index].conj()
            G_cross[-i-1][index] = G_mat[1,0].conj()

    # Save everything to a file for later use if we need it.
    import numpy
    numpy.savez('fromgreensinv_iter'.format(iter),weiss=(weiss_up,weiss_down,weiss_cross),G=(G_up,G_down,G_cross),G_mat=G_mat,SE_mat=SE_mat)

    # Double check for NaN issues
    if (weiss_up != weiss_up).any() or (weiss_down != weiss_down).any():
        raise Exception("NaNs found from C code!")

    # Enforce the realness of the imaginary time Green's functions that are 
    # diagonal in spin.
    weiss_up[:maxomega/2] = weiss_up[maxomega/2:][::-1].conj()
    weiss_down[:maxomega/2] = weiss_down[maxomega/2:][::-1].conj()

    #############
    # Do the impurity calculations
    if parallelize_CT and solver == 'CT-AUX':
        pool = FakePool()
    else:
        num,pool = CreatePool()

    res = []
    for i in range(num_calc_sites):
        if solver == 'CT-AUX':
            import CT_AUX_SM
            if parallelize_CT:
                res += [pool.apply_async(CT_AUX_SM.CT_AUX_SM_Parallel,(U,beta,(weiss_up[:,i],weiss_down[:,i],weiss_cross[:,i]),CT_sweeps))]
            else:
                res += [pool.apply_async(CT_AUX_SM.CT_AUX_SM,(U,beta,(weiss_up[:,i],weiss_down[:,i],weiss_cross[:,i]),CT_sweeps))]
        elif solver == 'ED':
            import ED_new
            res += [pool.apply_async(ED_new.EDSMGreens,(omega,U,mu,beta,(weiss_up[:,i],weiss_down[:,i],weiss_cross[:,i])),{'num_l':num_l,'start_params':ED_params[i],'return_weiss':True})]


    ###################
    # Calculate the new self-energies from the solver output.

    # Create arrays to return the output
    new_SE_up = zeros((maxomega,num_calc_sites),complex)
    new_SE_down = zeros((maxomega,num_calc_sites),complex)
    new_SE_cross = zeros((maxomega,num_calc_sites),complex)
    new_G_up = zeros((maxomega,num_calc_sites),complex)
    new_G_down = zeros((maxomega,num_calc_sites),complex)
    new_G_cross = zeros((maxomega,num_calc_sites),complex)
    # The extra +1 here is due to the output coming from the solver being based 
    # on the number of omega points. 
    new_G_T = zeros((3,maxomega+1,num_calc_sites),complex)
    N_up = zeros(num_calc_sites)
    N_down = zeros(num_calc_sites)
    N_dbl = zeros(num_calc_sites)
    Sx = zeros(num_calc_sites)
    Sy = zeros(num_calc_sites)
    Sz = zeros(num_calc_sites)

    new_W_T = zeros((maxomega+1,num_calc_sites),complex)
    
    for i in range(num_calc_sites):
        if solver =='CT-AUX':
            Gw,GT,pert_order,dbl_occ,WT,Gworig,Gwmatsu = res[i].get()
            new_G_up[:,i] = Gw[0]
            new_G_down[:,i] = Gw[1]
            new_G_cross[:,i] = Gw[2]
            N_up[i] = -GT[0][-1].real
            N_down[i] = -GT[1][-1].real
            N_dbl[i] = dbl_occ
            new_G_T[:,:,i] = GT
            new_W_T[:,i] = WT

            Sx[i] = -GT[2][-1].real
            Sy[i] = -GT[2][-1].imag
            Sz[i] = 1/2. * (N_up[i] - N_down[i])
        elif solver == 'ED':
            Gw,Nf,ED_params[i],W = res[i].get()
            new_G_up[:,i] = Gw[0]
            new_G_down[:,i] = Gw[1]
            new_G_cross[:,i] = Gw[2]
            N_up[i] = Nf[0]
            N_down[i] = Nf[1]
            N_dbl[i] = Nf[3]
            new_G_T[:,:,i] = -1
            new_W_T[:,i] = -1
            # We need to update the weiss functions that were the best fit of 
            # the ED code
            weiss_up[:,i] = W[0]
            weiss_down[:,i] = W[1]
            weiss_cross[:,i] = W[2]

            Sx[i] = Nf[2].real
            Sy[i] = Nf[2].imag
            Sz[i] = 1/2. * (N_up[i] - N_down[i])


        # Convert to the 2x2 matrix forms for the Dyson equation and calculate 
        # the new self-energy.
        for freq in range(maxomega):
            G_mat = array([[Gw[0][freq],Gw[2][freq]],[Gw[2][-freq-1].conj(),Gw[1][freq]]])
            G0_mat = array([[weiss_up[freq,i],weiss_cross[freq,i]],[weiss_cross[-freq-1,i].conj(),weiss_down[freq,i]]])
            SE_mat = inv(G0_mat) - inv(G_mat)
            new_SE_up[freq,i] = SE_mat[0,0]
            new_SE_down[freq,i] = SE_mat[1,1]
            new_SE_cross[freq,i] = SE_mat[0,1]

        if fit_large_SE:
            # This is only valid without spin-mixing. It turned out that this 
            # only hides problems and is not of real benefit.
            # TODO: This is temporarily set to the non-spin-mixing form for 
            # testing.
            assert gamma == 0.
            from numpy import arange
            fit_ind = arange(maxomega) > fit_large_SE 
            new_SE_up[fit_ind,i] = U*(N_down[i] - 0.5) + U**2/(1j*omega[fit_ind]) * N_down[i] * (1 - N_down[i])
            new_SE_down[fit_ind,i] = U*(N_up[i] - 0.5) + U**2/(1j*omega[fit_ind]) * N_up[i] * (1 - N_up[i])
            new_SE_cross[fit_ind,i] = 0.

        logger.info("Site {0}, Nf = {1},{2} = {3} and {4}, Nf_dbl = {5}".format(i,N_up[i],N_down[i],N_up[i] + N_down[i],N_up[i] - N_down[i],N_dbl[i]))

    pool.close()
    pool.join()
    
    return (new_SE_up,new_SE_down,new_SE_cross),N_up,N_down,N_dbl,(Sx,Sy,Sz),(new_G_up,new_G_down,new_G_cross),(weiss_up,weiss_down,weiss_cross),(G_mat,SE_mat,G0_mat),new_G_T,new_W_T

def ConvergenceSpin(U,mu,beta,gamma,potential,num_omega,(N,M),boundaries,t,(p,q),SE=None,max_iters=100,symmetry=None,use_tmpdir=False,force_para=False,CT_sweeps=None,row_sym=None,fit_large_SE=False,force_filling=False,continue_from_iter=False,solver=None,randomize=True):
    r'''This function is the glue between the self-consistency loops. It 
    calculates any quantities that are required between loops and performs dampening, etc...

    The parameters ``U,mu,gamma,potential,(N,M),boundaries,t,(p,q),SE,symmetry,CT_sweeps,fit_large_SE,solver`` are 
    the same as for :func:`SelfConsistencySpin` and are mostly passed on directly.

    The array of omega frequencies is calculated from ``num_omega`` and ``beta`` using the :mod:`ImagConv` module.
    Note that ``num_omega`` specifies the number of *positive* frequencies only, which will be doubled in 
    the output to include the negative Matsubara frequencies.

    The file ``info.dat``, which should already exist from a higher-level function, will be 
    appended to, in order to record the progress. At the current time, the following information 
    is saved in a tabulated format:

        * ``iter``: The iteration number.
        * ``SE_err``: The error in the self-energy, calculated by integrating 
                      over ``omega`` the difference of ``SE`` from one iteration to the next, 
                      for each site and then averaging over all sites. Each component :math:`\Sigma_{\sigma\sigma'}` 
                      is calculated separately and averaged.
        * ``G_err``: The estimated error in the Green's functions, calculated 
                     as for ``SE_err``. Only the imaginary part is shown.
        * ``N_up_err``: The largest difference of N_up on all sites, from the 
                        last iteration.
        * ``N_up_err2``: As for ``N_up_err`` but from two iterations previous.
        * ``N_tot``: The average total filling.
        * ``Sx``: The average (absolute) spin projected along x-axis.
        * ``Sy``: The average (absolute) spin projected along y-axis.
        * ``Sz``: The average (absolute) spin projected along z-axis.
        * ``cur_mu``: Current value of chemical potential. Only relevant for 
                      finite ``force_filling``.

    An example ``info.dat`` file follows: 
    
    .. include:: /example_figures/RDMFT_TISM/info.dat
        :literal:

    In addition, several other files are created. The files ``Nup_Ndown_<iter>.dat`` list 
    ``Nup,Ndown,Sx,Sy,Sz`` in text format for each site of the lattice, as follows:
    
    .. include:: /example_figures/RDMFT_TISM/Nup_Ndown_10.dat
        :literal:
        
    The files ``SE_iter<iter>.npy`` 
    store the self-energy for all sites which are evaluated in the impurity solver in the 
    numpy ``.npy`` format. The files ``everything_iter<iter>.npz`` store much more information in
    a numpy ``.npz`` format, including self-energy, Green's functions, weiss functions, 
    imaginary time functions, and double occupancy. For information on how to read the numpy 
    file formats, see :ref:`reading-formats`.

    The function loops over the self-consistency for up to ``max_iters``, unless convergence is reached in this 
    time. At the moment there is no method with which to adjust the convergence, and it is hard-coded 
    to be that the error in ``N_up_err`` and ``Im_G_err`` is ``< 1e-4``. If force_filling is also enabled, 
    then the filling difference to the desired filling must be ``< 1e-4``.

        Note that these tight criteria will only ever be reached for the ED solver. In the CT-AUX case, 
        one should not (and really cannot) rely on such criteria, due to the jitter of the solver. 
        One should instead look at how the results behave over many iterations.

    The argument ``continue_from_iter`` has no numerical effect here, and only affects the formatting of 
    the output files. See :func:`TI_CTAUX` for more information.

    To allow for different magnetic order, the self-energies of the first two iterations are randomized if 
    the parameter ``randomize==True``. That is, all of the self-energy functions are averaged for each 
    omega

    .. math::
        A(i \omega_n) = \sum_{\sigma,\sigma',i} \Sigma_{\sigma,\sigma',i}(i\omega_n)

    and then this averaged quantity is spread over the self-energy with a random coefficient. That is:

    .. math::
        \Sigma'_{\sigma,\sigma',i}(i \omega_n) = C_{\sigma,\sigma',i,n} A(i \omega_n)

    where C is a random variable that is evenly distributed over 0-1.

    If the parameter ``force_filling`` is False, then there the chemical potential is never adjusted. 
    Otherwise ``force_filling`` must be a filling value, which is the target filling, as measured by 
    ``N_tot``, described above. The procedure for adjusting the filling is:


        #. Run for a few (five) iterations without any adjustments.
        #. Wait for a global filling error of < 0.01 to activate the search of filling.
        #. When the search is activated, if ``N_up_err`` is > 0.05 then do nothing special 
           this iteration (this is to prevent false bounding below). Otherwise:

             #. If the filling is above the target, set ``max_mu`` as a bound.
             #. If the filling is below the target, set ``min_mu`` as a bound.
             #. If the target filling is bounded by ``max_mu`` and ``min_mu`` then bisect this region 
                 to choose the new ``mu``.
             #. If the target filling is not bounded, adjust ``mu`` in the right direction by a change 
                 ``dmu = max(0.5,U/10)``.

    **Notes:**

        The code contains some very weak dampening using only the previous iteration,

        .. math:: 

            \Sigma_n' = \alpha \Sigma_n + (1-\alpha) \Sigma_{n-1}

        which can be adjusted in the function itself. Currently ``alpha=0.9``.

        It is possible to force paramagnetic behaviour by setting the argument ``force_para=True``. 
        However this does not take into account the cross terms and so only works correctly without 
        spin-mixing.

        An option ``use_tmpdir`` is provided, which allows one to run this code is a randomly 
        generated temporary directory. This is not recommended, as all useful output is 
        actually saved to files in the directory.

    **Returns:**
        ``SE,N_up,N_down,N_dbl,G``

        ``SE`` (complex double array)
            The self-energy of the sites where the impurity problem is calculated.
        ``N_up`` (double array)
            The filling of the up spin at each site.
        ``N_down`` (double array)
            As above for down spin.
        ``N_dbl`` (double array)
            Double occupancy at each site.
        ``G`` (complex double array)
            Green's function in Matsubara frequency at calculated sites.

        Note that the return values of this function are not necessary, as all information 
        is also saved to various files.
    '''
    
    from numpy import abs,array,r_

    # Make tmp directory for files
    import os
    cwd = os.getcwd()
    if use_tmpdir:
        import tempfile
        tmpdir = tempfile.mkdtemp(dir=os.path.expanduser('~/temp'))
        os.chdir(tmpdir)

    NRG_previous = None

    N_up_list = []
    N_down_list = []
    N_dbl_list = []
    SE_list = []
    G_list = []

    cur_mu = mu
    import ImagConv
    omega = ImagConv.OmegaN(num_omega,beta) 
    # Allow for positive and negative frequencies
    omega = r_[-omega[::-1],omega]

    # This code assumes the function above has already set up an info.dat file
    # with the basic parameters in it
    info_filename = 'info.dat'
    if continue_from_iter == False:
        with open(info_filename,'ab') as f:
            f.write("iter | SE_err | G_err | N_up_err | N_up_err2 | N_tot | Sx | Sy | Sz | cur_mu \n")
        # This is to make the for loop later behave
        continue_from_iter = -1

    # Setup for force_filling
    cur_mu = mu
    low_mu = None
    low_filling = None
    high_mu = None
    high_filling = None
    activate_filling_search = False

    # Setup for ED params (if needed)
    if solver == 'CT-AUX':
        ED_params = None
    else:
        num_calc_sites = symmetry.count(None)
        ED_params = [None] * num_calc_sites

    try:
        print(continue_from_iter)
        print(max_iters)
        for iter in range(continue_from_iter+1,max_iters):
            old_SE = SE

            # Dampening
            if len(SE_list) >= 2:
                #alpha = 0.6
                #alpha = 1.0
                alpha = 0.9
                SE = alpha*SE_list[-1] + (1-alpha)*SE_list[-2]
            print([SE,SE_list])
          

            if force_para and SE != None:
                SE[0] = (SE[0]+SE[1])/2.
                #SE[1] = (SE[0]+SE[1])/2.
            if SE != None:
                # Here we force the SE to be real (in imaginary time) for the 
                # spin-diagonal components.
                SE[0][:num_omega] = (SE[0][num_omega:][::-1]).conj()
                SE[1][:num_omega] = (SE[1][num_omega:][::-1]).conj()

            SE,N_up,N_down,N_dbl,(Sx,Sy,Sz),G,weiss,test_mat,GT,WT = SelfConsistencySpin(U,cur_mu,beta,gamma,SE,potential,1j*omega,(N,M),boundaries,t,(p,q),symmetry=symmetry,CT_sweeps=CT_sweeps,fit_large_SE=fit_large_SE,solver=solver,ED_params=ED_params)

            # Convert to arrays for easier handling of averaging
            SE = array(SE)
            G = array(G)

            # Add some randomness in to allow for better selection of magnetic 
            # order (first two iterations only).
            if randomize and iter < 2:
            #if False:
                logger.info("Randomising self-energy")
                from numpy.random import rand
                old_SE = SE.copy()
                SE[:] = 0.
                for site in range(old_SE.shape[2]):
                    for func in range(old_SE.shape[0]):
                        SE[func,:,site] = (rand(*old_SE.shape) * old_SE).sum(axis=2).sum(axis=0) / old_SE.shape[2] / old_SE.shape[0]
            else:
                old_SE = None

            N_up_list += [N_up]
            N_down_list += [N_down]
            N_dbl_list += [N_dbl]
            SE_list += [SE]
            G_list += [G]

            # Only store up to three
            store_amount = 3
            N_up_list = N_up_list[-store_amount:]
            N_down_list = N_down_list[-store_amount:]
            N_dbl_list = N_dbl_list[-store_amount:]
            SE_list = SE_list[-store_amount:]
            G_list = G_list[-store_amount:]

            with open('Nup_Ndown_{0}.dat'.format(iter),'wb') as file:
                file.write('Nup Ndown Sx Sy Sz\n')
                for i in range(len(N_up)):
                    file.write('{0:9.5f} {1:9.5f} {2:9.5f} {3:9.5f} {4:9.5f}\n'.format(float(N_up[i]), float(N_down[i]),float(Sx[i]),float(Sy[i]),float(Sz[i])))

            def CalcIntDiff(old,new):
                if old == None:
                    return None

                from scipy.integrate import trapz
                err = 0.
                diff = abs(new - old)
                for func in range(diff.shape[0]):
                    for site in range(diff.shape[2]):
                        err += trapz(diff[func,:,site],omega)
                # Divide by the number of lattice sites
                err /= diff.shape[2]*3
                err = float(err.real)
                return err


            if len(N_up_list) >= 2:
                N_up_err = abs(N_up_list[-1] - N_up_list[-2]).max()
                Im_SE_err = CalcIntDiff(SE_list[-2].imag,SE_list[-1].imag)
                Im_G_err = CalcIntDiff(G_list[-2],G_list[-1])
            else:
                #Im_SE_err = Im_G_err = N_up_err = N_up_err2 = None
                Im_SE_err = Im_G_err = N_up_err = N_up_err2 = -1

            if len(N_up_list) >= 3:
                N_up_err2 = abs(N_up_list[-1] - N_up_list[-3]).max()
            else:
                #N_up_err2 = None
                N_up_err2 = -1
                
            if row_sym == 'AF':
                # Need to complete the row
                N_tot = (sum(N_up_list[-1][:-1])*2 + N_up_list[-1][-1]) / N
                N_tot += (sum(N_down_list[-1][:-1])*2 + N_down_list[-1][-1]) / N
                mag = (sum( abs(N_up_list[-1][:-1] - N_down_list[-1][:-1])) * 2 + abs(N_up_list[-1][-1] - N_down_list[-1][-1]) ) / N
                # Technically this is not correct, but it's with open 
                # boundaries anyway, so I don't think there's any benefit in 
                # being 100% precise.
                Sx_avg = avg(abs(Sx))
                Sy_avg = avg(abs(Sy))
                Sz_avg = avg(abs(Sz))
            else:
                N_tot = avg(N_up_list[-1]) + avg(N_down_list[-1])
                mag = avg(abs(N_up_list[-1] - N_down_list[-1]))
                Sx_avg = avg(abs(Sx))
                Sy_avg = avg(abs(Sy))
                Sz_avg = avg(abs(Sz))
                
            with open(info_filename,'a') as f:
                formatspec = '9.5f'
                f.write("{1:2} {2:{0}} {3:{0}} {4:{0}} {5:{0}} {6:{0}} {7:{0}} {8:{0}} {9:{0}} {10:{0}}\n".format(formatspec,iter,Im_SE_err,Im_G_err,N_up_err,N_up_err2,N_tot,Sx_avg,Sy_avg,Sz_avg,cur_mu))
            # Output the SE at each iteration
            with FileLock():
                from numpy import save,savez
                #file = gzip.open('SE_iter{0}.npy'.format(iter),'wb')
                save('SE_iter{0}.npy'.format(iter),SE)

                # Temporarily output everything
                savez('everything_iter{0}.npz'.format(iter),SE=SE,G=G,weiss=weiss,test_mat=test_mat,GT=GT,WT=WT,old_SE=old_SE,N_dbl=N_dbl)

            # Check to see if convergence has been reached.
            if Im_G_err < 1e-4 and N_up_err < 1e-4 and (force_filling == None or N_tot - force_filling < 1e-4) and \
               len(N_up_list) >= 3:
                logger.info("Reached convergence, breaking out of loop")
                #break

            if force_filling != None:
                if not activate_filling_search:
                    if iter > 5 and N_up_err != None and N_up_err < 1e-2:
                        activate_filling_search = True
                        logger.info("Activating filling search")

                if activate_filling_search:
                    cur_filling_diff = N_tot - force_filling
                    #if abs(cur_filling_diff) < 1e-3:
                    #if False:
                    if abs(cur_filling_diff) < N_up_err*2:
                        # Don't bother trying to get any closer
                        logger.info("Not bothering to adjust filling - it is close enough")
                        pass
                    elif abs(N_up_err) > 5e-2:
                        # If we have moved too fast, then slow down, in case we 
                        # are too restrictive with the bounds too early.
                        logger.info("Not adjusting mu - need more convergence")
                    else:
                        mu_motion = max(0.5,U*0.1)
                        # Try to get close to the right filling
                        if low_mu == None or high_mu == None:
                            # Guess in the right direction
                            if cur_filling_diff < 0:
                                low_mu = cur_mu
                                low_filling = N_tot
                                cur_mu += mu_motion
                                logger.info("Set low mu")
                            else:
                                high_mu = cur_mu
                                high_filling = N_tot
                                cur_mu -= mu_motion
                                logger.info("Set high mu")

                            # If we are bounded then override the cur_mu to be 
                            # the middle point (bisecting)
                            if low_mu != None and high_mu != None:
                                cur_mu = (low_mu + high_mu)/2.
                                logger.info("Bounded filling")
                        else:
                            # We must have bounds on the filling at this point
                            if cur_filling_diff < 0:
                                low_mu = cur_mu
                                low_filling = N_tot
                                logger.info("Updated low mu")
                            else:
                                high_mu = cur_mu
                                high_filling = N_tot
                                logger.info("Updated high mu")
                            cur_mu = (low_mu + high_mu)/2.
                
    finally:
        os.chdir(cwd)
        if use_tmpdir:
            try:
                import shutil
                shutil.rmtree(tmpdir)
            except OSError:
                logger.warning("Ignoring OSError")
                pass

    return SE_list[-1],N_up_list[-1],N_down_list[-1],N_dbl_list[-1],G_list[-1]

def FourierTransformSpin(U,mu,gamma,SE,potential,omega,(N,M),boundaries,t,(p,q),symmetry=None,y_only=False,no_transform=False,only_omega0=False,complete_mat=False,no_invert=False,output_filename=None,matsubara=None):
    r'''Calculate the transformed Green's functions with a given self-energy. The resultant functions 
    are not returned and are saved to a ``.npz`` file with filename given by ``output_filename``. 
    See :ref:`reading-formats` for more information.

    The system parameters ``U,mu,gamma,(N,M),boundaries,t,(p,q)`` are described in 
    :func:`GreensMatSinglePython`.

    The frequencies on which to calculate are given in ``omega``. These may be either 
    Matsubara or real frequencies. If Matsubara frequencies are used, then one should 
    also set ``matsubara=True`` as described below.

    The lattice symmetries, as given by ``symmetry`` and used by ``SE`` and ``potential`` 
    is described in :func:`SelfConsistencySpin` and :func:`TI_Symmetry`. On top of this, 
    it is necessary to know the symmetry of :math:`G_{\sigma,\bar\sigma}^*(\omega)` which 
    is different for real and Matsubara frequencies. Hence, one must specify ``matsubara`` 
    appropriately.

    If ``y_only==True`` then the transformed functions are G(x,k_y). If ``y_only==False`` 
    then the transformed functions are G(k_x,k_y). If ``no_transform==True`` then no Fourier 
    transform is performed at all.

    If ``only_omega0==True`` then the omega value closest to zero in the specified list 
    ``omega`` is used. This is useful for looking at behaviour near the Fermi edge.

    Normally, only the diagonal part of the total Green's matrix is returned, however one 
    can access the entire matrix with ``complete_mat=True``. Note that this is slower to 
    calculate.

    One can also access the single-particle Hamiltonian by specifying ``no_invert=True``.

    **Notes:**

        The input ``mu`` should be adjusted appropriately, if the self-energies have been 
        calculated with the CT-AUX difference of ``mu-U/2`` (i.e. this function does not 
        do any adjustment, unlike :func:`SelfConsistencySpin`. See :func:`TI_FourierCTAUX` 
        for an example.

    '''
    from numpy import fromfile,zeros,diag,zeros_like,pi,imag,abs,empty
    import multiprocessing
    import subprocess

    assert output_filename != None

    # Determine how many sites must be calculated
    if symmetry == 'column':
        raise NotImplementedError
    elif symmetry == None:
        symmetry = [None] * N*M

    # Make local copy and ensure it is a list
    symmetry = list(symmetry)
    assert len(symmetry) == N*M
    num_calc_sites = symmetry.count(None)
    assert num_calc_sites > 0
    assert all(z == None or -num_calc_sites <= z < num_calc_sites for z in symmetry)
    # Turn the Nones into indexes as well
    for i in xrange(num_calc_sites):
        symmetry[symmetry.index(None)] = i
    assert None not in symmetry
    assert matsubara in [True,False]

    size = N*M

    totomega = len(omega)
    if matsubara:
        maxomega = len(omega)/2
    else:
        maxomega = totomega

    assert len(SE) == 3
    SE_up,SE_down,SE_cross = SE

    if SE_up == None:
        SE_was_None = True
        SE_up = zeros((totomega,num_calc_sites),complex)
        SE_down = zeros((totomega,num_calc_sites),complex)
        SE_cross = zeros((totomega,num_calc_sites),complex)
    if potential == None:
        potential = zeros((totomega,num_calc_sites))

    SE_up = SE_up.astype(complex)
    SE_down = SE_down.astype(complex)
    SE_cross = SE_cross.astype(complex)
    potential = potential.real.astype(float)

    assert SE_up.shape == (totomega,num_calc_sites)
    assert SE_down.shape == (totomega,num_calc_sites)
    assert SE_cross.shape == (totomega,num_calc_sites)
    assert potential.shape == (num_calc_sites,)

    from dancommon import CreatePool
    num_procs,pool = CreatePool()

    res = []
    import tempfile
    tmpdir = tempfile.mkdtemp() + '/'
    try:
        save_omega = []
        if only_omega0:
            if matsubara:
                omega_list = [maxomega]
            else:
                omega_list = [abs(omega).argmin()]
        else:
            omega_list = range(maxomega)

        for i in omega_list:
            save_omega += [omega[i]]

            SE_temp = empty((4,N*M),complex)
            pot_temp = empty((N*M))
            for j,index in enumerate(symmetry):
                pot_temp[j] = potential[index]
                if index < 0:
                    SE_temp[0][j] = SE_down[i][index]
                    SE_temp[1][j] = SE_up[i][index]
                    SE_temp[2][j] = -SE_cross[i][index]
                    if matsubara:
                        SE_temp[3][j] = -SE_cross[-i-1][index].conj()
                    else:
                        SE_temp[3][j] = -SE_cross[i][index].conj()
                else:
                    SE_temp[0][j] = SE_up[i][index]
                    SE_temp[1][j] = SE_down[i][index]
                    SE_temp[2][j] = SE_cross[i][index]
                    if matsubara:
                        SE_temp[3][j] = SE_cross[-i-1][index].conj()
                    else:
                        SE_temp[3][j] = SE_cross[i][index].conj()

            if no_transform:
                #G_filename = tmpdir + 'Gx_{0}.raw'.format(i)
                #cmdline = [Ccode_location + 'RDMFT_TISM_greensinv',G_filename,boundaries,str(N),str(M),repr(omega[i].real),repr(omega[i].imag),repr(t),str(p),str(q),repr(mu),repr(gamma),SE_filename,potential_filename,no_invert_param,return_diag_param]
                #res += [pool.apply_async(subprocess.check_call,(cmdline,))]
                res += [pool.apply_async(GreensMatSinglePython,(SE_temp,pot_temp,(N,M),boundaries,omega[i],(p,q),mu,gamma,no_invert,complete_mat))]
            else:
                assert not no_invert
                #G_filename = tmpdir + 'Gk_{0}.raw'.format(i)
                #cmdline = [Ccode_location + 'RDMFT_TISM_trans',G_filename,boundaries,str(N),str(M),repr(omega[i].real),repr(omega[i].imag),repr(t),str(p),str(q),repr(mu),repr(gamma),SE_filename,potential_filename,y_only_param]
                #res += [pool.apply_async(subprocess.check_call,(cmdline,))]
                res += [pool.apply_async(FourierTransformSpinSingle,(SE_temp,pot_temp,(N,M),boundaries,omega[i],(p,q),mu,gamma,no_invert,y_only,complete_mat))]

        # Check for exceptions
        logger.info("Waiting for RDMFT_trans to finish...")

        # Collect results into one file
        if no_transform:
            file_prefix = 'Gx'
        else:
            file_prefix = 'Gk'

        from numpy import empty,fromfile,array,r_
        save_omega = array(save_omega)
        if matsubara:
            save_omega = r_[save_omega,-save_omega[::-1]]
        savelen = len(save_omega)
        if complete_mat:
            greens_up = empty((N,M,N,M,savelen),complex)
            greens_down = empty((N,M,N,M,savelen),complex)
            greens_cross = empty((N,M,N,M,savelen),complex)
        else:
            greens_up = empty((N,M,savelen),complex)
            greens_down = empty((N,M,savelen),complex)
            greens_cross = empty((N,M,savelen),complex)

        import os
        for i in range(len(res)):
            res[i] = res[i].get()
            if i % num_procs == 0:
                logger.info("Done omega index {0}".format(i))
            if complete_mat:
                greens_up[:,:,:,:,i] = res[i][:,:,0,:,:,0]
                greens_down[:,:,:,:,i] = res[i][:,:,1,:,:,1]
                greens_cross[:,:,:,:,i] = res[i][:,:,0,:,:,1]
                if matsubara:
                    greens_up[:,:,:,:,-i-1] = res[i][:,:,0,:,:,0].conj()
                    greens_down[:,:,:,:,-i-1] = res[i][:,:,1,:,:,1].conj()
                    greens_cross[:,:,:,:,-i-1] = res[i][:,:,1,:,:,0].conj()
            else:
                greens_up[:,:,i] = res[i][:,:,0]
                greens_down[:,:,i] = res[i][:,:,1]
                greens_cross[:,:,i] = res[i][:,:,2]
                if matsubara:
                    greens_up[:,:,-i-1] = res[i][:,:,0].conj()
                    greens_down[:,:,-i-1] = res[i][:,:,1].conj()
                    greens_cross[:,:,-i-1] = res[i][:,:,3].conj()
            res[i] = None
            import gc
            gc.collect()

        from numpy import savez
        savez(output_filename,omega=save_omega,greens=[greens_up,greens_down,greens_cross])

        pool.close()
        pool.join()

    finally:
        try:
            os.rmdir(tmpdir)
        except:
            logger.warning("Couldn't remove temp directory")
            pass

def TI_CTAUX(U,mu,beta,(N,M),boundaries,(p,q),max_iters=100,num_sweeps=1e6,row_sym='AF',num_omega=400,SE=None,lambda_x=0.,gamma=0.,force_filling=None,fit_large_SE=False,continue_from_iter=False,solver=None):
    '''Here is where the main part of the calculation begins. This function sets up all the parameters 
    to be passed to the worker function :func:`ConvergenceSpin`.

    Arguments ``U,mu,beta,(N,M),boundaries,(p,q),max_iters,num_sweeps,num_omega,SE,gamma,force_filling,fit_large_SE,solver`` are passed directly to :func:`ConvergenceSpin` and are documented there.

    Arguments ``row_sym,lambda_x`` are passed directly to :func:`TI_Symmetry` and are documented there. 
    The resultant arrays for ``symmetry`` and ``potential``, which :func:`ConvergenceSpin` 
    requires, are determined there.

    The function will generate a ``info.dat`` file, with a header that describes the parameters specified. 
    See :func:`ReadInfo` for a useful way to parse this file.

    If ``continue_from_iter`` is an integer, an existing calculation will be resumed from that iteration. 
    The self-energy for the iteration specified is loaded from ``SE_iter<iter>.npy`` and if ``force_filling`` 
    is active then the latest value of ``mu`` is found with the help of :func:`ReadInfo`.
    
    If ``SE==None`` then the initial self-energy will be set to zero. However, if a ``SE_init.npy`` 
    file exists in the directory, then this will be read and used as the initial self-energy. This 
    can be useful for using the result of a run with lower ``CT_sweeps``, or starting from a magnetic 
    solution. When this occurs, randomisation is disable in :func:`ConvergenceSpin`.

    Note: although it is possible to use this function to calculate non-interacting properties, 
    it is recommended to use :func:`GenerateAllDataForU0` to directly generate the output.
    '''
    from numpy import ones,zeros,load

    assert solver == 'CT-AUX' or solver.startswith('ED')

    logger.setLevel(logging.INFO)

    t = 1.
    size = N*M

    #print(symmetry)
    symmetry,potential = TI_Symmetry((N,M),row_sym,lambda_x)

    if U == 0:
        # FIXME
        # This is just a dodgy hack for the moment
        num_sweeps=1e3
        max_iters=1
        pass

    if continue_from_iter:
        #data = load('everything_iter{0}.npz'.format(continue_from_iter))
        #SE = data['SE']
        SE = load('SE_iter{0}.npy'.format(continue_from_iter))
        info = ReadInfo()
        mu = info['final_mu']
        randomize = True
    else:
        with open('info.dat','wb') as file:
            file.write('Version {0}\n'.format(file_version))
            file.write('Solver {0}\n'.format(solver))
            file.write('N,M {0} {1}\n'.format(N,M))
            file.write('Boundaries {0}\n'.format(boundaries))
            file.write('U {0}\n'.format(U))
            file.write('mu {0}\n'.format(mu))
            file.write('num_sweeps {0}\n'.format(num_sweeps))
            file.write('num_omega {0}\n'.format(num_omega))
            file.write('row_sym {0}\n'.format(row_sym))
            file.write('p,q {0} {1}\n'.format(p,q))
            file.write('beta {0}\n'.format(beta))
            file.write('t {0}\n'.format(t))
            file.write('lambda_x {0}\n'.format(lambda_x))
            file.write('Force_filling {0}\n'.format(force_filling))
            file.write('gamma {0}\n'.format(gamma))
            file.write('fit_large_SE {0}\n'.format(fit_large_SE))
            file.write('-----\n')

        # Check to see if there is a initial SE given in a special file. If so, 
        # use it!
        if SE == None:
            import os
            if os.path.exists('SE_init.npy'):
                SE = load('SE_init.npy')
                logger.info('Using the initial SE that was given!')
                randomize = False
            else:
                randomize = True

    SE,N_up,N_down,N_dbl,G = ConvergenceSpin(U,mu,beta,gamma,potential,num_omega,(N,M),boundaries,t,(p,q),SE=SE,symmetry=symmetry,max_iters=max_iters,force_para=False,CT_sweeps=num_sweeps,row_sym=row_sym,fit_large_SE=fit_large_SE,force_filling=force_filling,continue_from_iter=continue_from_iter,solver=solver,randomize=randomize)

    from numpy import savez
    savez('data.npz',SE=SE,N_up=N_up,N_down=N_down,N_dbl=N_dbl,G=G)


def TI_AnotherIter(high_steps,overwrite_old=False):
    r'''This function exists, so that one may perform an additional iteration 
    at the conclusion of the calculation, with a greater number of sweeps in 
    the CT solver.

    The function will determine the original parameters from the ``info.dat`` file 
    using :func:`ReadInfo` and apply ``high_steps`` instead of the normal ``CT_sweeps`` 
    parameter.

    The output is then saved into the file ``additional.npz`` along with some information 
    in the file ``additional_info.dat``. See :ref:`reading-formats`.

    The argument ``overwrite_old`` is made available, so that one may request 
    not to run this function if some ``additional.npz`` file already exists.
    '''

    import os
    if os.path.exists('additional.npz') and not overwrite_old:
        return

    info = ReadInfo()
    U = info['U']
    final_mu = info['final_mu']
    beta = info['beta']
    gamma = info['gamma']
    row_sym = info['row_sym']
    lambda_x = info['lambda_x']
    num_omega = info['num_omega']
    N = info['N']
    M = info['M']
    boundaries = info['boundaries']
    t = info['t']
    p = info['p']
    q = info['q']

    if U == 0.:
        return

    import ImagConv
    from numpy import r_
    omega = ImagConv.OmegaN(num_omega,beta)
    omega = r_[-omega[::-1],omega]

    symmetry,potential = TI_Symmetry((N,M),row_sym,lambda_x)

    G,SE = AverageData(-5,-1)
    #SE = SE[:,:,:18]

    logger.info("Final mu is {0}".format(final_mu))
    SE,N_up,N_down,N_dbl,(Sx,Sy,Sz),G,weiss,test_mat,GT,WT = SelfConsistencySpin(U,final_mu,beta,gamma,SE,potential,1j*omega,(N,M),boundaries,t,(p,q),symmetry=symmetry,CT_sweeps=high_steps)

    from numpy import savez
    savez('additional.npz',SE=SE,G=G,weiss=weiss,GT=GT,WT=WT)
    
    file = open('additional_info.dat','wb')
    file.write(
'''high_steps {high_steps}
-------------------------------
Info dict read from info.dat:
'''.format(**locals()))
    for key in info:
        file.write("{0} {1!r}\n".format(key,info[key]))
    file.write(
'''
-------------------------------
Fillings
-------------------------------
''')
    for i in range(len(N_up)):
        file.write('{0} {1}\n'.format(float(N_up[i]), float(N_down[i]),float(Sx[i]),float(Sy[i]),float(Sz[i])))
    file.close()


def TI_Symmetry((N,M),row_sym,lambda_x):
    '''This function defines the lattice symmetries. From the arguments ``row_sym``, 
    the arrays ``symmetry`` and ``potential`` which are used in :func:`SelfConsistencySpin` 
    are generated for a grid of size ``NxM`` and staggering ``lambda_x``.

    Details of the structure of ``symmetry`` are specified in :func:`SelfConsistencySpin`, but 
    will be briefly repeated here.

        ``symmetry`` is an array, whose elements are in row-major ordering, indicating sites 
        of the lattice. If an element is ``None`` then it will be fully calculated in the 
        impurity solver. These sites make up a list ``calc_sites``. If an element is a positive integer 
        then this indicates that the site has an identical Green's function with ``calc_sites[i]``. 
        If the element is a negative integer then the site has antiferromagnetic symmetry with 
        ``calc_sites[i]``.

    There are several different types of symmetry possible:

        ``row_sym=='AF'``
            Calculate a half row of the lattice, and mirror this to the rest of the row, and then 
            enforce AF order in the y-direction. Only valid for open boundaries in x with 
            an odd number of sites in the x-direction and gamma = 0 or 0.25.
        ``row_sym=='AFwholerow'``
            Calculate an entire row of the lattice, and enforce AF order in the y-direction.
        ``row_sym=='AFwholerow_quartic<val>'``
            As above, but with an additional trapping in the x direction. <val> is a number that 
            defines the height of the trap at the system edges.
        ``row_sym=='gridAF'``
            Assume only the simplest AF order in periodic case. If ``lambda_x=0`` then only calculate 
            one site and enforce AF order in x and y directions. If ``lambda_x!=0`` then calculate 
            two sites in the x-direction and repeat these in x.
        ``row_sym=='gridAF<val>'``
            Calculate a line of sites <val> long in x, and repeat these along x. Enforce AF order in y. 
            Assumes periodic boundaries.
        ``row_sym=='grid<val>'``
            Calculate a square of sites <val>x<val> (i.e. <val>^2 sites in total). Repeat these along 
            x and y. Assumes periodic boundaries.
        ``row_sym=='rect<x>x<y>'``
            Calculate a recentable of sites <x>x<y> (i.e. <x>*<y> sites in total). Repeat these along 
            x and y. Assumes periodic boundaries.
        ``row_sym==<val>``
            Calculate <val> number of rows and repeat these in y.
        ``row_sym==None``
            Calculate all sites.

    Note that this is the only point in the code at which the parameter ``lambda_x`` is explicitly 
    considered. Afterwards, ``potential`` is all that needed.

    **Returns:** ``symmetry,potential``

        ``symmetry`` (list of ``None`` or integers)
            See description above.
        ``potential`` (double array)
            The local potential at sites that will be calculated in the impurity solver.
    '''
    size = N*M
    from numpy import zeros,arange
    if type(row_sym) == str:
        if row_sym == 'AF':
            symmetry = [None] * size
            #assert lambda_x == 0.
            # Need to assume an odd number of columns for the moment.
            assert N%2 == 1

            max_col = (N-1)/2
            for i in xrange(size):
                col = i%N
                # Mirror symmetry along x center of lattice
                if col > max_col:
                    col = (N-1) - col

                # AF row symmetry
                if (i/N)%2 == 0:
                    symmetry[i] = col
                else:
                    symmetry[i] = -(max_col+1) + col

            for i in xrange(max_col+1):
                symmetry[i] = None
            #potential = zeros(N*2)
            #potential = zeros(N)
            #potential = zeros(max_col+1)
            potential = lambda_x*(-1)**arange(max_col+1)
        elif row_sym.startswith('AFwholerow'):
            assert M%2 == 0
            symmetry = [None] * size
            for i in xrange(size):
                row = int(i/N)
                if row > 0:
                    symmetry[i] = i%N
                    if row%2 == 1:
                        symmetry[i] -= N
            # Add staggering in
            potential = (-1)**arange(N)
            potential = lambda_x*potential
            if 'quartic' in row_sym:
                ind = row_sym.find('quartic') + len('quartic')
                val = float(row_sym[ind:].split('_')[0])
                x = iarange(-(N-1)/2., (N-1)/2., 1.)
                potential += val*abs(x/max(x))**4
        elif row_sym == 'gridAF':
            from numpy import arange,tile
            assert N%2 == 0 and M%2 == 0
            if lambda_x == 0:
                symmetry = array([[0,-1],[-1,0]])
                symmetry = tile(symmetry,(M/2,N/2))
                symmetry = symmetry.flatten().tolist()
                symmetry[0] = None
                potential = array([0.])
            else:
                symmetry = array([[0,1],[-2,-1]])
                symmetry = tile(symmetry,(M/2,N/2))
                symmetry = symmetry.flatten().tolist()
                symmetry[0] = None
                symmetry[1] = None
                potential = array([lambda_x,-lambda_x])
        elif row_sym.startswith('gridAF'):
            # This one is for arbitrary grids in contrast to above
            gridsize = int(row_sym[6:])
            from numpy import arange,tile,r_
            assert N%gridsize == 0 and M%2 == 0
            symmetry = arange(gridsize)[None,:]
            symmetry = r_[symmetry,symmetry-gridsize]
            symmetry = tile(symmetry,(M/2,N/gridsize))
            symmetry = symmetry.flatten().tolist()
            symmetry[:gridsize] = [None]*gridsize
            potential = lambda_x * (-1)**arange(gridsize)
        elif row_sym.startswith('grid'):
            # This is a gridded symmetry useful for periodic boundaries
            gridsize = int(row_sym[4:])

            assert N%gridsize == 0 and M%gridsize == 0
            assert lambda_x == 0 or N%2 == 0

            from numpy import arange,tile
            symmetry = arange(gridsize*gridsize).reshape(gridsize,gridsize)
            symmetry = tile(symmetry,(M/gridsize,N/gridsize))
            symmetry = symmetry.flatten().tolist()
            for i in range(gridsize*gridsize):
                symmetry[symmetry.index(i)] = None

            potential = (-1)**arange(gridsize)
            potential = lambda_x*potential[None,:].repeat(gridsize,axis=0)
            potential = potential.flatten()
        elif row_sym.startswith('rect'):
            # This is a gridded symmetry useful for periodic boundaries but not of 
            # equal size in the x and y directions
            xsize,ysize = [int(z) for z in row_sym[4:].split('x')]

            assert N%xsize == 0 and M%ysize == 0
            assert lambda_x == 0 or N%2 == 0

            from numpy import arange,tile
            symmetry = arange(xsize*ysize).reshape(ysize,xsize)
            symmetry = tile(symmetry,(M/ysize,N/xsize))
            symmetry = symmetry.flatten().tolist()
            for i in range(xsize*ysize):
                symmetry[symmetry.index(i)] = None

            potential = (-1)**arange(xsize)
            potential = lambda_x*potential[None,:].repeat(ysize,axis=0)
            potential = potential.flatten()
    elif row_sym is not None:
        symmetry = [None] * size
        for i in xrange(size):
            if i/(N*row_sym) > 0:
                symmetry[i] = i%(N*row_sym)
        #potential = zeros(N*row_sym)
        # Add staggering in
        potential = (-1)**arange(N)
        potential = lambda_x*potential[None,:].repeat(row_sym,axis=0)
        potential = potential.flatten()
    else:
        symmetry = [None] * size
        #for i in xrange(size):
        #    if i/N > 0:
        #        symmetry[i] = i%N
        from numpy import tile,arange
        potential = lambda_x * (-1)**arange(N)
        potential = tile(potential,(M,1))
        potential = potential.flatten()

    print(symmetry)
    return symmetry,potential

def TI_FourierCTAUXRealFreq(redo_peaks=False,redo_all=False,min_weight=None,max_weight=None,use_U0=False,y_only=True,no_transform=False,show_figures=True,only_omega0=False,complete_mat=False,r_deriv='N',no_invert=False,broadening=None,omega=None,interpolate_omega=True):
    r'''This function provides :func:`FourierTransformSpin` with the appropriate parameters to 
    calculate various Fourier transformed Green's function matrices of real frequencies.

    The self-energy is taken from the continued self-energy in ``SE_real.npz`` and interpolated onto 
    a grid of frequencies, chosen to be finer around omega=0, so as to generate smoother figures. If 
    ``interpolate_omega==False`` then the frequencies are taken from that provided in ``omega``.
    If ``omega==None`` then the grid from the analytical continuation is used instead.

    Some broadening is required in real space, and this is calculated from the number of sites in 
    the y-direction. This is so that an edge mode will be smoothly continued as it crosses a bulk 
    gap, rather than consisting of individual peaks. It is possible to override this by 
    provided a value in ``broadening``.

    The parameters ``y_only,no_transform,complete_mat,no_invert,only_omega0`` are described in 
    :func:`FourierTransformSpin`. However, their values determine the filename of which to save.
    The filename is::

        <base><comp><onlyomega0><noinvert>_real.npz

    see :func:`TI_FourierCTAUX` for more information.

    The option ``use_U0`` exists to allow a zero self-energy to be chosen. However, it is still required 
    that a ``info.dat`` file already exists. See :func:`GenerateAllDataForU0`.

    This functions also determines the peaks using :func:`SavePeaksTI`. The arguments ``redo_peaks``, 
    ``redo_all``, ``min_weight``, ``max_weight``.
    '''

    info = ReadInfo()
    U = info['U']
    mu = info['mu']
    boundaries = info['boundaries']
    t = info['t']
    N = info['N']
    M = info['M']
    p = info['p']
    q = info['q']
    row_sym = info['row_sym']
    lambda_x = info['lambda_x']
    gamma = info['gamma']
    final_mu = info['final_mu']

    if use_U0:
        # Force half-filling
        U = 0.
        mu = 0.

    if boundaries[0] == 'P':
        assert (N % q) == 0

    if broadening in [None,0.]:
        from numpy import sqrt,round
        broadening = 4./M * sqrt(2)
        broadening /= 2.
        broadening = round(broadening,3)

    # Have to use the mu here that may have been changed by the force_filling 
    # search
    U *= t
    final_mu *= t

    import os
    if no_transform:
        filename = 'Gx'
    elif y_only:
        filename = 'Gk'
    else:
        filename = 'Gkk'
    if complete_mat:
        filename += '_comp'
    if only_omega0:
        filename += '_o0'
    if r_deriv != 'N':
        filename += '_D' + r_deriv
    if no_invert:
        filename += '_inv'
    filename += '_real'
    filename += '.npz'
    if not os.path.exists(filename) or redo_all:
        symmetry,potential = TI_Symmetry((N,M),row_sym,lambda_x)

        if use_U0:
            SE = (None,None,None)

            if omega == None:
                from numpy import linspace,array
                omega = linspace(-5,5,200)
        elif U == 0.:
            logger.info("Using a zero self-energy because of U=0")
            SE = (None,None,None)

            if omega == None:
                from numpy import linspace,array
                omega = linspace(-5,5,200)
        else:
            import numpy
            file = numpy.load('SE_real.npz')
            omega = file['omega']
            SE = file['SE']
            SE = SE.transpose([0,2,1])

        if interpolate_omega and not use_U0:
            # Interpolate the input to a set number of points, so that we can 
            # achieve the resolution to identify edge states.
            orig_omega = omega
            orig_SE = SE

            from numpy import empty,arange,r_,linspace
            if interpolate_omega == True:
                omega = r_[linspace(-10,-4,21)[:-1],linspace(-4,-1,51)[:-1],linspace(-1,1,301)[:-1],linspace(1,4,51)[:-1],linspace(4,10,21)]

            SE = empty((3,len(omega),orig_SE.shape[2]),complex)

            from scipy.interpolate import UnivariateSpline
            for index in range(3):
                for index2 in range(orig_SE.shape[2]):
                    interp_real = UnivariateSpline(orig_omega,orig_SE[index,:,index2].real,s=0)
                    interp_imag = UnivariateSpline(orig_omega,orig_SE[index,:,index2].imag,s=0)
                    SE[index,:,index2] = interp_real(omega) + 1j*interp_imag(omega)

        # Need to put some broadening here since we are in real-space
        omega = omega + 1j*broadening

        FourierTransformSpin(U,final_mu,gamma,SE,potential,omega,(N,M),boundaries,t,(p,q),symmetry=symmetry,y_only=y_only,no_transform=no_transform,only_omega0=only_omega0,complete_mat=complete_mat,no_invert=no_invert,output_filename=filename,matsubara=False)

    if not os.path.exists('peaksup.pickle.gz') or redo_peaks or redo_all:
        if no_transform == False and only_omega0 is False and y_only:
            SavePeaksTI('up')
            print("Done SavePeaks up")
            SavePeaksTI('down')
            print("Done SavePeaks down")

    if show_figures and not no_transform and only_omega0 == False and y_only:
        # Plot them
        from pylab import figure,title,xlabel,ylabel
        figure()
        PlotPeaksWithWeights(min_weight=min_weight,max_weight=max_weight,spin='up')
        title(r'$\alpha={p}/{q}$, $\Lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**locals()))
        xlabel('$k_y$')
        ylabel('Energy')
        #figure()
        #PlotPeaksWithWeights(min_weight=min_weight,max_weight=max_weight,spin='down')

def TI_FourierCTAUX(use_U0=False,y_only=True,no_transform=False,complete_mat=False,no_invert=False,use_additional=None,only_omega0=False,num_average=5):
    r'''This function provides :func:`FourierTransformSpin` with the appropriate parameters to 
    calculate various Fourier transformed Green's function matrices of Matsubara frequencies.

    The self-energy is given using :func:`AverageData` or using ``additional.npz`` if ``use_additional==True``.

    The parameters ``y_only,no_transform,complete_mat,no_invert,only_omega0`` are described in 
    :func:`FourierTransformSpin`. However, their values determine the filename of which to save.
    The filename is::

        <base><comp><onlyomega0><noinvert>.npz

    where ``<base>`` is 'Gx' when ``no_transform == True``, 'Gk' if ``y_only == True`` or 'Gkk' otherwise. 
    ``<comp>,<onlyomega0>,<noinvert>`` are empty if ``complete_mat,only_omega0,no_invert`` are False and 
    appropriately named otherwise. For example, a filename::

        Gk_comp_o0.npz

    indicates the quantity G(x,x',ky,ky') at omega~0 has been saved.

    The option ``use_U0`` exists to allow a zero self-energy to be chosen. However, it is still required 
    that a ``info.dat`` file already exists. See :func:`GenerateAllDataForU0`.
    '''
    import ImagConv
    import os

    info = ReadInfo()
    U = info['U']
    mu = info['mu']
    beta = info['beta']
    boundaries = info['boundaries']
    t = info['t']
    N = info['N']
    M = info['M']
    p = info['p']
    q = info['q']
    gamma = info['gamma']
    final_mu = info['final_mu']

    if use_U0:
        # Force half-filling
        U = 0.
        mu = 0.

    if boundaries[0] == 'P':
        assert (N % q) == 0

    # Have to use the mu here that may have been changed by the force_filling 
    # search
    # Also put in a -U/2 because we are always using the CT-AUX solver here.
    U = U*t
    final_mu = final_mu*t-U/2.*t

    symmetry,potential = TI_Symmetry((N,M),info['row_sym'],info['lambda_x'])

    import os
    if no_transform:
        filename = 'Gx'
    elif y_only:
        filename = 'Gk'
    else:
        filename = 'Gkk'
    if complete_mat:
        filename += '_comp'
    if only_omega0:
        filename += '_o0'
    #if r_deriv != 'N':
    #    filename += '_D' + r_deriv
    if no_invert:
        filename += '_inv'
    filename += '.npz'
    if use_U0:
        SE = (None,None,None)
    elif U == 0:
        SE = (None,None,None)
        logger.info("Using empty SE because U=0")
    else:
        with FileLock():
            if use_additional == None:
                if os.path.exists('additional.npz'):
                    use_additional = True
                else:
                    use_additional = False
            if use_additional:
                from numpy import load
                data = load('additional.npz')
                SE = data['SE']
            else:
                G,SE = AverageData(-num_average,-1)

    from numpy import r_
    omega = ImagConv.OmegaN(info['num_omega'],beta)
    omega = r_[-omega[::-1],omega]
    omega = 1j*omega

    if not os.path.exists(filename):
        FourierTransformSpin(U,final_mu,gamma,SE,potential,omega,(N,M),boundaries,t,(p,q),symmetry=symmetry,y_only=y_only,no_transform=no_transform,complete_mat=complete_mat,no_invert=no_invert,output_filename=filename,only_omega0=only_omega0,matsubara=True)

def TI_ContinueCTAUX_SE(use_U0=False,num_omegaA=400,covar_factor=1.,use_additional=None,num_average=5,**keys):
    '''Continue all of the calculated Matsubara frequency self-energies into 
    real-frequency spectra. These will then be saved into ``SE_real.npz``. If 
    that file already exists, then this function is not run. Additional information is 
    stored in the text file ``SE_real_info.dat`` and some plots with details are 
    saved to ``SE_real_imag.png`` and ``SE_real_real.png``.

    ``covar_factor`` is the artificial variance to give to the self-energy data. ``num_omegaA`` 
    is the number of grid points in the real frequency grid to use.

    The -U/2 factor that is added in the CT-AUX solver is adjusted at this point.

    The continuation itself is performed in :func:`ContinueSelfEnergy`, and this function 
    parallelizes the execution over the sites.

    Any additional information for the MaxEnt can be passed via ``**keys``.
    
    An example ``SE_real_info.dat`` file is given below:

    .. include:: /example_figures/RDMFT_TISM/SE_real_info.dat
        :literal:
    
    '''
    #matplotlib.use('Agg')

    from numpy import zeros,savez,empty,identity,linspace,load
    info = ReadInfo()

    U = info['U']
    mu = info['mu']
    beta = info['beta']
    N = info['N']
    M = info['M']
    t = info['t']

    if use_U0:
        # Force half-filling
        U = 0.
        mu = 0.

    mu = mu - U/2.
    size = N*M

    import os
    if not os.path.exists('SE_real.npz'):

        if use_additional == None:
            if os.path.exists('additional.npz'):
                use_additional = True
            else:
                use_additional = False

        symmetry,potential = TI_Symmetry((N,M),info['row_sym'],info['lambda_x'])
        num_sites = len(potential)

        omegaA = linspace(-10,10,num_omegaA)

        if use_U0:
            SE = zeros((3,num_sites,num_omegaA))
            savez('SE_real.npz',omega=omegaA,SE=SE)
            return
        elif use_additional:
            from numpy import load
            data = load('additional.npz')
            SE_iw = data['SE']
        else:
            with FileLock():
                G,SE_iw = AverageData(-num_average,-1)

        SE_w = empty((3,num_sites,num_omegaA),complex)
        import ImagConv
        num_T = info['num_omega']

        from dancommon import CreatePool
        num,pool = CreatePool()
        res = {}

        for func in [0,1,2]:
            for site in range(num_sites):
                res[func,site] = pool.apply_async(ContinueSelfEnergy,(SE_iw[func][:,site],omegaA,num_T,beta,covar_factor),keys)

        for func in [0,1,2]:
            for site in range(num_sites):
                #omegaA,SE_w[func,site],alpha = res[func,site].get()
                SE_w[func,site] = res[func,site].get()
                # Correct for the mu' = mu - U/2 in the CT solver.
                # This could also be done before the continuation, but it 
                # doesn't make a difference.
                if func == 0 or func == 1:
                    SE_w[func,site] += U/2.
                logger.info("Done site {site}, func {func}".format(**locals()))

        pool.close()
        pool.join()

        savez('SE_real.npz',omega=omegaA,SE=SE_w)
        # Also plot these for info
        import pylab as pl
        pl.figure()
        # The legend handling here requires assigning the legend to only one of 
        # each set of lines
        up = pl.plot(SE_w[0].T.imag,'b',label='up')
        down = pl.plot(SE_w[1].T.imag,'g',label='down')
        cross = pl.plot(SE_w[2].T.imag,'r',label='cross')
        pl.legend([up[0],down[0],cross[0]],['up','down','cross'])
        pl.savefig('SE_real_imag.png')
        pl.close()
        pl.figure()
        up = pl.plot(SE_w[0].T.real,'b',label='up')
        down = pl.plot(SE_w[1].T.real,'g',label='down')
        cross = pl.plot(SE_w[2].T.real,'r',label='cross')
        pl.legend([up[0],down[0],cross[0]],['up','down','cross'])
        pl.savefig('SE_real_real.png')
        pl.close()

        with open('SE_real_info.dat','wb') as file:
            omegaA_min = min(omegaA)
            omegaA_max = max(omegaA)
            file.write(
'''Info file for continued self-energy
num_T {num_T}
num_omegaA {num_omegaA}
omegaA_limits {omegaA_min} {omegaA_max}
covar_factor {covar_factor}
beta {beta}
-------------------------------
Additional keys for MaxEnt:
'''.format(**locals()))
            for key in keys:
                file.write("{0} {1!r}\n".format(key,keys[key]))

            file.write(
'''
-------------------------------
Info dict read from info.dat:
'''.format(**locals()))

            for key in info:
                file.write("{0} {1!r}\n".format(key,info[key]))

def ContinueSelfEnergy(SE_iw,real_omega,num_T,beta,covar_factor,bias_covar=False,show_figures=False,**keys):
    r'''This function takes a single self-energy ``SE_iw`` in Matsubara 
    frequency and then analytically continues it to real frequencies, on the 
    grid specified in ``real_omega``.

    Because the self-energy is not sampled in the CT solver, it is necessary 
    to impose an artificial variance in order to perform the continuation. This 
    is ``covar_factor``, and should be varied to determine the best fit to the 
    data. Values that are too small will generate spikey spectra, whereas values 
    that are too large will lose information and blur out the spectra. I have 
    been working with values in the range ``0.01 < covar_factor < 1.0``.
    
    One can also specify ``bias_covar==True``, in which case the covariance will be 
    biased towards small values of ``T``. Use this only for testing purposes.

    To perform the continuation, the input is Fourier transformed into imaginary time, of
    which a grid of ``num_T`` may be specified (it is recommended that this is set to 
    the same size as ``SE_iw`` although smaller values can speed up the continuation). 
    The inverse temperature ``beta`` is also required here.

    **Notes:**
        
        * If the input is zero, the continuation will not be performed and zero 
          will be returned.
        * As the model functions in the continuation are normalised to unity, the 
          input is first renormalised, and after continuation the result is again corrected. 
          This normalisation comes from :math:`G(\tau = 0^+) - G(\tau = 0^-)`.
        * If the normalisation of the input is found to be less than 10^-8, it 
          is assumed to be zero.
        * The function should be passed with both positive and negative frequency 
          components. I.e. ``SE_iw[0]`` is :math:`G(-i \omega_N)` where N is the largest 
          frequency.
        * The :math:`1/\omega` term is factored out of the input with a polynomial fit in 
          inverse powers of :math:`1/\omega` to the last 200 points. This is necessary 
          for a smooth Fourier transformation.
        * From the polynomial fit, the constant mean-field term is also extracted and 
          then added later onto the result, as this is arbitrary in the Kramers-Kronig 
          relations.
        * The final 'spectra' of the continuation is transformed into the full real-space 
          self-energy with the Kramers-Kronig relations

    The actual continuation is done by the function :func:`Continuation.ClassicalMaxEnt`. 
    To pass any additional arguments to this function, supply them in ``**keys``.

    **Returns:**
    
        ``full_SE`` (complex double array)
    '''
    from numpy import vectorize,exp,logspace,sin,where,identity,polyfit,polyval,arange,diag,r_
    import ImagConv
    import Continuation
    import pylab

    # Special case a zero function
    if all(SE_iw == 0):
        return 0*real_omega

    matsu_omega = ImagConv.OmegaN(len(SE_iw)/2,beta)
    matsu_omega = r_[-matsu_omega[::-1], matsu_omega]

    # Numerically work out leading order C/1j*omega term for transformation.
    x = 1./matsu_omega
    # TODO: This should be set in a nicer manner.
    # Note that there is no need to account for the larger length of y, as we 
    # only take the last few points.
    num_fit_points = 200
    x = x[::-1][:num_fit_points]
    y = SE_iw[::-1][:num_fit_points]
    #coeffs_imag = polyfit(x**2,y.imag/x,2)
    coeffs_imag = polyfit(x,y.imag,2)
    coeffs_real = polyfit(x**2,y.real,2)

    logger.info("Real asymptote is {0}".format(coeffs_real[2]))
    SE_iw -= coeffs_real[2]

    logger.info("Leading order coeff is {0}".format(-coeffs_imag[1]))
    logger.info("Constant term of imaginary part is {0}".format(coeffs_imag[0]))

    if show_figures:
        pylab.figure()
        pylab.title('SE match with coeffs')
        pylab.plot(matsu_omega,SE_iw.imag,'g')
        #pylab.plot(matsu_omega,polyval(coeffs_imag,matsu_omega**2)*matsu_omega)
        #pylab.plot(matsu_omega,coeffs_imag[-1]*1./matsu_omega,'b:')
        #pylab.plot(matsu_omega,coeffs_imag[-1]*1./matsu_omega + -coeffs_imag[-2]*1./(matsu_omega**3),'r')
        #pylab.plot(matsu_omega,coeffs_imag[-1]*1./matsu_omega + coeffs_imag[-2]*1./(matsu_omega**3),'r:')
        pylab.plot(matsu_omega,polyval(coeffs_imag,1./matsu_omega),'b')
        pylab.plot(matsu_omega,coeffs_imag[-1] + 0*matsu_omega,'b:')
        #pylab.plot(matsu_omega,coeffs_imag[-1]*1.+ -coeffs_imag[-2]*1./(matsu_omega),'r')
        pylab.plot(matsu_omega,coeffs_imag[-1]*1.+ coeffs_imag[-2]*1./(matsu_omega),'r:')
        pylab.ylim(-1e0,1e0)
        pylab.xlim(0,pylab.xlim()[1])
        pylab.savefig('Continue_firstplot.png')

        pylab.figure()
        pylab.plot(x,y.imag)
        pylab.plot(x,polyval(coeffs_imag,x))
        pylab.savefig('Continue_asdf.png')

    T = ImagConv.GetT(num_T,beta)
    SE_T = ImagConv.FreqToImagTimeComplex(matsu_omega,SE_iw,T,beta,leading_order_coeff=-coeffs_imag[-1])
    #SE_T[SE_T<1e-8] = -SE_T[SE_T<1e-8]

    if show_figures:
        pylab.figure()
        pylab.title('T transform with leading order comparison')
        pylab.plot(T,SE_T.real,'b')
        orig_T = ImagConv.FreqToImagTime(matsu_omega,SE_iw,T,beta)
        pylab.plot(T,orig_T.real,'g:')
        no_T = ImagConv.FreqToImagTime(matsu_omega,SE_iw,T,beta,leading_order_coeff=0.)
        pylab.plot(T,no_T.real,'r')
        pylab.savefig('Continue_secondplot.png')

        pylab.figure()
        pylab.title('T transform with leading order comparison (imag)')
        pylab.plot(T,SE_T.imag,'b')
        orig_T = ImagConv.FreqToImagTime(matsu_omega,SE_iw,T,beta)
        pylab.plot(T,orig_T.imag,'g:')
        no_T = ImagConv.FreqToImagTime(matsu_omega,SE_iw,T,beta,leading_order_coeff=0.)
        pylab.plot(T,no_T.imag,'r')
        pylab.savefig('Continue_secondplot_imag.png')

    norm = vectorize(lambda eta: 1/beta * (sum(exp(1j*matsu_omega*eta) * SE_iw - exp(-1j*matsu_omega*eta) * SE_iw).real*2))

    eta = logspace(-5,0,1000)
    y = norm(eta)
    #pylab.figure()
    #pylab.plot(eta,y)
    # Look for the max, and then the closest local-min to the right
    maxpoint = argmax(y)
    if (y[maxpoint:-1] > y[maxpoint+1:]).all():
        nextminpoint = maxpoint
    else:
        nextminpoint = maxpoint + where((y[maxpoint+1:] - y[maxpoint:-1]) > 0)[0][0]

    norm_guess = (y[maxpoint]+y[nextminpoint])/2.
    logger.info("Norm guess is {0}".format(norm_guess))

    if norm_guess < 1e-8:
        logger.info("Norm is too small, assuming result is zero")
        return 0*real_omega

    if bias_covar:
        covar = diag(arange(1,len(T)+1)**(1/2.)) * covar_factor
    else:
        covar = identity(len(T))*covar_factor

    real_omega,A,alpha = Continuation.ClassicalMaxEnt(T,-SE_T/norm_guess,covar,beta,real_omega,**keys)
    from numpy import trapz
    logger.info("Norm is {0}".format(trapz(A,real_omega)))
    A *= norm_guess

    # Force A to be real here... this is needed for the KramersKronig 
    # relations, but may actually be bad! FIXME
    A = A.real

    full_SE = KramersKronig(real_omega,A)
    full_SE += coeffs_real[2]

    return full_SE

def TI_ContinueCTAUX_Gk(num_omegaA=400,covar_factor=1.,**keys):
    '''Continue the Fourier transformed Matsubara Green's functions to 
    real-frequency.  The ``covar_factor`` is the artificial variance to give to the data.

    Any additional information for the MaxEnt can be passed via ``**keys``.
    
    After testing, it was found that the continuation from this form does not 
    work as well as continuing the self-energies directly. This is because there 
    are far fewer self-energies to continue and the results are therefore more consistent.
    
    This function is not suitable for casual use. All details are hard-coded and 
    may not function as expected. Please use :func:`TI_ContinueCTAUX_SE` instead.
    '''

    from numpy import zeros,savez,empty,identity,linspace,load,r_
    info = ReadInfo()

    U = info['U']
    mu = info['mu']
    beta = info['beta']
    N = info['N']
    M = info['M']
    t = info['t']

    mu = mu - U/2.
    size = N*M

    TI_FourierCTAUX(y_only=True)


    import os
    if not os.path.exists('Gk_real_directfft.npz'):
        omegaA = linspace(-20,20,num_omegaA)

        Gk_w = empty((3,N*M,num_omegaA),complex)
        import ImagConv
        num_T = info['num_omega']
        T = ImagConv.GetT(info['num_omega'],beta)
        omega = ImagConv.OmegaN(info['num_omega'],beta)
        omega = r_[-omega[::-1],omega]

        from dancommon import CreatePool
        num,pool = CreatePool()
        res = {}

        #for func in [0,1,2]:
        for func in [2,1,0]:
            dummy,Gk_iw = ReadGkData(spin=func)

            Gk_iw = Gk_iw.transpose(2,1,0).reshape(info['num_omega']*2,N*M)
            for site in range(N*M):
                if func == 2:
                    #Gk_T = ImagConv.FreqToImagTimeComplex(omega,Gk_iw[:,site],T,beta,leading_order_coeff=0.)
                    #import pylab as pl
                    #pl.figure()
                    #pl.plot(T,Gk_T)
                    #pl.plot(T,Gk_T)
                    #pl.plot(omega,Gk_iw[:,site].real)
                    #pl.plot(omega,Gk_iw[:,site].imag)
                    #raise SystemExit
                    res[func,site] = pool.apply_async(ContinueSelfEnergy,(Gk_iw[:,site],omegaA,num_T,beta,covar_factor),keys)
                else:
                    Gk_T = ImagConv.FreqToImagTimeComplex(omega,Gk_iw[:,site],T,beta)
                    import Continuation
                    res[func,site] = pool.apply_async(Continuation.ClassicalMaxEnt,(T,-Gk_T,identity(len(T))*covar_factor,beta,omegaA),keys)

        #for func in [0,1,2]:
        for func in [2,1,0]:
            for site in range(N*M):
                #omegaA,SE_w[func,site],alpha = res[func,site].get()
                if func == 2:
                    Gk_w[func,site] = res[func,site].get()
                else:
                    real_omega,A,alpha = res[func,site].get()
                    Gk_w[func,site] = KramersKronig(omegaA,A)
                # Correct for the mu' = mu - U/2 in the CT solver.
                Gk_w[func,site] += U/2.
                logger.info("Done site {site}, func {func}".format(**locals()))

        pool.close()
        pool.join()

        Gk_w = Gk_w.reshape(3,M,N,num_omegaA).transpose(0,2,1,3)

        savez('Gk_real_directfft.npz',omega=omegaA,greens=Gk_w)
        # Also plot these for info
        import pylab as pl
        pl.figure()
        # The legend handling here requires assigning the legend to only one of 
        # each set of lines
        up = pl.plot(Gk_w[0].reshape(N*M,-1).T.imag,'b',label='up')
        down = pl.plot(Gk_w[1].reshape(N*M,-1).T.imag,'g',label='down')
        cross = pl.plot(Gk_w[2].reshape(N*M,-1).T.imag,'r',label='cross')
        pl.legend([up[0],down[0],cross[0]],['up','down','cross'])
        pl.savefig('Gk_real_directfft_imag.png')
        pl.close()

        with open('Gk_real_directfft_info.dat','wb') as file:
            omegaA_min = min(omegaA)
            omegaA_max = max(omegaA)
            file.write(
'''Info file for continued Gk
num_T {num_T}
num_omegaA {num_omegaA}
omegaA_limits {omegaA_min} {omegaA_max}
covar_factor {covar_factor}
beta {beta}
-------------------------------
Additional keys for MaxEnt:
'''.format(**locals()))
            for key in keys:
                file.write("{0} {1!r}\n".format(key,keys[key]))

            file.write(
'''
-------------------------------
Info dict read from info.dat:
'''.format(**locals()))

            for key in info:
                file.write("{0} {1!r}\n".format(key,info[key]))



def ReadInfo():
    '''This is a convenience function to access common information about the output 
    of a job. It reads the ``info.dat`` file and performs some simple calculations 
    that can then be easily accessed in a dictionary.

    The information returned currently includes the system parameters:

        ``N``,``M``,``U``,``mu``,``num_sweeps``,``num_omega``,``row_sym``,``p``,``q``,``beta``,``t``,``lambda_x``,``force_filling``,``gamma``,

    as well as the following:

        ``num_iters``
            The number of iterations performed. Note that this is not the requested number of iterations.
        ``final_mu``
            The final chemical potential at the last iteration. If ``force_filling==None`` then ``final_mu==mu``.
        ``Nup_diff``
            The largest difference of a single site's up-spin filling from the last iteration. See :func:`ConvergenceSpin`.

    **Returns:**
        ``info`` (dictionary)
    '''

    with open('info.dat') as file:
        front,version = file.readline().split()
        version = int(version)
        assert front == 'Version'
        front,solver = file.readline().split()
        assert front == 'Solver'
        #assert solver == 'CT-AUX'
        front,N,M = file.readline().split()
        assert front == 'N,M'
        front,boundaries = file.readline().split()
        assert front == 'Boundaries'
        front,U = file.readline().split()
        assert front == 'U'
        front,mu = file.readline().split()
        assert front == 'mu'
        front,num_sweeps = file.readline().split()
        assert front == 'num_sweeps'
        front,num_omega = file.readline().split()
        assert front == 'num_omega'
        if version <= 2:
            front,AF_sym = file.readline().split()
            assert front == 'AF_sym'
        front,row_sym = file.readline().split()
        assert front == 'row_sym'
        front,p,q = file.readline().split()
        assert front == 'p,q'
        front,beta = file.readline().split()
        assert front == 'beta'
        front,t = file.readline().split()
        assert front == 't'
        if version >= 3:
            front,lambda_x = file.readline().split()
            if version < 6:
                assert front == 'Lambda'
            else:
                assert front == 'lambda_x'
        else:
            lambda_x = '0.0'
        if version >= 6:
            front,force_filling = file.readline().split()
            assert front == 'Force_filling'
        else:
            force_filling = 'None'
        front,gamma = file.readline().split()
        assert front == 'gamma'
        if version >= 7:
            front,fit_large_SE = file.readline().split()
            assert front == 'fit_large_SE'
        else:
            fit_large_SE = False

        assert file.readline().strip() == '-----'

        N = int(N)
        M = int(M)
        U = float(U)
        mu = float(mu)
        num_sweeps = float(num_sweeps)
        num_omega = int(num_omega)
        if version <= 2 and AF_sym == 'True':
            row_sym = 'AF'
        elif row_sym == 'None':
            row_sym = None
        elif not row_sym.startswith('AF') and not row_sym.startswith('grid') and not row_sym.startswith('rect'):
            row_sym = int(row_sym)
        p = int(p)
        q = int(q)
        beta = float(beta)
        t = float(t)
        lambda_x = float(lambda_x)
        force_filling = None if force_filling == 'None' else float(force_filling)
        gamma = float(gamma)

        # Work out last iteration from info file
        with open('info.dat') as file:
            last_line = file.readlines()[-1].strip()
        try:
            first_item = last_line.split()[0]
            last_item = last_line.split()[-1]
            Nup_diff = last_line.split()[3]

            num_iters = int(first_item)
            final_mu = float(last_item)
            Nup_diff = float(Nup_diff)
        except (ValueError,IndexError):
            logger.info("No valid iterations found!")
            num_iters = None
            final_mu = mu
            Nup_diff = None
        
        info = {}
        for key in ['solver','N','M','boundaries','U','mu','num_sweeps','num_omega','row_sym','p','q','beta','t','lambda_x','force_filling','gamma','num_iters','final_mu','Nup_diff']:
            info[key] = locals()[key]

        return info

def ReadGkData(spin='up',xdata=False,kkdata=False,complete_mat=False,r_deriv='N',no_invert=False,real_freq=False,directfft=False):
    '''This is a convenience function to read the data for spin ``spin`` from the appropriate file of the form::
    
        <base><comp><onlyomega0><noinvert>.npz

    See :func:`TI_FourierCTAUX` for details about the file name and arguments when ``real_freq==False`` and 
    see :func:`TI_FourierCTAUXRealFreq` for details when ``real_freq==True``.

    ``spin`` can be either one of 'up','down','cross' or 0,1,2 respectively.

    The argument ``directfft`` is designed for continuation of the Green's function directly (see :func:`TI_ContinueCTAUX_Gk`). It should not be used.

    **Returns:** ``omega,trans``

        ``omega`` (complex double array)
            The frequencies (Matsubara or real) of the Green's function.
        ``trans`` (complex double array)
            The Green's function itself as an array, which will be many-dimensional.
    '''
    assert spin in ['up','down','cross',0,1,2]
    assert not xdata or not kkdata

    if xdata:
        filename = 'Gx'
    elif kkdata:
        filename = 'Gkk'
    else:
        filename = 'Gk'
    if complete_mat:
        filename += '_comp'
    if r_deriv != 'N':
        filename += 'D' + r_deriv
    if no_invert:
        filename += '_inv'
    if real_freq:
        filename += '_real'
    if directfft:
        filename += '_directfft'
    filename += '.npz'
    print(filename)

    import os
    if not os.path.exists(filename):
        omega = trans = None
    else:
        from numpy import load
        data = load(filename)
        omega = data['omega']
        if spin == 'up' or spin == 0:
            trans = data['greens'][0]
        elif spin == 'down' or spin == 1:
            trans = data['greens'][1]
        elif spin == 'cross' or spin == 2:
            trans = data['greens'][2]

    return omega,trans

def PlotGkData(spin,x=None,map=False,directfft=False):
    r'''Plot some basic graphs from the G(x,y) Green's function data saved with :func:`TI_FourierCTAUXRealFreq`.

    The ``spin`` parameter is passed directly to :func:`ReadGkData`.

    If ``map==True`` then ``x`` is ignored and a colormap of the spectra, integrated over ``x`` is 
    plotted. An example is shown below:

    .. image:: example_figures/RDMFT_TISM/spec_dens.png

    If ``map==False`` then :math:`Im G(x)` is plotted for either a specific x==``x`` or for all 
    values of x when ``x==None``. An example that results from the following code::

        PlotGkData('up',x=0)
        PlotGkData('up',x=int(info['N']/2))
        PlotGkData('up',x=int(info['N']/2 + 1))

    is shown below:

    .. image:: example_figures/RDMFT_TISM/edge_mid_comp.png

    Note that these figures are not saved. To save these figures, after calling this function, 
    one can use the follow code (as an example)::

        savefig('spec_dens_ky.png')

    '''
    info = ReadInfo()
    omega,trans = ReadGkData(spin,xdata=True,real_freq=True,directfft=directfft)

    from pylab import plot,pcolor,arange,xlim,colorbar,title,rcParams
    rcParams['text.usetex'] = True
    from matplotlib import cm
    if map:
        avg_trans = -trans.sum(axis=1).imag / trans.shape[1]
        avg_trans[avg_trans<0] = 1e-16
        from numpy import log10,meshgrid
        #avg_trans = log10(avg_trans)
        #avg_trans[avg_trans<0] = 0.
        #avg_trans = log(avg_trans)**(2)
        #avg_trans /= avg_trans.max()
        from pylab import cm
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),
                 'green' : ((0.0, 0.0, 0.0),
                            (0.5, 1.0, 1.0),
                            (1.0, 1.0, 1.0)),
                 'blue' : ((0.0, 0.0, 0.0),
                           (0.5, 1.0, 1.0),
                           (1.0, 0.4, 0.4))}
        from matplotlib.colors import LinearSegmentedColormap
        colormap = LinearSegmentedColormap('danny',cdict,256)
        #pcolor(arange(avg_trans.shape[0]+1),omega,avg_trans.T,cmap=cm.BuGn)
        #pcolor(arange(avg_trans.shape[0]+1),omega,avg_trans.T,cmap=colormap)
        pcolor(arange(avg_trans.shape[0]+1),omega.real,avg_trans.T)
        xlim(0,avg_trans.shape[0])
        #pcolor(-avg_trans.sum(axis=1).T.imag,cmap=cm.BuGn)
        colorbar()
        title(r'$\alpha={p}/{q}$, $\Lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
	savefig('spec_dens_ky.png')
    else:
        avg_trans = -trans[:,:trans.shape[1]/2].sum(axis=1).imag / trans.shape[1]
        if x == None:
            plot(omega.real,avg_trans.T)
        else:
            plot(omega.real,avg_trans[x])

def PlotPeaksMap(spin='up',x=None,directfft=False):
    r'''Plot some graphs from the G(x,k_y) Green's function data saved with :func:`TI_FourierCTAUXRealFreq`.

    The ``spin`` parameter is passed directly to :func:`ReadGkData`.

    The graphs are a color map of the spectra of the Green's functions integrated over x: 
    :math:`\int G(x,k_y) dx`. Depending on the value of ``x``, different regions are 
    integrated over with respect to x.

        * ``x==None`` Integrate the entire region.
        * ``x=='mid'`` Integrate the middle third of the lattice.
        * ``x=='lefthalf'`` Integrate the left half of the lattice.
        * ``x==<int>`` Do not integrate, but plot only :math:`G(x=x,k_y)`.

    An example of ``x=='lefthalf'`` is shown below:

    .. figure:: example_figures/RDMFT_TISM/spec_dens_ky_lefthalf.png

    Note that the name of this function is misleading and does not 
    actually have anything to do with the Peaks functions.
    '''

    info = ReadInfo()
    omega,trans = ReadGkData(spin,real_freq=True,directfft=directfft)

    from pylab import plot,pcolor,arange,xlim,colorbar,title,ylim,imshow
    from matplotlib import cm
    from numpy import log10,meshgrid,pi,linspace

    if x == None:
        avg_trans = -trans.sum(axis=0).imag / trans.shape[0] / pi
    elif x == 'mid':
        # Sum over the middle third
        third = int(trans.shape[0] / 3)
        avg_trans = -trans[third:third*2].sum(axis=0).imag / third / pi
    elif x == 'lefthalf':
        half = int(trans.shape[0] / 2)
        avg_trans = -trans[:half].sum(axis=0).imag / trans.shape[0] / pi
    else:
        avg_trans = -trans[x].imag / pi
    avg_trans[avg_trans<0] = 1e-16

    pcolor(arange(avg_trans.shape[0]+1),omega.real,avg_trans.T)
    xlim(0,avg_trans.shape[0])

    #xspaced,yspaced = meshgrid(linspace(0,avg_trans.shape[0],100), linspace(min(omega),max(omega),100))
    #xspaced = linspace(0,avg_trans.shape[0],1000)
    #yspaced = linspace(min(omega).real,max(omega).real,1000)
    #xorig,yorig = meshgrid(arange(avg_trans.shape[0]),omega.real)
    #from matplotlib.mlab import griddata
    #spaced_data = griddata(xorig.flatten(),yorig.flatten(),avg_trans.T.flatten(),xspaced,yspaced,'linear')
    #from scipy.interpolate import interp2d
    #spaced_data = interp2d(xorig,yorig,avg_trans.T)(xspaced,yspaced)
    #imshow(spaced_data,extent=(-2*pi,2*pi,min(omega),max(omega)),aspect='auto')
    #imshow(avg_trans.T,extent=(-2*pi,2*pi,min(omega),max(omega)),aspect='auto',interpolation='nearest')
    #imshow(avg_trans.T,aspect='auto',interpolation='nearest')

    ylim(-5,5)
    colorbar()

    if x == None:
        title(r'$\alpha={p}/{q}$, $\Lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    elif x == 'mid':
        title(r'Bulk $\alpha={p}/{q}$, $\Lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    else:
        title(r'x={x} $\alpha={p}/{q}$, $\Lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(x=x,**info))

    import pylab
    pylab.xlabel(r'$k_y$')
    pylab.ylabel(r'$\omega$')
    pylab.xticks([0,(avg_trans.shape[0])/2.,avg_trans.shape[0]],['0',r'$\pi$',r'2$\pi$'])

def PlotPeaksEdgeHighlight(spin='up'):
    # TODO: This was never finished - it should be!
    info = ReadInfo()
    omega,trans = ReadGkData(spin,real_freq=True)

    from pylab import plot,pcolor,arange,xlim,colorbar,title,ylim
    from matplotlib import cm
    from numpy import log10,meshgrid,pi

    # Sum over the middle third
    third = int(trans.shape[0] / 3)
    mid_trans = -trans[third:third*2].sum(axis=0).imag / third / pi
    mid_trans[mid_trans<0] = 1e-16

    left_trans = -trans[0].imag / pi
    left_trans[left_trans<0] = 1e-16

    pcolor(arange(avg_trans.shape[0]+1),omega.real,avg_trans.T)
    xlim(0,avg_trans.shape[0])
    ylim(-5,5)
    colorbar()
    

def PlotOmega0Data(spin='up',omega_index=None,map=True,filename=None):
    '''This function will plot the spectra at omega=0 with both x and k_y resolution. 
    
    If ``map==True`` then this will be shown as a color map with all details. An 
    example is shown below:

    .. figure:: example_figures/RDMFT_TISM/Ef_map.png

    If ``map==False`` then four separate values of x will be chosen, and 
    the corresponding :math:`G(\omega=0,k_y)` functions plotted. These values 
    are the left side, right side, and two central points. An example follows:

    .. figure:: example_figures/RDMFT_TISM/Ef_ky_edge_bulk_comp.png

    The idea of this function was to attempt to identify the transition points to 
    metallic/insulating phases with a more quantitative approach. Unfortunately it 
    cannot be done so easily due to the inherent broadening in the calculations.
    '''
    import os
    from numpy import load,arange,pi,meshgrid
    import pylab

    info = ReadInfo()
    if omega_index:
        assert os.path.exists('Gk.npz')
        data = load('Gk.npz')
        omega = data['omega']
        print(omega[omega_index])
        greens = data['greens']
    else:
        if filename == None:
            filename = 'Gk_o0.npz'

        assert os.path.exists(filename)
        greens = load(filename)['greens']
        omega_index = 0

    # Only get up spin for the moment
    if spin == 'up':
        greens = greens[0][:,:,omega_index]
    else:
        greens = greens[1][:,:,omega_index]

    greens = -greens.imag/pi

    if map:
        X,Y = meshgrid(arange(greens.shape[0]+1), arange(greens.shape[1]+1)*(2*pi)/greens.shape[1])
        pylab.pcolor(X,Y,greens.T)
        pylab.xlim(0,greens.shape[0])
        pylab.ylim(0,2*pi)
    else:
        ky = arange(greens.shape[1])*(2*pi)/greens.shape[1]
        pylab.plot(ky,greens[0],label='Left')
        pylab.plot(ky,greens[-1],label='Right')
        pylab.plot(ky,greens[greens.shape[0]/2],label='Mid1')
        pylab.plot(ky,greens[greens.shape[0]/2+1],label='Mid2')
        pylab.xlim(0,2*pi)
        pylab.legend()


def CalcNZInvariant(spin='up'):
    r'''Calculate the integer topological invariant of the effective spin-less 
    system. This is the Chern number which is proportional to the quantum Hall current 
    and is applicable to the QSH effect only when S_z is a good quantum number.

    The formula is:

    .. math::
    
        Z = (2\pi)^2 \int k_x k_y z G^{-1} \partial_{k_x} G G^{-1} \partial_{k_y} G G^{-1} \partial_{z} G 

    where z are Matsubara frequencies and :math:`G=G(k_x,k_y,z)`.

    .. warning::
        
        This code is currently not working.

    Note that this is only relevant for periodic boundaries.
    '''
    from numpy import dstack,arange,trapz,pi,r_
    info = ReadInfo()

    assert info['boundaries'] == 'PP'

    TI_FourierCTAUX(y_only=False,no_transform=False)
    #TI_FourierCTAUX(y_only=False,no_invert=True)

    omega,Gkk = ReadGkData(spin,kkdata=True)
    omega = omega.imag
    #omega = omega[info['num_omega']:]
    #Gkk = Gkk[:,:,info['num_omega']:]
    #omega = r_[-omega[::-1],omega]
    #Gkk = dstack((Gkk[:,:,::-1].conj(),Gkk))
    #print(omega)
    # Temp testing
    #Gkk = Gkk[0]

    #omega,Gkk_inv = ReadGkData(kkdata=True,no_invert=True,complete_mat=True)
    Gkk_inv = 1./Gkk

    Gkk_inv = (Gkk_inv[1:] + Gkk_inv[:-1])/2.
    Gkk_inv = (Gkk_inv[:,1:] + Gkk_inv[:,:-1])/2.
    Gkk_inv = (Gkk_inv[:,:,1:] + Gkk_inv[:,:,:-1])/2.

    kx = 2*pi * arange(info['N']) / info['N']
    ky = 2*pi * arange(info['M']) / info['M']

    GkkDX = (Gkk[1:] - Gkk[:-1]) / (kx[1:,None,None] - kx[:-1,None,None])
    GkkDX = (GkkDX[:,1:] + GkkDX[:,:-1])/2.
    GkkDX = (GkkDX[:,:,1:] + GkkDX[:,:,:-1])/2.

    integrand = Gkk_inv * GkkDX
    del GkkDX

    GkkDY = (Gkk[:,1:,:] - Gkk[:,:-1,:]) / (ky[None,1:,None] - ky[None,:-1,None])
    GkkDY = (GkkDY[1:] + GkkDY[:-1])/2.
    GkkDY = (GkkDY[:,:,1:] + GkkDY[:,:,:-1])/2.

    integrand *= Gkk_inv * GkkDY
    del GkkDY

    GkkDw = (Gkk[:,:,1:] - Gkk[:,:,:-1]) / (omega[None,None,1:] - omega[None,None,:-1])
    GkkDw = (GkkDw[1:] + GkkDw[:-1])/2.
    GkkDw = (GkkDw[:,1:] + GkkDw[:,:-1])/2.

    integrand = Gkk_inv * GkkDw
    del GkkDw
    
    Gkk_inv = (Gkk_inv[1:] - Gkk_inv[:-1])/2.
    Gkk_inv = (Gkk_inv[:,1:] - Gkk_inv[:,:-1])/2.
    Gkk_inv = (Gkk_inv[:,:,1:] - Gkk_inv[:,:,:-1])/2.

    #integrand = (Gkk_inv * GkkDX) * (Gkk_inv * GkkDY) * (Gkk_inv * GkkDw)
    integrand = integrand * (omega[1:] - omega[:-1])[None,None,:]/2.
    integrand = integrand * (kx[1:] - kx[:-1])[:,None,None]/2.
    integrand = integrand * (ky[1:] - ky[:-1])[None,:,None]/2.
    
    #import pdb;pdb.set_trace()

    # Trapz integration
    integrand[1:-1] *= 2
    integrand[:,1:-1] *= 2
    integrand[:,:,1:-1] *= 2

    #result = integrand.sum() * 1/6. * 3 / (2*pi)**2
    #result = integrand.sum() * 1/6. / (2*pi)**2
    result = integrand.sum() / (2*pi)**2

    with open('NZ_spin{0}.dat'.format(spin),'wb') as file:
        file.write('Calculated Z_N invariant from spin {0}: {1}\n'.format(spin,result.imag))

    return result

def GenerateAllDataCTAUX(use_U0=False,directfft=False,num_average=5):
    '''This is a convenience function to generate all the common graphs that are
    saved for one run of the code.

    If ``use_U0==True`` then a zero self-energy will be given to all the respective 
    functions. Otherwise ``num_average`` is the parameter that will designate how many 
    of the previous iterations will be averaged. See :func:`AverageData`.

    The following files will be saved:

        * ``SE_matsu_imag.png``
        * ``SE_matsu_real.png``
        * ``peaks_up.png``
        * ``spec_dens_ky_left.png``
        * ``spec_dens_ky_lefthalf.png``
        * ``spec_dens_ky_bulk.png``
        * ``spec_dens_ky.png``
        * ``edge_mid_comp.pdf``
        * ``edge_mid_comp.png``
        * ``spec_dens.pdf``
        * ``spec_dens.png``
        * ``Ef_ky_edge_bulk_comp.png``
        * ``Ef_map.png``

    as well as the following if ``directfft==True``:

        * ``spec_dens_ky_directfft.png``
        * ``spec_dens_ky_directfft_left.png``
        * ``spec_dens_ky_directfft_bulk.png``
    '''
    from numpy import r_
    
    #matplotlib.use('Agg')
    import pylab
    info = ReadInfo()

    if not use_U0:
        # Output the self-energy in Matsubara frequency for sanity checking.
        G,SE_iw = AverageData(-num_average,-1)
        import ImagConv
        omega = ImagConv.OmegaN(info['num_omega'],info['beta'])
        omega = r_[-omega[::-1],omega]
        pylab.figure()
        pylab.plot(omega,SE_iw[0].imag,'b')
        pylab.plot(omega,SE_iw[1].imag,'g')
        pylab.plot(omega,SE_iw[2].imag,'r')
        pylab.savefig('SE_matsu_imag.png')
        pylab.close()
        pylab.figure()
        pylab.plot(omega,SE_iw[0].real,'b')
        pylab.plot(omega,SE_iw[1].real,'g')
        pylab.plot(omega,SE_iw[2].real,'r')
        pylab.savefig('SE_matsu_real.png')
        pylab.close()
    
    TI_ContinueCTAUX_SE(use_U0=use_U0,num_average=num_average)

    TI_FourierCTAUX(only_omega0=True,num_average=num_average)
    TI_FourierCTAUXRealFreq(no_transform=True,use_U0=use_U0)
    logger.info("Done Gx")

    pylab.figure()
    TI_FourierCTAUXRealFreq(use_U0=use_U0)
    logger.info("Done Gk,peaks")
    pylab.ylim(-5,5)
    pylab.savefig('peaks_up.png')
    pylab.close()

    pylab.figure()
    PlotPeaksMap(x=0)
    pylab.savefig('spec_dens_ky_left.png')
    pylab.close()

    pylab.figure()
    PlotPeaksMap(x='lefthalf')
    pylab.savefig('spec_dens_ky_lefthalf.png')
    pylab.close()

    pylab.figure()
    PlotPeaksMap(x='mid')
    pylab.savefig('spec_dens_ky_bulk.png')
    pylab.close()

    pylab.figure()
    PlotPeaksMap()
    #pylab.ylim(-2,2)
    pylab.savefig('spec_dens_ky.png')
    pylab.close()

    if directfft:
        pylab.figure()
        PlotPeaksMap(directfft=True)
        pylab.savefig('spec_dens_ky_directfft.png')
        pylab.close()

        pylab.figure()
        PlotPeaksMap(x=0,directfft=True)
        pylab.savefig('spec_dens_ky_directfft_left.png')
        pylab.close()

        pylab.figure()
        PlotPeaksMap(x='mid',directfft=True)
        pylab.savefig('spec_dens_ky_directfft_bulk.png')
        pylab.close()

    # Edge/Inner Gk func comparison
    pylab.figure()
    PlotGkData('up',x=0)
    pylab.gca().lines[-1].set_label('Edge')
    PlotGkData('up',x=int(info['N']/2))
    pylab.gca().lines[-1].set_label('Mid1')
    PlotGkData('up',x=int(info['N']/2 + 1))
    pylab.gca().lines[-1].set_label('Mid2')
    pylab.xlim(-5,5)
    pylab.legend()
    pylab.xlabel(r'$\omega$')
    pylab.ylabel(r'$\int_0^\pi G_{xy} dy (\omega)$')
    pylab.title(r'$\alpha={p}/{q}$, $\lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    #pylab.savefig('edge_mid_comp.pdf')
    pylab.savefig('edge_mid_comp.png')
    pylab.close()

    # Spec density plot
    pylab.figure()
    PlotGkData('up',map=True)
    pylab.xlabel('$x$')
    pylab.ylabel(r'$\omega$')
    pylab.ylim(-5,5)
    pylab.title(r'Spec density: $\alpha={p}/{q}$, $\lambda_x={lambda_x}$, $\gamma={gamma}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    #pylab.savefig('spec_dens.pdf')
    pylab.savefig('spec_dens.png')
    pylab.close()

    # Edge ky plots
    pylab.figure()
    PlotOmega0Data(map=False)
    pylab.title(r'At $E_f$: $\alpha={p}/{q}$, $\lambda_x={lambda_x}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    pylab.xlabel('$k_y$')
    pylab.ylabel(r'$G_x(\omega=0,k_y)$')
    pylab.savefig('Ef_ky_edge_bulk_comp.png')
    pylab.close()

    pylab.figure()
    PlotOmega0Data(map=True)
    pylab.title(r'At $E_f$: $\alpha={p}/{q}$, $\lambda_x={lambda_x}$, $U={U}$, $({N},{M})$, {boundaries}'.format(**info))
    pylab.xlabel(r'Site index $x$')
    pylab.ylabel(r'$k_y$')
    pylab.savefig('Ef_map.png')
    pylab.close()

    # Probably only want to do this if we have interactions or have explicitly 
    # specified a filling.
    #if use_U0 == False or info['force_filling'] != None:
    #    CalcNZInvariant('up')
    #    CalcNZInvariant('down')

def GenerateAllDataForU0(N,M,boundaries,mu,p,q,lambda_x,gamma,beta,only_info=False):
    '''In order to generate output from the analysis functions, one requires a 
    ``info.dat`` file. However, for U=0 it is not necessary to run :func:`ConvergenceSpin` 
    and so one can create the necessary files with this function.

    The necessary arguments are ``N,M,boundaries,mu,p,q,lambda_x,gamma,beta`` as described by 
    :func:`GreensMatSinglePython`. There are some default parameters, which are:

        * ``num_omega = 400``
        * ``t = 1``
    
    This function calls :func:`GenerateAllDataCTAUX` to do the actual work. If you would 
    like to only create the ``info.dat`` file, then specify ``only_info==True``.
    '''

    with open('info.dat','wb') as file:
        file.write('Version {0}\n'.format(file_version))
        file.write('Solver CT-AUX\n')
        file.write('N,M {0} {1}\n'.format(N,M))
        file.write('Boundaries {0}\n'.format(boundaries))
        file.write('U 0\n')
        file.write('mu {0}\n'.format(mu))
        file.write('num_sweeps 0\n')
        file.write('num_omega 400\n')
        file.write('row_sym None\n')
        file.write('p,q {0} {1}\n'.format(p,q))
        file.write('beta {0}\n'.format(beta))
        file.write('t 1\n')
        file.write('lambda_x {0}\n'.format(lambda_x))
        file.write('Force_filling None\n')
        file.write('gamma {0}\n'.format(gamma))
        file.write('fit_large_SE False\n')
        file.write('-----\n')

    if not only_info:
        GenerateAllDataCTAUX(use_U0=True)
    #FIXME

def AverageData(first_iter=-5,last_iter=-1):
    '''A convenience function to average some iterations of the 
    self-consistency loops. Normally this will be the last 5 
    iterations, unless overridden with ``first_iter`` and 
    ``last_iter``. These parameters should be given as if python 
    indices to a list of iterations.
    '''
    
    from numpy import load,savez

    info = ReadInfo()
    
    if info['U'] == 0.:
        return None,None

    if last_iter == -1:
        # This means that we must take the last set. First find the last iter 
        # from the info.dat file.
        file = open('info.dat')
        iter = int(file.readlines()[-1].split()[0])

        first_iter = (iter+1) + first_iter
        last_iter = (iter+1) + last_iter

        first_iter = max(first_iter,0)
    else:
        assert first_iter > 0
        assert last_iter > 0

    # Check to see if there's an existing file with the required iterations.
    import os
    if os.path.exists('average_data.npz'):
        data = load('average_data.npz')
        if data['last_iter'] == last_iter and data['first_iter'] == first_iter:
            return data['G'],data['SE']
        else:
            logger.info("More iterations added: updating average_data.npz")

    G = []
    SE = []
    for i in irange(first_iter,last_iter):
        a = load('everything_iter{0}.npz'.format(i))
        G += [a['G']]
        SE += [a['SE']]

    G = avg(G)
    SE = avg(SE)

    savez('average_data.npz',G=G,SE=SE,first_iter=first_iter,last_iter=last_iter)

    return G,SE

def DisplayFillingFromNup(filename,rowsym_override=None,vmin=0.,vmax=0.5):
    '''Given a particular file with ``filename``, saved in the 
    ``Nup_Ndown_<iter>.dat`` format that :func:`ConvergenceSpin` does, this
    function will plot the filling, and various spins as an array of squares. 
    An example is below:

    .. figure:: example_figures/RDMFT_TISM/a1_6_gamma0_125_LAM0_0Stot_map.png

    The range of the colorbar can be specified with ``vmin`` and ``vmax``. 
    As well, one can override the number of blocks displayed, with the 
    ``rowsym_override`` parameter, by passing something that is expected 
    by the :func:`TI_Symmetry` function.

    Note that the function doesn't display the entire grid, but only the 
    small region specified by ``rowsym`` in the ``info.dat`` files or 
    ``rowsym_override``.

    The following files are saved:

        * ``Nftot_map.png``
        * ``Sz_map.png``
        * ``Sx_map.png``
        * ``Sy_map.png``
        * ``Stot_map.png``
    '''

    from numpy import genfromtxt,empty,sqrt
    info = ReadInfo()
    N,M = info['N'],info['M']

    file = open(filename)
    file.readline()
    data = genfromtxt(file)
    file.close()
    Nup,Ndown,Sx,Sy,Sz = data.T

    symmetry,potential = TI_Symmetry((N,M),info['row_sym'],info['lambda_x'])

    #Nup_mat = Nup.reshape(N,M)
    #Ndown_mat = Ndown.reshape(N,M)

    Nup_mat = empty((N*M))
    Ndown_mat = empty((N*M))
    Sx_mat = empty((N*M))
    Sy_mat = empty((N*M))
    Sz_mat = empty((N*M))

    num_Nones = 0
    for i in range(N*M):
        if symmetry[i] == None:
            symmetry[i] = num_Nones
            num_Nones += 1

        if symmetry[i] >= 0:
            Nup_mat[i] = Nup[symmetry[i]]
            Ndown_mat[i] = Ndown[symmetry[i]]
            Sx_mat[i] = Sx[symmetry[i]]
            Sy_mat[i] = Sy[symmetry[i]]
            Sz_mat[i] = Sz[symmetry[i]]
        else:
            Ndown_mat[i] = Nup[symmetry[i]]
            Nup_mat[i] = Ndown[symmetry[i]]
            Sx_mat[i] = -Sx[symmetry[i]]
            Sy_mat[i] = -Sy[symmetry[i]]
            Sz_mat[i] = -Sz[symmetry[i]]

    Nup_mat = Nup_mat.reshape(M,N)
    Ndown_mat = Ndown_mat.reshape(M,N)
    Ntot_mat = Nup_mat + Ndown_mat

    Sx_mat = Sx_mat.reshape(M,N)
    Sy_mat = Sy_mat.reshape(M,N)
    Sz_mat = Sz_mat.reshape(M,N)

    Stot_mat = sqrt(Sx_mat**2 + Sy_mat**2 + Sz_mat**2)

    if rowsym_override:
        info['row_sym'] = rowsym_override

    if info['row_sym'].startswith('rect'):
        repx,repy = info['row_sym'][4:].split('x')
        repx = float(repx)
        repy = float(repy)
        reduced_size = slice(repy),slice(repx)
        fig_size = (8,2)
        subplots_adjust = {'left':0.1,'right':0.75,'bottom':0.21,'top':0.81}
    else:
        reduced_size = slice(None),slice(None)
        fig_size = (8,6)
        subplots_adjust = {}

    from pylab import figure,imshow,colorbar,title,savefig,quiver
    from pylab import cm
    #colormap = cm.Greys
    #colormap = None
    #colormap = cm.summer
    #colormap = cm.spring
    #colormap = cm.cool
    #colormap = cm.PiYG
    #colormap = cm.RdGy
    #colormap = cm.winter
    #colormap = cm.PRGn
    #colormap = cm.RdBu
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
             'green' : ((0.0, 0.4, 0.4),
                        (0.5, 1.0, 1.0),
                        (1.0, 1.0, 1.0)),
             'blue' : ((0.0, 1.0, 1.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.4, 0.4))}
    from matplotlib.colors import LinearSegmentedColormap
    colormap = LinearSegmentedColormap('danny',cdict,256)
    colormap = None
    aspect = 'auto'
    #figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    #imshow(Nup_mat,interpolation='nearest',cmap=colormap,aspect=aspect)
    #colorbar()
    #figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    #imshow(Ndown_mat,interpolation='nearest',cmap=colormap,aspect=aspect)
    #colorbar()
    figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    imshow(Ntot_mat[reduced_size],interpolation='nearest',cmap=colormap,aspect=aspect,vmin=0.5,vmax=1.5)
    colorbar()
    title('Total filling')
    savefig('Nftot_map.png')
    figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    imshow(Sz_mat[reduced_size],interpolation='nearest',cmap=colormap,aspect=aspect,vmin=-0.5,vmax=0.5)
    colorbar()
    title('S_z')
    savefig('Sz_map.png')
    figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    imshow(Sx_mat[reduced_size],interpolation='nearest',cmap=colormap,aspect=aspect,vmin=-0.5,vmax=0.5)
    colorbar()
    title('S_x')
    savefig('Sx_map.png')
    figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    imshow(Sy_mat[reduced_size],interpolation='nearest',cmap=colormap,aspect=aspect,vmin=-0.5,vmax=0.5)
    colorbar()
    title('S_y')
    savefig('Sy_map.png')
    figure(figsize=fig_size).subplots_adjust(**subplots_adjust)
    imshow(Stot_mat[reduced_size],interpolation='nearest',cmap=colormap,aspect=aspect,vmin=vmin,vmax=vmax)
    colorbar()
    title('S_tot - filling conv is {0}'.format(info['Nup_diff']))

    # Assume that all the non-z stuff is in y.
    #X,Y = meshgrid(arange(N),arange(M))
    #X = X.flatten()
    #Y = Y.flatten()
    u = (Sy_mat/Stot_mat)
    v = (Sz_mat/Stot_mat)
    quiver(u[reduced_size],v[reduced_size],pivot='middle')
    savefig('Stot_map.png')

    #return Nup_mat
    return Nup_mat,Ndown_mat,Sz,Stot_mat,reduced_size



def FitMagExp(num_iters=70,show_figures=True,also_last=False):
    '''This is a convenience function to extrapolate the magnetisation 
    calculated in the DMFT iterations to infinite iteration. It assumes the 
    magnetisation can be found in the ``info.dat`` file in the second last 
    column.
    '''

    from numpy import exp,arange
    from scipy.optimize import leastsq

    with open('info.dat') as file:
        lines = file.readlines()
        file_iters = int(lines[-1].split()[0])
        if file_iters < 10:
            temp = float(lines[-1].split()[-2])
            if also_last:
                return temp, temp
            else:
                return temp
        if file_iters-10 < num_iters:
            lines = lines[-file_iters+10:]
        else:
            lines = lines[-num_iters:]
        mags = [float(z.split()[-2]) for z in lines]

    func = lambda (a,b,c): a*exp(b*arange(len(mags))) + c - mags
    fit,dummy,infodict,mesg,ier = leastsq(func,(1.,-1.,0.),full_output=True,maxfev=10000)

    #print(ier,mesg)
    print(avg(infodict['fvec']**2))
    #print(fit)
    #print(lines[-1].split()[0])

    #assert ier in [1,2,3,4]
    #assert avg(infodict['fvec']**2) < 1e-5

    if show_figures:
        from pylab import figure,plot
        figure()
        plot(mags)
        plot(func(fit) + mags)

    if also_last:
        return fit[-1],mags[-1]
    else:
        return fit[-1]


def JudgeValidity(num_average=5):
    '''This function tests the sanity and validity of the results from a TISM 
    run.

    It will distinguish between being in the RDMFT runs, or the continuation 
    part, by looking at the directory name ('covar*'). If it is in a 
    continuation directory then it will (in future) run further tests.

    The current tests are:

        * Existing iterations.
        * Force filling converged.
        * For ED: filling convergence.
        * Negative imaginary diagonal self-energy (and consequently real 
          spectra).

    **Returns:**
        ``validity,mesg``

        ``validity`` (boolean)
        ``mesg_list`` (list of strings): Reasons when invalid.
    '''

    import os
    from numpy import r_,genfromtxt

    if os.path.abspath('.').split('/')[-1].startswith('covar'):
        dirprefix = '../'
    else:
        dirprefix = ''

    info = ReadInfo()
    num_omega = info['num_omega']

    mesg_list = []

    try:
        # If there are no iterations, then this is obviously not valid!
        if info['num_iters'] == None:
            mesg_list += ['No iterations']

        # Read the final iteration data.
        with open(dirprefix + 'Nup_Ndown_{0}.dat'.format(info['num_iters'])) as file:
            #Nup,Ndown,Sx,Sy,Sz = genfromtxt(file,skip_header=1).T
            Nup,Ndown,Sx,Sy,Sz = genfromtxt(file,skiprows=1).T

        avg_Nf = avg(Nup + Ndown)
        avg_Sx = avg(Sx)
        avg_Sy = avg(Sy)
        avg_Sz = avg(Sz)

        # Check the final filling against the desired one.
        if info['force_filling'] == None:
            force_filling = 1.0
        else:
            force_filling = info['force_filling']

        tol = 1e-2
        if avg_Nf - force_filling > tol:
            mesg_list += ['Did not reach desired filling']

        # Check the differences in filling in the last two iters.
        # Only check ED as CT can vary a lot anyway.
        if info['solver'].startswith('ED'):
            with open(dirprefix + 'Nup_Ndown_{0}.dat'.format(info['num_iters']-1)) as file:
                Nup2,Ndown2,Sx2,Sy2,Sz2 = genfromtxt(file,skiprows=1).T

            maxNdiff = abs(Nup - Nup2).max()
            if maxNdiff > 1e-3:
                mesg_list += ['Max diff is greater than 1e-3']


        # Check for problems with the imaginary part of the diagonal self-energy
        G,SE = AverageData(-num_average,-1)
        SE_diag = r_[SE[0][num_omega:].flatten(),SE[1][num_omega:].flatten()]

        if SE_diag[SE_diag.imag > 0].imag.sum() / SE_diag.imag.sum() > 1e-2:
            mesg_list += ['Diagonal self-energy should have imaginary part always negative.']
    except:
        mesg_list += ['Exception in JudgeValidity!']


    with open('validity.txt','wb') as file:
        if len(mesg_list) == 0:
            file.write('Valid!\n')
        else:
            for mesg in mesg_list:
                file.write(mesg + '\n')

    if len(mesg_list) == 0:
        return True,['Valid!']
    else:
        return False,mesg_list

def LookForGaps():
    from numpy import load
    info = ReadInfo()
    #data = load('everything_iter{0}.npz'.format(info['num_iters']))
    G,SE = AverageData()

    difflist = []
    fitlist = []
    for i in range(info['N']):
        func = G[0][info['num_omega']:,i]
        reldiff = abs(func[0].imag) / abs(func[1].imag)
        print("Site {0}: {1}".format(i,reldiff))
        difflist += [reldiff]

        # Also look at the extrapolated omega=0
        import ImagConv
        x = ImagConv.OmegaN(info['num_omega'],info['beta'])
        x = x[:9]
        y = func[:9].imag
        from numpy import polyfit
        fit = polyfit(x,y,2)
        fitlist += [fit[-1]]

    return fitlist,difflist


def PlotLastMags():
    from numpy import genfromtxt
    from pylab import plot
    info = ReadInfo()

    for i in range(info['num_iters']-4,info['num_iters']+1):
        mag = genfromtxt('Nup_Ndown_{0}.dat'.format(i),skiprows=1).T[-1]
        plot(mag,marker='.')

def PlotGkLargestAtZero(num_x=10):
    from pylab import plot,legend
    omega,G = ReadGkData()

    for x,item in enumerate(G[:num_x]):
        ind = item[:,399].imag.argmax()

        plot(omega.imag,item[ind].imag,label='x={0}'.format(x))

    legend()

def DoNiceMagPlot(filename):
    from pylab import title,xticks,yticks,grid,close
    out = DisplayFillingFromNup(filename,rowsym_override='rect6x2',vmin=0.2,vmax=0.40)

    for i in range(1,5):
        close(i)

    title('')
    #xticks(iarange(0.5,11.5),['']*11)
    #yticks([0.5,1.5,2.5,3.5],['','','',''])
    xticks(iarange(0.5,5.5),['']*5)
    yticks([0.5],[''])
    grid()


# Copied this directly from SS_NRG
def KramersKronig(omega,spec):
    from numpy import zeros,pi,r_
    from scipy.integrate import simps,trapz
    real_part = zeros(len(omega),float)

    # Suppress warnings.
    assert (spec.imag == 0).all()
    spec = spec.real
    omega = omega.real

    for i in range(len(omega)):
        real_part[i] = 0.
        if i > 0:
            temp = 1./pi * -pi * spec[:i] / (omega[:i] - omega[i])
            real_part[i] += trapz( temp, omega[:i].real)
        if i < len(omega)-1:
            temp = 1./pi * -pi * spec[i+1:] / (omega[i+1:] - omega[i])
            real_part[i] += trapz( temp, omega[i+1:].real)

    out = real_part + 1j*(-1.*pi)*spec

    return out

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
# OLD STUFF

if False:
    def ReadUpToSentinel(file,sentinel='DATA'):
        while True:
            c = file.read(1)
            if c == '':
                raise ValueError("Reached EOF")
            if c == sentinel[0]:
                i = 1
                while i < len(sentinel):
                    c = file.read(1)
                    if c != sentinel[i]:
                        break
                    i += 1
                else:
                    return True

