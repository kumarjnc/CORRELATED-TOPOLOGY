from __future__ import print_function
from std_imports import *

from pylab import *
from numpy import *

from RDMFT_TISM import GreensMatSinglePython as GreensMatSingle

def NonInteractingProblem(potential,(N,M),boundaries,(p,q),num_fraction,gamma,allow_total_mag=False):
    '''This function will solve the non-interacting problem in a naive way, 
    while allowing for an arbitrary potential. This is used in the mean-field solution.
    
    If ``potential == None``, then a zero potential is assumed. Otherwise, potential should be a tuple of size two, with ``potential = (pot_up,pot_down)``.'''

    from numpy import zeros,argsort,arange,c_,diag,tile
    from numpy.linalg import eigh,eig,qr

    if potential == None:
        potential = [zeros(N*M),zeros(N*M)]
    elif type(potential) == float:
        # This is assuming staggering
        temp = tile(arange(N),(M,1)).T.flatten()
        potential = (-1)**temp * potential
        potential = [potential,potential]
    else:
        assert len(potential) == 2
        assert potential[0].shape == potential[0].shape == (N*M,)


    SE = (zeros(N*M,complex),zeros(N*M,complex),zeros(N*M,complex),zeros(N*M,complex))
    # Here we set the potential to zero. This will be corrected later.
    mu = 0.
    mat = GreensMatSingle(SE,zeros(N*M),(N,M),boundaries,0.,(p,q),mu,gamma,no_invert=True,complete_mat=True)
    # Need a negative here as mat is really (w - H) at the moment (with w=0).
    mat = -mat

    # Get eigenvalues/eigenvectors
    mat = mat.reshape(N*M*2,N*M*2)
    # First add in potential. Rearrange it to be alternating with spins.
    if not allow_total_mag:
        diff = (potential[0] - potential[1]).sum() / (N*M)
        potential[0] -= diff/2.
        potential[1] += diff/2.
    potential_alt = c_[potential[0],potential[1]].flatten()
    mat = mat + diag(potential_alt)

    if gamma == 0.:
        # Split this into the up and down parts
        mat_spin = [mat[::2,::2], mat[1::2,1::2]]
        evecs = [None,None]
        evals = [None,None]
        for spin in [0,1]:
            evals[spin],evecs[spin] = eig(mat_spin[spin])
            #evals[spin],evecs[spin] = eigh(mat_spin[spin])
            evecs[spin] = evecs[spin][:,evals[spin].argsort()]
            evals[spin].sort()
            # Need to orthogonalise the eigenvectors
            Q,R = qr(evecs[spin])
            evecs[spin] = Q

        import __main__
        __main__.evecs = evecs
        #tot_evecs = c_[evecs[0],evecs[1]]
        tot_evecs = zeros((N*M*2,N*M*2),complex)
        tot_evecs[::2,:N*M] = evecs[0]
        tot_evecs[1::2,N*M:] = evecs[1]
        tot_evals = r_[evals[0],evals[1]]
        tot_evecs = tot_evecs[:,tot_evals.argsort()]
        tot_evals.sort()

        if not allow_total_mag:
            E = 0.
            num = int(N*M * num_fraction)
            #evecs_small = c_[evecs[0][:,:num], evecs[1][:,:num]]
            Nf_up = (abs(evecs[0][:,:num])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            #Nf_up = Nf_up.reshape(M,N).T
            Nf_up = Nf_up.reshape(N,M)
            E += sum(evals[0][:num])
            Nf_down = (abs(evecs[1][:,:num])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            #Nf_down = Nf_down.reshape(M,N).T
            Nf_down = Nf_down.reshape(N,M)
            E += sum(evals[1][:num])
            E = E.real
        else:
            num = int(N*M*2 * num_fraction)
            evecs_small = tot_evecs[:,:num]
            E = sum(tot_evals[:num])
            E = E.real

            # Times 2 so that max filling is 2
            Nf_up = (abs(evecs_small[::2])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            Nf_up = Nf_up.reshape(M,N).T
            Nf_down = (abs(evecs_small[1::2])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            Nf_down = Nf_down.reshape(M,N).T
    else:
        tot_evals,tot_evecs = eig(mat)
        #tot_evals,tot_evecs = eigh(mat)
        tot_evecs = tot_evecs[:,tot_evals.argsort()]
        tot_evals.sort()
        # Need to orthogonalise the eigenvectors
        Q,R = qr(tot_evecs)
        tot_evecs = Q

        #if allow_total_mag:
        if True:
            num = int(N*M*2 * num_fraction)
            evecs_small = tot_evecs[:,:num]
            E = sum(tot_evals[:num])
            E = E.real

            # Times 2 so that max filling is 2
            Nf_up = (abs(evecs_small[::2])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            Nf_up = Nf_up.reshape(M,N).T
            Nf_down = (abs(evecs_small[1::2])**2).sum(axis=1)# / (len(evals) / (N*M)) * 2
            Nf_down = Nf_down.reshape(M,N).T
        else:
            raise NotImplementedError

    return Nf_up,Nf_down,tot_evals,tot_evecs,mat,E

def MeanFieldHamiltonian(Nf,U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma,row_sym=False,**kwds):
    if not isiterable(V_pot):
        V_pot = StaggeredPot(V_pot,(N,M))

    #if row_sym:
    #    assert len(Nf[0]) == N
    #    old_Nf = Nf
    #    Nf = [None,None]
    #    Nf[0] = old_Nf[0][None,:].repeat(M,axis=0)
    #    Nf[1] = old_Nf[1][None,:].repeat(M,axis=0)
    #else:
    if True:
        #Nf = [Nf[0].T, Nf[1].T]
        pass

        #print(Nf[0].T)

    tot_pot = c_[V_pot+Nf[0].flatten()*U,V_pot+Nf[1].flatten()*U].T
    Nf_up,Nf_down,evals,evecs,mat,E = NonInteractingProblem(tot_pot,(N,M),boundaries,(p,q),num_fraction,gamma,**kwds)

    return Nf_up,Nf_down,evals,evecs,mat,E

def MeanFieldConsistency(U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma,num_iters=20,bias_iters=2,fixup_point=60,starting_Nf=None,allow_kb_interrupt=True,row_sym=False,**kwds):
    '''Perform the consistency loops to determine the mean-field solution.

    Note that if V_pot is a constant, then staggering is assumed with this value.
    
    The chemical potential is automatically adjusted for the interaction to place mu=0 at half-filling.'''

    #mu = mu + U/2.

    from numpy import c_,arange
    if not isiterable(V_pot):
        V_pot = StaggeredPot(V_pot,(N,M))

    if starting_Nf is None:
        tot_pot = array([V_pot,V_pot])
    else:
        if len(starting_Nf[0].shape) == 1:
            tot_pot = array([V_pot+starting_Nf[0]*U,V_pot+starting_Nf[1]*U])
        else:
            #tot_pot = array([V_pot+starting_Nf[0].T.flatten()*U,V_pot+starting_Nf[1].T.flatten()*U])
            tot_pot = array([V_pot+starting_Nf[0].flatten()*U,V_pot+starting_Nf[1].flatten()*U])

    Nf_up = None
    old_Nf_up = None
    try:
        for iter in range(num_iters):
            old2_Nf_up = old_Nf_up
            old_Nf_up = Nf_up
            Nf_up,Nf_down,evals,evecs,mat,E = NonInteractingProblem(tot_pot,(N,M),boundaries,(p,q),num_fraction,gamma,**kwds)

            #meanfield_up = Nf_down.T.flatten() * U
            #meanfield_down = Nf_up.T.flatten() * U
            meanfield_up = Nf_down.flatten() * U
            meanfield_down = Nf_up.flatten() * U

            if row_sym:
                if row_sym == True:
                    row_sym = 1

                meanfield_up = meanfield_up.reshape(N,M)
                for m in range(row_sym,M):
                    meanfield_up[:,m] = meanfield_up[:,m % row_sym]
                meanfield_up = meanfield_up.flatten()
                meanfield_down = U * (1 - meanfield_up/U)

            #meanfield_up = (meanfield_up + meanfield_down)/2.
            #meanfield_down = meanfield_up

            #meanfield_up = meanfield_up.T
            #meanfield_down = meanfield_down.T
            #V_pot = V_pot.T

            old_tot_pot = tot_pot
            new_tot_pot = array([meanfield_up + V_pot,
                                 meanfield_down + V_pot])
            #alpha = 0.4
            alpha = 1. - 0.2*rand(1)
            #alpha = 1.0
            #alpha = 0.05
            tot_pot = (1-alpha)*old_tot_pot + alpha*new_tot_pot

            if iter < bias_iters:
                # For the bias iters, we added a staggered magnetisation over the 
                # lattice
                #tot_pot[0] += ((-1)**arange(M)[:,None] + (-1)**arange(N)[None,:]).flatten()
                #tot_pot[1] += -((-1)**arange(N)[:,None] + (-1)**arange(M)[None,:]).flatten()

                # Also add a small randomisation to mix things up
                # Adding iter dependence here so to control the strength of 
                # randomisation by specifying how many bias iters to use.
                tot_pot[0] += (iter+1)*rand(N*M)/2. * tot_pot.max()
                tot_pot[1] += (iter+1)*rand(N*M)/2. * tot_pot.max()

            if iter >= fixup_point and iter % 10 == 0:
                # Try to fix up any oscilations
                #ind = abs(Nf_up - old_Nf_up) > (1./iter)
                ind = abs(Nf_up - old_Nf_up) > 0.1
                #logger.info(ind.sum())
                ind = ind.flatten()
                if True:
                    before = tot_pot.copy()
                    scale = tot_pot.max()
                    tot_pot[0][ind] += scale*rand(N*M)[ind]
                    tot_pot[1][ind] += scale*rand(N*M)[ind]
                    after = tot_pot
                    #return before,after
                else:
                    old = old_tot_pot
                    before = tot_pot.copy()
                    tot_pot[0][ind] = (old_tot_pot[0][ind] + tot_pot[0][ind])/2.
                    tot_pot[1][ind] = (old_tot_pot[1][ind] + tot_pot[1][ind])/2.
                    tot_pot[0][ind] = (tot_pot[0][ind] + tot_pot[1][ind])/2.
                    tot_pot[1][ind] = tot_pot[0][ind]
                    after = tot_pot
                    #return old,before,after,ind

            # Temporary Danny forcing
            #tot_pot[0] = tot_pot[0][:N].repeat(M)
            #tot_pot[1] = tot_pot[0][:N].repeat(M)

            if iter > 0:
                diff = abs(Nf_up - old_Nf_up).sum() / (N*M)
            else:
                diff = None
            if iter > 1:
                diff_2 = abs(Nf_up - old2_Nf_up).sum() / (N*M)
            else:
                diff_2 = None

            mag = avg((Nf_up - Nf_down).flatten())
            tot_Nf = avg((Nf_up + Nf_down).flatten())
            meanfield_contrib = meanfield_up.sum()/(N*M)
            logger.info("Done iter {0}, total mag is {1}, total_Nf is {2}, diff is {3}, diff2 is {4}, MF_contrib is {5}, E is {6}".format(iter,mag,tot_Nf,diff,diff_2,meanfield_contrib,E))

            if iter > 2 and diff < 1e-8 and diff_2 < 1e-8:
            #if iter > 2 and diff_2 < 1e-15:
            #if False:
                tol_count += 1
                if tol_count >= 3:
                    logger.info("Stopping because reached tolerance")
                    break
            else:
                tol_count = 0

    except KeyboardInterrupt:
        if not allow_kb_interrupt:
            raise

    #E = sum(evals[evals<0]).real
    logger.info("Total energy is {0}".format(E))
    return Nf_up,Nf_down,evals,evecs,mat,old_Nf_up,E,diff,diff_2

def MeanFieldGroundState(U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma,num_iters=100,quiet=False,consistency_iters=55,**kwds):

    # First work out paramagnetic ground state energy.
    Nf = ones(N*M) * num_fraction
    Nf_up,Nf_down,evals,evecs,mat,E = MeanFieldHamiltonian([Nf,Nf],U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma,**kwds)
    from numpy import savez
    assert E == E
    
    logger.info("Paramagnetic ground state energy is {0}".format(E))


    E = 999.
    Nf_list = []
    Nf_iter = None
    best_out = None
    same_count = 0

    required_same = 4
    bias_cycle = 3

    def plotfig(data):
        figure();imshow(concatenate((data[0].T,ones((M,1))*0.5,data[1].T,ones((M,1))*0.5,(data[0].T + data[1].T)/2.),axis=1),interpolation='nearest'); colorbar()

    if quiet:
        printing = NoPrint()
    else:
        printing = DummyWith()

    for iter in range(num_iters):
        #if Nf is not None:
        #    logger.info("Difference = {0}".format(abs(Nf_iter[0] - Nf[0]).sum()))
        with printing:
            out = MeanFieldConsistency(U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma,num_iters=consistency_iters,bias_iters=(iter-1)%bias_cycle+1,fixup_point=10,starting_Nf=Nf_iter,allow_kb_interrupt=False,**kwds)

        new_E = out[6]
        new_Nf = [out[0],out[1]]
        new_diff = out[7]

        Nf_tot = (new_Nf[0].sum() + new_Nf[1].sum())/(N*M)

        mag = new_Nf[0] - new_Nf[1]
        mag_tot = mag.sum()
        mesg = "E is {0}, Nf_tot is {1}, Mag tot is {2:6.3f}, Max mag {3:6.3f}".format(new_E,Nf_tot,mag_tot,mag.max())

        if new_diff > 1e-5:
            mesg += ": Not including attempt - no convergence"
            new_E = 0.
            # But still go from this point for the next loop if we haven't got 
            # anywhere yet
            #if Nf is None:
            #    Nf = new_Nf
            #same_count = 0
        elif not kwds.get('allow_total_mag',False) and abs(mag_tot) > 1e-5:
            mesg += ": Not including attempt - total magnetisation present"
            new_E = 0.
            same_count = 0
        elif abs(new_E - E) < 1e-3:
            same_count += 1
            for i in range(len(Nf_list)):
                fill_diff = abs(new_Nf[0] - Nf_list[i][0]).sum() / (N*M*2)
                if fill_diff < 1e-3:
                    break
            else:
                plotfig(new_Nf)
                Nf_list += [new_Nf]
            mesg += ", Difference of {0:6.5f} in the up filling".format(fill_diff)
            if same_count >= required_same-1:
                break
        elif new_E < E:
            mesg += ": Accepting! Kaching***"
            Nf_list = [new_Nf]
            E = new_E
            best_out = out
            same_count = 0
            plotfig(new_Nf)
        else:
            mesg += ": Rejecting!"
            #same_count = 0
            #plotfig()

        Nf_iter = new_Nf

        logger.info(mesg)

    return best_out,Nf_list

def ShowStatekSpace((N,M),evals,evecs,states=[0,1,2,3]):
    from numpy.fft import fftshift, fft

    #evecs_k = fftshift(fft(fft(evecs.reshape(M,N,2,N*M*2),axis=1),axis=0),axes=(0,1))
    #evecs_k = fft(fft(evecs.reshape(M,N,2,N*M*2),axis=1),axis=0)
    evecs_k = fft(fft(evecs.reshape(N,M,2,N*M*2),axis=1),axis=0)

    temp = avg(abs(evecs_k[...,i])**2 for i in states)
    print('Sum is ',temp[...,0].sum())
    print(temp[...,0])
    imshow(abs(temp[...,0].T),interpolation='nearest'); colorbar()

def StaggeredPot(val,(N,M)):
    if val == 0:
        return zeros(N*M)

    row = val * (-1)**arange(N)
    #complete = row[:,None].repeat(M,axis=1)
    complete = row[None,:].repeat(M,axis=0)
    #row = val * (-1)**arange(M)
    #complete = row[None,:].repeat(N,axis=0)

    return complete.T.flatten()

def ShowSpectra((N,M),evals,evecs,left=False,spin=0,show_figures=True):
    import numpy.fft as fft
    num_bins = 200
    Ebins = linspace(min(evals),max(evals),num_bins)
    Ebin_spacing = Ebins[1] - Ebins[0]
    print("Spacing is {0}".format(Ebin_spacing))

    bins = zeros((num_bins,M))

    for i in range(len(evals)):
        bin_i = (evals[i] - Ebins[0]) / Ebin_spacing
        bin_i = int(bin_i.real)

        #temp = evecs[:,i].reshape(M,N,2)
        temp = evecs[:,i].reshape(N,M,2)
        # Can I Fourier transform after summing? No!
        temp = temp[:,:,spin]
        #temp = fft.fft(temp,axis=0)
        temp = fft.fft(temp,axis=1)
        temp = abs(temp)**2
        if left == True:
            #temp = temp[:,:1].sum(axis=1)
            temp = temp[:N/2].sum(axis=0)
        elif left == 'mid':
            #temp = temp[:,int(N/3):int(2*N/3)].sum(axis=1)
            temp = temp[int(N/3):int(2*N/3)].sum(axis=0)
        else:
            #temp = temp.sum(axis=1)
            temp = temp.sum(axis=0)

        bins[bin_i] += temp

    if show_figures:
        figure()
        imshow(bins[::-1],aspect='auto',interpolation='nearest')

    return bins



# currently testing
# Nf_up,Nf_down,evals,evecs,mat = RDMFT_TISM_Meanfield.MeanFieldConsistency(5.,0.,(N,M),'PP',(0,6),0.+1e-8,0.,num_iters=500,bias_iters=2,fixup_point=100)
# imshow(concatenate((Nf_up.T,Nf_down.T),axis=1),interpolation='nearest'); colorbar()
# Nf_up,Nf_down,evals,evecs,mat,old_Nf_up,E,diff,diff_2 = RDMFT_TISM_Meanfield.MeanFieldGroundState(15.,0.,(N,M),'PP',(0,6),0.+1e-8,0.,quiet=True,num_iters=500)



def MeanFieldAFHamiltonian_Periodic(mag,cdw,U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma):
    assert N%2 == 0
    assert M%2 == 0
    assert boundaries == 'PP'
    assert num_fraction == 0.5


    Nf_up = array([0.5 + cdw/2. + mag[0]/2.,0.5 - cdw/2. + mag[1]/2.])
    Nf_down = array([0.5 + cdw/2. - mag[0]/2.,0.5 - cdw/2. - mag[1]/2.])
    Nf_up = Nf_up[None,:].repeat(N/2,axis=1).repeat(M,axis=0)
    Nf_down = Nf_down[None,:].repeat(N/2,axis=1).repeat(M,axis=0)

    Nf = array([Nf_up,Nf_down])

    Nf_up,Nf_down,evals,evecs,mat,E = MeanFieldHamiltonian(Nf,U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma)

    return Nf_up,Nf_down,evals,evecs,mat,E

def MeanFieldAFMinimization(U,V_pot,(N,M),(p,q),gamma):

    mag = [0.,0.]
    cdw = 0.

    def func(args,mag1):
        #mag1,mag2,cdw = args
        mag2,cdw = args
        Nf_up,Nf_down,evals,evecs,mat,E = MeanFieldAFHamiltonian_Periodic([mag1,mag2],cdw,U,V_pot,(N,M),'PP',(p,q),0.5,gamma)

        new_mag1 = Nf_up[0,0] - Nf_down[0,0]
        new_mag2 = Nf_up[0,1] - Nf_down[0,1]
        new_cdw = (Nf_up[0,0] + Nf_down[0,0])/2. - (Nf_up[0,1] + Nf_down[0,1])/2.

        #print(args)
        #print(new_mag1,new_mag2,new_cdw)
        #print(new_mag1-mag1,new_mag2-mag2,new_cdw-cdw)
        print(E)
        return new_mag1-mag1,new_mag2-mag2,new_cdw-cdw
    try:
        import scipy.optimize
        for mag1 in arange(0.,1.,0.1):
            print(scipy.optimize.leastsq(func,[0.2,0.1],(mag1,)))
    except KeyboardInterrupt:
        if not allow_kb_interrupt:
            raise

    #E = sum(evals[evals<0]).real
    logger.info("Total energy is {0}".format(E))
    return Nf_up,Nf_down,evals,evecs,mat,old_Nf_up,E,diff,diff_2

def MeanFieldCDWHamiltonian_Periodic(cdw,U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma):
    from numpy import tile
    assert N%2 == 0
    assert M%2 == 0
    assert boundaries == 'PP'

    Nf_up = array([num_fraction + cdw/2.,num_fraction - cdw/2.])
    Nf_down = array([num_fraction + cdw/2.,num_fraction - cdw/2.])
    Nf_up = tile(Nf_up[:,None],(N/2,M))
    Nf_down = tile(Nf_down[:,None],(N/2,M))

    Nf = array([Nf_up,Nf_down])

    Nf_up,Nf_down,evals,evecs,mat,E = MeanFieldHamiltonian(Nf,U,V_pot,(N,M),boundaries,(p,q),num_fraction,gamma)
    return Nf_up,Nf_down,evals,evecs,mat,E,Nf

def MeanFieldCDWMinimization(U,V_pot,(N,M),num_fraction,(p,q),gamma):

    cdw = 0.5

    def func(args):
        cdw = args[0]
        Nf_up,Nf_down,evals,evecs,mat,E,orig = MeanFieldCDWHamiltonian_Periodic(cdw,U,V_pot,(N,M),'PP',(p,q),num_fraction,gamma)
        
        #import pdb;pdb.set_trace()

        new_cdw = (Nf_up[0,0] + Nf_down[0,0])/2. - (Nf_up[1,0] + Nf_down[1,0])/2.
        #logger.info(new_cdw-cdw)

        return new_cdw-cdw
    try:
        import scipy.optimize
        final_cdw,flag = scipy.optimize.leastsq(func,[0.2])#,ftol=1e-6,xtol=1e-6)
    except KeyboardInterrupt:
        if not allow_kb_interrupt:
            raise

    final_cdw = final_cdw[0]
    Nf_up,Nf_down,evals,evecs,mat,E,orig = MeanFieldCDWHamiltonian_Periodic(final_cdw,U,V_pot,(N,M),'PP',(p,q),num_fraction,gamma)

    err = abs(Nf_up - orig[0]).sum() / (N*M)
    if err > 0.01:
        logger.error("Warning large error! With U={0}".format(U))
        #logger.info(Nf_up)
        #logger.info(orig[0])

    return Nf_up,Nf_down,evals,evecs,mat,E,orig,final_cdw,err

def GapSize(evals,(N,M),num_fraction):
    return evals[int(N*M*2*num_fraction)] - evals[int(N*M*2*num_fraction) - 1]

def PlotGapVsU(V_pot,(N,M),num_fraction,(p,q),gamma):
    U_list = linspace(0.,5.,10)
    gap_list = []

    from dancommon import CreatePool
    num,pool = CreatePool()

    res = []
    for U in U_list:
        res += [pool.apply_async(MeanFieldCDWMinimization,(U,V_pot,(N,M),num_fraction,(p,q),gamma))]

    pool.close()
    pool.join()

    for i in range(len(U_list)):
        out = res[i].get()
        gap = GapSize(out[2],(N,M),num_fraction)
        gap_list += [gap]
        logger.info("Gap is {0}".format(gap))

    from pylab import plot
    plot(U_list,gap_list)
    return U_list,gap_list

def BisectGap(V_pot,(N,M),num_fraction,(p,q),gamma):
    import scipy.optimize

    def func(U):
        out = MeanFieldCDWMinimization(U,V_pot,(N,M),num_fraction,(p,q),gamma)
        gap = GapSize(out[2],(N,M),num_fraction)
        logger.info("Got gap {0} for U {1}".format(gap,U))
        return gap

    return scipy.optimize.leastsq(func,0,ftol=1e-3,xtol=1e-3)[0]

def RunAndAppend(V_pot,(N,M),num_fraction,(p,q),gamma):
    logger.info("Running N={N} and M={M} with V={V_pot}".format(**locals()))
    U = BisectGap(V_pot,(N,M),num_fraction,(p,q),gamma)

    with FileLock():
        filename = 'MF_critU_a{p}_{q}_frac{num_fraction}_LAM{V_pot}.dat'.format(**locals())
        file = open(filename,'ab')
        file.write('N{N}M{M} {U}\n'.format(**locals()))


def RunLots(num_fraction,(p,q),gamma):
    V_list = [1.1,1.2,1.4,1.5,1.8,2.0]
    Nlist = [6,12,18,24,30,60,120]

    combinations = [(N,M,N*M) for N in Nlist for M in Nlist]
    combinations.sort(key=lambda x: x[2])

    from dancommon import CreatePool
    num,pool = CreatePool()

    for N,M,z in combinations:
        if z > 1800:
            continue
        for V_pot in V_list:
            pool.apply_async(RunAndAppend,(V_pot,(N,M),num_fraction,(p,q),gamma))

    pool.close()
    pool.join()


from pylab import *
def StaggeredMagField():
    N,M = 31,60
    boundaries = 'OP'
    p,q = 1,6
    num_fraction = 0.5
    gamma = 0.25
    lambda_x = 0.6
    fake_mag = 0.2

    potential = lambda_x * (-1)**tile(arange(N),(M,1)).T.flatten()
    mag = fake_mag * (-1)**tile(arange(M),(1,N)).T.flatten()
    print(potential)
    print(fake_mag)
    pot = [potential+mag,potential-mag]
    Nf_up,Nf_down,evals,evecs,mat,E = NonInteractingProblem(pot,(N,M),boundaries,(p,q),num_fraction,gamma)

    ShowSpectra((N,M),evals,evecs,left='mid')
    ShowSpectra((N,M),evals,evecs)

    return Nf_up,Nf_down,evals,evecs,mat,E

def TestingOrthogonality():
    N,M = 21,60
    boundaries = 'OP'
    p,q = 1,6
    num_fraction = 0.5
    gamma = 0.0
    lambda_x = 0.0

    Nf_up,Nf_down,evals,evecs,mat,E = NonInteractingProblem(0.,(N,M),boundaries,(p,q),num_fraction,gamma)

    ShowSpectra((N,M),evals,evecs,left='mid')
    ShowSpectra((N,M),evals,evecs)

    TestingOrthogonality2(evecs,N,M)

    return Nf_up,Nf_down,evals,evecs,mat,E

def TestingOrthogonality2(evecs,N,M):
    bragg_q = 1.
    bragg_exp = exp(-1j*bragg_q*arange(M))
    # 415 is edge,
    # 600-700 is second band
    edge = evecs[:,415][::2]
    temp = 0.
    for i in range(600,700):
        band = evecs[:,i][::2]
        band = band.reshape(N,M)
        band *= bragg_exp
        band = band.flatten()

        overlap = (edge * band.conj()).sum()
        print(overlap)
        temp += abs(overlap)**2

    print("----")
    temp2 = 0.
    for i in range(1000,1100):
        band = evecs[:,i][::2]
        band = band.reshape(N,M)
        band *= bragg_exp
        band = band.flatten()

        overlap = (edge * band.conj()).sum()
        print(overlap)
        temp2 += abs(overlap)**2

    print('---')
    print(temp,temp2)

