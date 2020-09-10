from __future__ import print_function
from std_imports import *

class ED(object):
    def __init__(self,num_l,U,mu,beta):
        self.num_l = num_l
        self.V = None
        self.eps = None
        self.U = U
        self.mu = mu
        self.beta = beta

    def DefineSpace(self):
        import itertools
        # Create a list of all combinations of [False,True,...] of length max_l
        bath = [z for z in itertools.product(*([[False,True]]*self.num_l))]
        # Create a list with the four combinations of up spin and down spin 
        impurity = [z for z in itertools.product([False,True], [False,True])]

        # Initialise the state dictionary to be a set of lists, separated by their 
        # total spin
        states = {}
        for S in irange(-self.num_l-1,self.num_l+1):
            for Q in irange(0,self.num_l*2 + 2):
                states[Q,S] = {}

        # Loop over all combinations of each up spin of the bath, down spin of the 
        # bath and each impurity combination
        for item in itertools.product(bath,bath,impurity):
            indiv = item[0] + item[1] + item[2]
            indiv = tuple(indiv)
            # Note, one up is S=1, one down is S=-1.
            Q = sum(indiv)
            S = sum(item[0]) - sum(item[1]) + item[2][0] - item[2][1]

            # The state itself does into the key.
            # We assign an index and a convenient numpy array in the value.
            states[Q,S][indiv] = [len(states[Q,S]), array(indiv,int)]

        # Remove any that didn't fit.
        for Q,S in states.keys():
            if len(states[Q,S]) == 0:
                del states[Q,S]

        self.states = states
        return states

    def Hamiltonian(self):
        # The Hamiltonian is given by
        # \sum_{\sigma} of
        # -\mu_\sigma \hat{n}^c_\sigma + U \hat{n}^c_\up \hat{n}^c_\down
        # + \sum_l
        # + \eps_{l\sigma} \hat{n}^l_{\sigma}
        # + (V_{l\sigma} \hat{c}^\dagger_{\sigma} \hat{a}_{l\sigma} + h.c.)
        #
        # l is the 'orbital' of the bath and runs l=0,...,
        # n^c is the number operator for the impurity and n^l is the number operator 
        # for the orbital l of the bath.
        # \hat{c} is the annihilation operator for the impurity and \hat{a}_l is the 
        # annihilation operator for the bath orbital l.
        # h.c. means Hermitian conjugate.
        
        from numpy import zeros,conj
        # zeros creates a matrix filled with zeros

        assert self.V is not None and self.eps is not None
        assert len(self.V) == self.num_l and len(self.eps) == self.num_l
        #assert type(self.V) == list and type(self.eps) == list
        self.V = list(self.V)
        self.eps = list(self.eps)

        if not hasattr(self,'states'):
            self.DefineSpace()

        H = {}

        # Speed up:
        eps_mult = array(self.eps + self.eps + [-self.mu,-self.mu])

        # Loop over the total number of fermions (Q) and the total spin of the 
        # fermions (S).
        states = self.states
        num_l = self.num_l
        logger.debug("Generate Hamiltonian...")
        for Q,S in states:
            #logger.debug('Q = {0}, S = {1}'.format(Q,S))
            num = len(states[Q,S])
            H[Q,S] = zeros((num,num))

            for state_i in states[Q,S]:
                i,i_arr = states[Q,S][state_i]
                # Diagonal elements
                H[Q,S][i,i] += self.U * state_i[-2] * state_i[-1]
                # Chemical Potential and orbital energy
                H[Q,S][i,i] += (eps_mult * i_arr).sum()

                #import pdb
                #if (Q,S) == (6,2):
                #    pdb.set_trace()
                # Off-diagonal elements
                for l in xrange(num_l):
                    # Spin-up, imp -> orb
                    j_arr = i_arr.copy()
                    j_arr[-2] -= 1
                    j_arr[l] += 1
                    j = states[Q,S].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        sign = 1 - 2*(j_arr[:l].sum() % 2)
                        H[Q,S][i,j] += sign * self.V[l]
                        H[Q,S][j,i] += sign * conj(self.V[l])

                    ## Spin-up, orb -> imp
                    #j_arr = i_arr.copy()
                    #j_arr[-2] += 1
                    #j_arr[l] -= 1
                    #j = states[Q,S].get(tuple(j_arr),None)
                    #if j is not None:
                    #    j = j[0]
                    #    sign = 1 - 2*(j_arr[:l].sum() % 2)
                    #    H[Q,S][i,j] += sign * conj(self.V[l])

                    # Spin-down, imp -> orb
                    j_arr = i_arr.copy()
                    j_arr[-1] -= 1
                    j_arr[num_l + l] += 1
                    j = states[Q,S].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        sign = 1 - 2*(j_arr[num_l:num_l+l].sum() % 2)
                        H[Q,S][i,j] += sign * self.V[l]
                        H[Q,S][j,i] += sign * conj(self.V[l])

                    ## Spin-down, orb -> imp
                    #j_arr = i_arr.copy()
                    #j_arr[-1] += 1
                    #j_arr[num_l + l] -= 1
                    #j = states[Q,S].get(tuple(j_arr),None)
                    #if j is not None:
                    #    j = j[0]
                    #    sign = 1 - 2*(j_arr[num_l:num_l+l].sum() % 2)
                    #    H[Q,S][i,j] += sign * conj(self.V[l])

        self.H = H
        return H

    def CalcGreens(self,omega):
        if not hasattr(self,'H'):
            self.Hamiltonian()

        from numpy.linalg import eig,eigh,qr
        from numpy import zeros,exp,dot
        H = self.H
        states = self.states
        beta = self.beta

        if not hasattr(self,'evecs') or not hasattr(self,'evals'):
            evals = {}
            evecs = {}
            logger.debug("Eigendecomposition...")
            for Q,S in H:
                evals[Q,S],evecs[Q,S] = eig(H[Q,S])

                # Make sure eigenvectors are orthogonal
                q,r = qr(evecs[Q,S])
                evecs[Q,S] = q

            # Rescale the eigenvalues
            min_eval = min(evals[Q,S].min() for Q,S in H)
            for Q,S in H:
                evals[Q,S] -= min_eval

            logger.debug("Computing matrix elements and Green's functions...")
            # This is \hat{c}_\uparrow. It is the matrix element between
            # <Sz,i| c | Sz+1,j>
            c_matelm = {}

            # Figure out the list that will be non-zero.
            Qmax = max(Q for Q,S in H)
            Smax = max(S for Q,S in H)
            QSlist = [(Q,S) for Q,S in H]
            QSlist = [(Q,S) for Q,S in QSlist if (Q+1,S+1) in QSlist]
            QSlist = [(Q,S) for Q,S in QSlist if Q != Qmax or S != Smax]

            for Q,S in QSlist:
                c_matelm[Q,S] = zeros((len(H[Q,S]),len(H[Q+1,S+1])))
                for state_i,(i,i_arr) in states[Q,S].iteritems():
                    j_arr = i_arr.copy()
                    j_arr[-2] += 1

                    j = states[Q+1,S+1].get(tuple(j_arr),None)
                    if j is not None:
                        # Take into account the swapping procedure
                        c_matelm[Q,S][i,j[0]] = 1.

            logger.debug("done c matelm")

            #n_opp_matelm = {}
            #for Q,S in QSlist:
            #    n_opp_matelm[Q,S] = array([state[-1] for state in states[Q+1,S+1]])
            #logger.debug("done nc matelm")

            # Calculate the Green's function from the eigenvalues/-vectors
            Z = sum(sum(exp(-beta*evals[Q,S])) for Q,S in H)
            logger.debug("done Z")

            # Only calculating the up Green's function (because of symmetry).
            #dot = lambda x,y: y
            c_avg = dict( [(Q,S), dot((evecs[Q,S].T).conj(),dot(c_matelm[Q,S],evecs[Q+1,S+1]))] for Q,S in QSlist)
            logger.debug("done c_avg")
            #deltaE = dict( [(Q,S),evals[Q,S][:,None] - evals[Q+1,S+1][None,:]] for Q,S in QSlist)
            deltaE = dict( [(Q,S),evals[Q+1,S+1][None,:] - evals[Q,S][:,None]] for Q,S in QSlist)
            logger.debug("done deltaE")

        self.evecs = evecs
        self.evals = evals
        self.c_avg = c_avg
        self.deltaE = deltaE
        self.Z = Z

        greens_diag = -1/self.Z * array([ \
            sum(
                ( self.c_avg[Q,S]*self.c_avg[Q,S].conj() \
                  * (exp(-beta*self.evals[Q,S])[:,None] + 
                      exp(-beta*self.evals[Q+1,S+1])[None,:]) \
                  / (self.deltaE[Q,S] - omega[n]) \
                ).sum() \
            for Q,S in QSlist) \
            for n in range(len(omega))])
        logger.debug("done Greens")

        self.greens = greens_diag
        return greens_diag


######################################################################
##--- Spin-mixing additions                                      ---##
######################################################################

class ED_SM(object):
    def __init__(self,num_l,U,mu,beta):
        self.num_l = num_l
        self.Vup = None
        self.Vdown = None
        self.Wup = None
        self.Wdown = None
        self.eps = None
        self.U = U
        self.mu = mu
        self.beta = beta

    def DefineSpace(self):
        import itertools
        # Create a list of all combinations of [False,True,...] of length max_l
        bath = [z for z in itertools.product(*([[False,True]]*self.num_l))]
        # Create a list with the four combinations of up spin and down spin 
        impurity = [z for z in itertools.product([False,True], [False,True])]

        # Initialise the state dictionary to be a set of lists, separated by their 
        # total spin
        states = {}
        for Q in irange(0,self.num_l*2 + 2):
            states[Q] = {}

        # Loop over all combinations of each up spin of the bath, down spin of the 
        # bath and each impurity combination
        for item in itertools.product(bath,bath,impurity):
            indiv = item[0] + item[1] + item[2]
            indiv = tuple(indiv)
            # Note, one up is S=1, one down is S=-1.
            Q = sum(indiv)
            #S = sum(item[0]) - sum(item[1]) + item[2][0] - item[2][1]

            # The state itself does into the key.
            # We assign an index and a convenient numpy array in the value.
            states[Q][indiv] = [len(states[Q]), array(indiv,int)]

        # Remove any that didn't fit.
        for Q in states.keys():
            if len(states[Q]) == 0:
                del states[Q]

        self.states = states
        return states

    def Hamiltonian(self):
        # FIXME: NEED TO UPDATE THIS.
        # The Hamiltonian is given by
        # \sum_{\sigma} of
        # -\mu_\sigma \hat{n}^c_\sigma + U \hat{n}^c_\up \hat{n}^c_\down
        # + \sum_l
        # + \eps_{l\sigma} \hat{n}^l_{\sigma}
        # + (V_{l\sigma} \hat{c}^\dagger_{\sigma} \hat{a}_{l\sigma} + h.c.)
        #
        # l is the 'orbital' of the bath and runs l=0,...,
        # n^c is the number operator for the impurity and n^l is the number operator 
        # for the orbital l of the bath.
        # \hat{c} is the annihilation operator for the impurity and \hat{a}_l is the 
        # annihilation operator for the bath orbital l.
        # h.c. means Hermitian conjugate.
        
        from numpy import zeros,conj
        # zeros creates a matrix filled with zeros

        for z in [self.Vup,self.Vdown,self.Wup,self.Wdown,self.eps]:
            assert z is not None
            assert len(z) == self.num_l
        self.Vup = list(self.Vup)
        self.Vdown = list(self.Vdown)
        self.Wup = list(self.Wup)
        self.Wdown = list(self.Wdown)
        self.eps = list(self.eps)

        if not hasattr(self,'states'):
            self.DefineSpace()

        H = {}

        # Speed up:
        eps_mult = array(self.eps + self.eps + [-self.mu,-self.mu])

        # Loop over the total number of fermions (Q) and the total spin of the 
        # fermions (S).
        states = self.states
        num_l = self.num_l
        logger.debug("Generate Hamiltonian...")
        for Q in states:
            #logger.debug('Q = {0}, S = {1}'.format(Q,S))
            num = len(states[Q])
            H[Q] = zeros((num,num),complex)

            for state_i in states[Q]:
                i,i_arr = states[Q][state_i]
                # Diagonal elements
                H[Q][i,i] += self.U * state_i[-2] * state_i[-1]
                # Chemical Potential and orbital energy
                H[Q][i,i] += (eps_mult * i_arr).sum()

                #import pdb
                #if (Q,S) == (6,2):
                #    pdb.set_trace()
                # Off-diagonal elements
                for l in xrange(num_l):
                    # Spin-up, imp -> orb
                    j_arr = i_arr.copy()
                    j_arr[-2] -= 1
                    j_arr[l] += 1
                    j = states[Q].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        sign = 1 - 2*(j_arr[:l].sum() % 2)
                        H[Q][i,j] += sign * self.Vup[l]
                        H[Q][j,i] += sign * conj(self.Vup[l])

                    # Spin-down, imp -> orb
                    j_arr = i_arr.copy()
                    j_arr[-1] -= 1
                    j_arr[num_l + l] += 1
                    j = states[Q].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        sign = 1 - 2*(j_arr[num_l:num_l+l].sum() % 2)
                        H[Q][i,j] += sign * self.Vdown[l]
                        H[Q][j,i] += sign * conj(self.Vdown[l])

                    # imp_up -> orb_down
                    j_arr = i_arr.copy()
                    j_arr[-2] -= 1
                    j_arr[num_l + l] += 1
                    j = states[Q].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        # Sign is all of l's and the other spin (if there is 
                        # one
                        sign = 1 - 2*((j_arr[:num_l+l].sum() + j_arr[-1]) % 2)
                        H[Q][i,j] += sign * self.Wup[l]
                        H[Q][j,i] += sign * conj(self.Wup[l])

                    # imp_down -> orb_up
                    j_arr = i_arr.copy()
                    j_arr[-1] -= 1
                    j_arr[l] += 1
                    j = states[Q].get(tuple(j_arr),None)
                    if j is not None:
                        j = j[0]
                        # Sign is all of the up ls that are not touched by the 
                        # a_l operator.
                        sign = 1 - 2*(j_arr[l+1:num_l].sum() % 2)
                        H[Q][i,j] += sign * self.Wdown[l]
                        H[Q][j,i] += sign * conj(self.Wdown[l])

        self.H = H
        return H

    def CalcGreens(self,omega):
        if not hasattr(self,'H'):
            self.Hamiltonian()

        from numpy.linalg import eig,eigh,qr
        from numpy import zeros,exp,dot
        H = self.H
        states = self.states
        beta = self.beta

        if not hasattr(self,'evecs') or not hasattr(self,'evals'):
            evals = {}
            evecs = {}
            logger.debug("Eigendecomposition...")
            for Q in H:
                evals[Q],evecs[Q] = eig(H[Q])

                ind = evals[Q].argsort()
                evals[Q].sort()
                evecs[Q] = evecs[Q][:,ind]

                # Make sure eigenvectors are orthogonal
                q,r = qr(evecs[Q])
                evecs[Q] = q


            # Rescale the eigenvalues
            min_eval = min(evals[Q].min() for Q in H)
            for Q in H:
                evals[Q] -= min_eval

            self.evecs = evecs
            self.evals = evals

            logger.debug("Computing matrix elements and Green's functions...")
            # This is \hat{c}_\uparrow. It is the matrix element between
            # <Sz,i| c | Sz+1,j>
            cup_matelm = {}
            cdown_matelm = {}

            # Figure out the list that will be non-zero.
            Qlist = range(max(H))

            for Q in Qlist:
                cup_matelm[Q] = zeros((len(H[Q]),len(H[Q+1])))
                cdown_matelm[Q] = zeros((len(H[Q]),len(H[Q+1])))

                for state_i,(i,i_arr) in states[Q].iteritems():
                    j_arr = i_arr.copy()
                    j_arr[-2] += 1
                    j = states[Q+1].get(tuple(j_arr),None)
                    if j is not None:
                        # Take into account the swapping procedure
                        cup_matelm[Q][i,j[0]] = 1.

                    j_arr = i_arr.copy()
                    j_arr[-1] += 1
                    j = states[Q+1].get(tuple(j_arr),None)
                    if j is not None:
                        # Take into account the swapping procedure
                        sign = 1 - 2*((j_arr[:self.num_l].sum() + j_arr[-2]) % 2)
                        cdown_matelm[Q][i,j[0]] = 1. * sign

            self.cup_matelm = cup_matelm
            self.cdown_matelm = cdown_matelm
            logger.debug("done c matelm")

            # Calculate the Green's function from the eigenvalues/-vectors
            Z = sum(sum(exp(-beta*evals[Q])) for Q in H)

            self.Z = Z
            logger.debug("done Z")

            #dot = lambda x,y: y
            cup_avg = dict( [Q, dot((evecs[Q].T).conj(),dot(cup_matelm[Q],evecs[Q+1]))] for Q in Qlist)
            cdown_avg = dict( [Q, dot((evecs[Q].T).conj(),dot(cdown_matelm[Q],evecs[Q+1]))] for Q in Qlist)

            self.cup_avg = cup_avg
            self.cdown_avg = cdown_avg
            logger.debug("done c_avg")

            deltaE = dict( [(Q),evals[Q+1][None,:] - evals[Q][:,None]] for Q in Qlist)

            self.deltaE = deltaE
            logger.debug("done deltaE")

        Z = self.Z
        evals = self.evals
        evecs = self.evecs
        deltaE = self.deltaE
        cup_avg = self.cup_avg
        cdown_avg = self.cdown_avg
        num_l = self.num_l

        # Figure out the list that will be non-zero.
        Qlist = range(max(H))

        greens_up = -1/Z * array([ \
            sum(
                ( cup_avg[Q]*cup_avg[Q].conj() \
                  * (exp(-beta*evals[Q])[:,None] + 
                      exp(-beta*evals[Q+1])[None,:]) \
                  / (deltaE[Q] - omega[n]) \
                ).sum() \
            for Q in Qlist) \
            for n in range(len(omega))])
        greens_down = -1/Z * array([ \
            sum(
                ( cdown_avg[Q]*cdown_avg[Q].conj() \
                  * (exp(-beta*evals[Q])[:,None] + 
                      exp(-beta*evals[Q+1])[None,:]) \
                  / (deltaE[Q] - omega[n]) \
                ).sum() \
            for Q in Qlist) \
            for n in range(len(omega))])
        greens_cross = -1/Z * array([ \
            sum(
                ( cup_avg[Q]*cdown_avg[Q].conj() \
                  * (exp(-beta*evals[Q])[:,None] + 
                      exp(-beta*evals[Q+1])[None,:]) \
                  / (deltaE[Q] - omega[n]) \
                ).sum() \
            for Q in Qlist) \
            for n in range(len(omega))])
        logger.debug("done Greens")

        # Also calculate the filling while we are here.
        Nfup_matelm = {}
        Nfdown_matelm = {}
        Nfcross_matelm = {}
        Nfdbl_matelm = {}
        for Q in H:
            # These are just diagonal matrices, stored as vectors.
            Nfup_matelm[Q] = zeros(len(states[Q]))
            Nfdown_matelm[Q] = zeros(len(states[Q]))
            Nfdbl_matelm[Q] = zeros(len(states[Q]))
            for i,i_arr in states[Q].itervalues():
                Nfup_matelm[Q][i] = i_arr[-2]
                Nfdown_matelm[Q][i] = i_arr[-1]
                Nfdbl_matelm[Q][i] = i_arr[-1]*i_arr[-2]
            # This is a full matrix.
            # Don't forget this is the c_\uparrow(\tau=0-) c^\dagger_\downarrow(\tau=0) matrix element, not c^\dagger_\uparrow c_\downarrow
            Nfcross_matelm[Q] = zeros((len(H[Q]),len(H[Q])))
            for state_i,(i,i_arr) in states[Q].iteritems():
                j_arr = i_arr.copy()
                j_arr[-2] -= 1
                j_arr[-1] += 1
                j = states[Q].get(tuple(j_arr),None)
                if j is not None:
                    # Take into account the swapping procedure
                    sign = 1 - 2*(j_arr[:num_l].sum() % 2)
                    Nfcross_matelm[Q][i,j[0]] = sign
        self.Nfcross_matelm = Nfcross_matelm

        from numpy import diag
        Nf_up = 1/Z * sum( \
                        (dot(abs(evecs[Q]).T**2, Nfup_matelm[Q]) \
                        * exp(-beta*evals[Q])).sum() \
                        for Q in H)
        Nf_down = 1/Z * sum( \
                        (dot(abs(evecs[Q]).T**2, Nfdown_matelm[Q]) \
                        * exp(-beta*evals[Q])).sum() \
                        for Q in H)
        Nf_cross = 1/Z * sum( \
                        (diag(dot(evecs[Q].conj().T, dot(Nfcross_matelm[Q],evecs[Q]))) \
                        * exp(-beta*evals[Q])).sum() \
                        for Q in H)
        Nf_dbl = 1/Z * sum( \
                        (dot(abs(evecs[Q]).T**2, Nfdbl_matelm[Q]) \
                        * exp(-beta*evals[Q])).sum() \
                        for Q in H)

        self.Nf = Nf_up.real,Nf_down.real,Nf_cross,Nf_dbl.real
        logger.debug("Nfup is {0}".format(self.Nf))
        #Nf_up = 1/Z * sum( \
        #                (diag(dot(evecs[Q].conj().T, dot(diag(Nfup_matelm[Q]),evecs[Q]))) \
        #                * exp(-beta*evals[Q])).sum() \
        #                for Q in H)
        #logger.debug("Different Nfup is {0}".format(Nf_up))

        self.greens = (greens_up,greens_down,greens_cross)
        return self.greens

    def ParamsFromWeiss(self,omega,weiss,start_params=None):
        from numpy import empty_like,array
        from numpy.linalg import inv

        logger.debug("Fitting Anderson parameters...")
        # First invert the weiss function.
        # Assuming here that omega is spaced symmetrically about zero.
        weiss_inv = empty((3,len(omega)),complex)
        for n in range(len(omega)):
            temp = array([[weiss[0][n], weiss[2][n]],
                          [weiss[2][-n-1].conj(), weiss[1][n]]])
            temp = inv(temp)
            weiss_inv[0][n] = temp[0,0]
            weiss_inv[1][n] = temp[1,1]
            weiss_inv[2][n] = temp[0,1]

        self.Vup,self.Vdown,self.Wup,self.Wdown,self.eps = WeissToParameters(omega,self.mu,weiss_inv,self.num_l,start_params=start_params)
        #self.Vup = self.Vdown
        #self.Wup = self.Wdown = [0] * self.num_l
        logger.debug("done")

def Weiss(omega,mu,args,invert=True,returninv=False):
    from numpy import conj,zeros,sqrt
    from numpy.linalg import inv

    if type(args) == ED_SM:
        Vup,Vdown,Wup,Wdown,eps = args.Vup,args.Vdown,args.Wup,args.Wdown,args.eps
    elif type(args) == tuple:
        Vup,Vdown,Wup,Wdown,eps = args
    else:
        raise TypeError("Unknown type for args: should be tuple of sets of parameters or a ED_SM object.")

    num_l = len(Vup)
    assert len(Vup) == len(Vdown) == len(Wup) == len(Wdown)

    precalc_upup = [abs(Vup[i])**2 + abs(Wup[i])**2 for i in range(num_l)]
    precalc_downdown = [abs(Vdown[i])**2 + abs(Wdown[i])**2 for i in range(num_l)]
    precalc_updown = [(Wup[i]*conj(Vdown[i]) + Vup[i]*conj(Wdown[i])) for i in range(num_l)]
    precalc_downup = [(Wdown[i]*conj(Vup[i]) + Vdown[i]*conj(Wup[i])) for i in range(num_l)]

    G = zeros((len(omega),2,2),complex)
    Ginv = zeros((len(omega),2,2),complex)
    for n in range(len(omega)):
        precalc_denom = [1./(omega[n] - eps[i]) for i in range(num_l)]

        G[n,0,0] = omega[n] + mu - sum(precalc_upup[i] * precalc_denom[i] for i in range(num_l))
        G[n,1,1] = omega[n] + mu - sum(precalc_downdown[i] * precalc_denom[i] for i in range(num_l))
        G[n,0,1] = - sum(precalc_updown[i] * precalc_denom[i] for i in range(num_l))
        G[n,1,0] = - sum(precalc_downup[i] * precalc_denom[i] for i in range(num_l))

        if invert:
            Ginv[n] = inv(G[n])

    if invert:
        G_up = Ginv[:,0,0]
        G_down = Ginv[:,1,1]
        G_cross = Ginv[:,0,1]
        G_cross2 = Ginv[:,1,0]
    else:
        G_up = G[:,0,0]
        G_down = G[:,1,1]
        G_cross = G[:,0,1]
        G_cross2 = G[:,1,0]

    if returninv:
        return (G_up,G_down,G_cross,G_cross2),G
    else:
        return G_up,G_down,G_cross
    #return G

def WeissToParameters(omega,mu,weiss_inv,num_l=3,start_params=None):
    weiss_inv = array(weiss_inv)

    def separgs(args):
        eps = args[:num_l]
        Vup = array(args[num_l:num_l*2])
        Vdown = args[num_l*2:num_l*3]
        Wup = args[num_l*3:num_l*4] + 1j*args[num_l*4:num_l*5]
        Wdown = args[num_l*5:num_l*6] + 1j*args[num_l*6:num_l*7]
        return Vup,Vdown,Wup,Wdown,eps

    def func(args):
        Vup,Vdown,Wup,Wdown,eps = separgs(args)
        newweiss_inv = Weiss(omega,mu,(Vup,Vdown,Wup,Wdown,eps),invert=False)

        temp = (array(newweiss_inv) - weiss_inv).flatten()

        from numpy import inf
        if (temp != temp).any() or (temp == inf).any():
            raise Exception("In normalfunc: Got inf and NaN here with params={0}".format(args))

        return r_[temp.real,temp.imag]
        #return (abs(temp)**2).sum()

    def derivfunc(args):
        Vup,Vdown,Wup,Wdown,eps = separgs(args)

        n,l = ix_(range(len(omega)),range(num_l))
        nothing = zeros((len(omega),num_l))

        denom = 1. / (omega[n] - eps[l])
        denom2 = 1. / (omega[n] - eps[l])**2
        # Assuming V is real here
        dGupdeps = -(abs(Vup[l])**2 + abs(Wup[l])**2) * denom2
        dGupdVup = -2*Vup[l] * denom
        dGupdVdown = nothing
        dGupdWrup = -2*Wup[l].real * denom
        dGupdWiup = -2*Wup[l].imag * denom
        dGupdWrdown = nothing
        dGupdWidown = nothing

        dGdowndeps = -(abs(Vdown[l])**2 + abs(Wdown[l])**2) * denom2
        dGdowndVup = nothing
        dGdowndVdown = -2*Vdown[l] * denom
        dGdowndWrup = nothing
        dGdowndWiup = nothing
        dGdowndWrdown = -2*Wdown[l].real * denom
        dGdowndWidown = -2*Wdown[l].imag * denom

        dGupcrossdeps = -(Wup[l]*conj(Vdown[l]) + Vup[l]*conj(Wdown[l])) * denom2
        dGupcrossdVup = -conj(Wdown[l]) * denom
        dGupcrossdVdown = -Wup[l] * denom
        dGupcrossdWrup = -Vdown[l] * denom
        dGupcrossdWiup = -1j*Vdown[l] * denom
        dGupcrossdWrdown = -Vup[l] * denom
        dGupcrossdWidown = +1j*Vup[l] * denom

        J = c_[ r_[dGupdeps,dGdowndeps,dGupcrossdeps],
                r_[dGupdVup,dGdowndVup,dGupcrossdVup],
                r_[dGupdVdown,dGdowndVdown,dGupcrossdVdown],
                r_[dGupdWrup,dGdowndWrup,dGupcrossdWrup],
                r_[dGupdWiup,dGdowndWiup,dGupcrossdWiup],
                r_[dGupdWrdown,dGdowndWrdown,dGupcrossdWrdown],
                r_[dGupdWidown,dGdowndWidown,dGupcrossdWidown]]
        J = r_[J.real, J.imag]

        from numpy import inf
        if (J != J).any() or (J == inf).any():
            raise Exception("In derivfunc: Got inf and NaN here with params={0}".format(args))

        return J

    if start_params == None:
        start_params = rand(num_l*7)
    else:
        # This should flatten!
        Vup,Vdown,Wup,Wdown,eps = array(start_params)
        start_params = r_[Vup.real,Vdown.real,Wup.real,Wup.imag,Wdown.real,Wdown.imag,eps.real]
        assert len(start_params) == num_l*7

    start_params = rand(num_l*7)

    from scipy.optimize import leastsq,fmin,fmin_bfgs,fmin_cg
    try:
        maxfev = num_l * 100
        params,covx,info,mesg,flag = leastsq(func,start_params,Dfun=derivfunc,ftol=1e-8,xtol=1e-8,full_output=True,maxfev=maxfev)
    except:
        logger.error("Exception in leastsq. Current params are {0}".format(start_params))
        raise
    print("Func eval was " + str(info['nfev']) + " and residual was " + str((abs(func(params))**2).max()))

    if not (0 <= flag <= 4):
        logger.warning("Didn't converge, with message: " + mesg)
        temp = (abs(func(params))**2).max()
        if temp > 1e-4:
            #raise Exception("Didn't converge and residual value was too big: " + str(temp))
            logger.error("Didn't converge and residual value was too big: " + str(temp))
        else:
            logger.info("But average residual value was small enough: " + str(temp))
    else:
        #logger.info('Required {0} func evals'.format(info['nfev']))
        pass
    #out = fmin(func,rand(num_l*7),ftol=1e-6,xtol=1e-3)
    #params = fmin_bfgs(func,rand(num_l*7),maxiter=200)
    #out = fmin_cg(func,rand(num_l*7))

    return separgs(params)

def EDSMGreens(omega,U,mu,beta,weiss,num_l=4,start_params=None,return_weiss=False):
    ed = ED_SM(num_l,U,mu,beta)
    ed.ParamsFromWeiss(omega,weiss,start_params=start_params)

    Wcheck = Weiss(omega,mu,ed)

    if False:
        from pylab import figure,plot
        from numpy import array
        #Wcheck = Weiss(omega,mu,ed)
        figure()
        plot(array(weiss).T.imag)
        plot(array(Wcheck).T.imag)
        figure()
        plot(array(weiss-Wcheck).T.imag)

        print('ed.Vup',ed.Vup)
        print('ed.Vdown',ed.Vdown)
        print('ed.Wup',ed.Wup)
        print('ed.Wdown',ed.Wdown)
        print('ed.eps',ed.eps)

    G = ed.CalcGreens(omega)

    params = (ed.Vup,ed.Vdown,ed.Wup,ed.Wdown,ed.eps)

    if return_weiss:
        return G,ed.Nf,params,Wcheck
    else:
        return G,ed.Nf,params

def DMFT(num_omega,U,mu,beta,num_l=4):
    from numpy import array,r_,zeros_like
    from numpy.linalg import inv

    import ImagConv
    omega = ImagConv.OmegaN(num_omega,beta)*1j
    omega = r_[-omega[::-1],omega]

    W = 1./(omega)
    W = array([W,W,0*W])
    params = None
    num_iters = 50

    for i in range(num_iters):
        G,Nf,params = EDSMGreens(omega,U,mu,beta,W,num_l,params)
        Wtest = Weiss(omega,mu,params)
        #figure()
        #plot(omega.imag,W[0].imag,marker='.')
        #plot(omega.imag,Wtest[0].imag,marker='.')
        print("W-Wtest",abs(W[0] - Wtest[0]).max())
        #figure()
        #plot(omega.imag,G[0].imag,marker='.')
        #import CT_AUX_SM
        #W_forCT = 1./(1./(W[0]) - U/2.)
        #W_forCT = [W_forCT,W_forCT,W[0]*0]
        #G_CT,GT,pert_order,dbl_occ,WT,G_orig,Gw = CT_AUX_SM.CT_AUX_SM(U,beta,W_forCT,1e5)
        #plot(omega.imag,G_CT[0].imag,marker='.')
        #print("Diff to CT",abs(G_CT[0] - G[0]).max())

        #SE = zeros(num_omega*2)
        #for n in range(num_omega*2):
        #    Gmat = zeros((2,2))
        #    Gmat[0,0] = G[0][n]
        #    Gmat[1,1] = G[1][n]
        #    Gmat[0,1] = G[2][n]
        #    Gmat[1,0] = G[2][-n-1].conj()

        #    Wmat = zeros((2,2))
        #    Wmat[0,0] = W[0][n]
        #    Wmat[1,1] = W[1][n]
        #    Wmat[0,1] = W[2][n]
        #    Wmat[1,0] = W[2][-n-1].conj()

        #    # Ignore off-diagonal SE
        #    temp = inv(Wmat) - inv(Gmat)
        #    SE[n] = temp[0,0]

        # Bethe eqn
        # Forcing para with no off diagonal SE
        Wold = W
        Wup = 1./(omega + mu - G[0])
        Wdown = 1./(omega + mu - G[1])
        W = [Wup,Wdown,0*Wup]

        import pylab
        #pylab.figure(1)
        #pylab.plot(omega.imag,W[0].imag)

        print("Nf are",Nf)
        #print("params are",params)
        print("Difference of W is",abs(W[0] - Wold[0]).max(),avg(W[0] - Wold[0]))


######################################################################
##--- Testing functions                                          ---##
######################################################################

from pylab import *
def BlockMatrix(H,plot_figures=True):
    H = H.copy()
    def swap(mat,i,j):
        if i == j:
            return
        temp = mat[i].copy()
        mat[i] = mat[j]
        mat[j] = temp
        #figure(); imshow(mat.copy(),interpolation='nearest'); colorbar()
        temp = mat[:,i].copy()
        mat[:,i] = mat[:,j]
        mat[:,j] = temp
        #figure(); imshow(mat.copy(),interpolation='nearest'); colorbar()

    swap_boundaries = [0]
    lowest_swap = 0
    rowsdone = 0
    for i in range(len(H)):
        did_a_swap = False
        for j in range(rowsdone,len(H)):
            if abs(H[i,j]) > 1e-8:
                swap(H,rowsdone,j)
                rowsdone += 1
                if i != j:
                    did_a_swap = True
                    lowest_swap = rowsdone-1
        if not did_a_swap and lowest_swap <= i:
            swap_boundaries += [i+1]

    if plot_figures:
        temp = H.copy()
        # Fill in the squares
        for i in range(len(swap_boundaries)-1):
            temp[swap_boundaries[i]:swap_boundaries[i+1],swap_boundaries[i]:swap_boundaries[i+1]] = 1.

        figure(); imshow(temp,interpolation='nearest'); colorbar()

    return H

def TestEDSM():
    from numpy.random import rand
    num_l = 3
    f = lambda: rand(num_l) + rand(num_l)*1j
    b = ED_SM(num_l,0.,1.0,10.); b.Vup=f(); b.Vdown=f(); b.Wup=f(); b.Wdown=f(); b.eps=f().real;
    #b = ED_SM(num_l,0.,1.0,10.); b.Vup=f(); b.Vdown=b.Vup; b.Wup=f(); b.Wdown=b.Wup; b.eps=f().real;
    #b = ED_SM(num_l,0.,1.0,10.); b.Vup=f(); b.Vdown=b.Vup; b.Wup=f()*0; b.Wdown=b.Wup; b.eps=f().real;
    #b = ED_SM(4,0.,1.0,10.); b.Vup = [1.,2.,3.,4.]; b.Vdown=[2.,3.,1.,0.1]; b.Wup = [2.,4.,1.,2.]; b.Wdown=[1,2,3,4]; b.eps = [2.,3.,4.,5.]; 
    #b = ED_SM(4,0.,1.0,10.); b.Vup = [1.,2.,3.,4.]; b.Vdown=b.Vup; b.Wup = [2.,4.,1.,2.]; b.Wdown=b.Wup; b.eps = [2.,3.,4.,5.]; 
    #b = ED_SM(3,0.,0.3,10.); b.Vup = [3.,5.,1.]; b.Vdown=b.Vup; b.Wup = [1.,2.1,4.]; b.Wdown=b.Wup; b.eps = [2.,3.,4.]; 
    #b = ED_SM(3,0.,0.3,10.); b.Vup = [3.,5.,1.]; b.Vdown=[1.,1.,1.]; b.Wup = [1.+1j,2.1+2j,4.]; b.Wdown=[2.+10j,0.,1.]; b.eps = [2.,3.,4.]; 
    #b = ED_SM(2,0.,0.3,10.); b.Vup = [3.,5.]; b.Vdown=b.Vup; b.Wup = [1.,2.1]; b.Wdown=b.Wup; b.eps = [2.,3.]; 
    #b = ED_SM(1,0.,1.0,10.); b.Vup = [2.]; b.Vdown=b.Vup; b.Wup = [3.]; b.Wdown=b.Wup; b.eps = [5]; 
    #b = ED_SM(1,0.,0.0,10.); b.Vup = [2.]; b.Vdown=b.Vup; b.Wup = [0.]; b.Wdown=b.Wup; b.eps = [2]; 
    #b = ED_SM(0,0.,0.0,10.); b.Vup = []; b.Vdown=b.Vup; b.Wup = []; b.Wdown=b.Wup; b.eps = []; 

    return b
