r'''This module implements a more explicit inversion for the case of a matrix 
that is defined on tri-diagonal blocks, as commonly appears in tight-binding 
models.  It also allows for periodic boundary conditions in these models, 
which have an additional block in the corners of the matrix. These matrices 
look something like:

.. math::

    \boldsymbol{G}^{-1} =
    \begin{bmatrix}
    \boldsymbol{A}_1 & \boldsymbol{B} & 0 & \ldots & 0 & 0 & \boldsymbol{B}*\\
    \boldsymbol{B}* & \boldsymbol{A}_2 & \boldsymbol{B} & \ldots & 0 & 0 & 0\\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & \ldots & \boldsymbol{B}* & \boldsymbol{A}_{M-1} & \boldsymbol{B} \\
    \boldsymbol{B} & 0 & 0 & \ldots & 0 & \boldsymbol{B}* & \boldsymbol{A}_M
    \end{bmatrix}
    
where the individual blocks are not tridiagonal:

.. math::
    \boldsymbol{A}_x = 
    \begin{bmatrix}
    G^{-1}_{xM} & -t_x & 0 & \ldots & 0 & 0 & -t_x^* \\
    -t_x^* & G^{-1}_{xM+1} & -t_x & \ldots & 0 & 0 & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
    0 & 0 & 0 & \ldots & -t_x^* & G^{-1}_{xM+1} & -t_x \\
    -t_x & 0 & 0 & \ldots & 0 & -t_x^* & G^{-1}_{xM} \\
    \end{bmatrix}.

Note that with spin-mixing included in the topological insulators system, 
B is not diagonal and cannot be simulataneously diagonalised with B*.

For more details of the derivation of these methods, please see Danny's notes.
'''
from __future__ import print_function
from std_imports import *

__all__ = ['BlockedInverseOpen','BlockedInversePeriodic','NormalInverse','CompareWithExact','BlockLU','ABtoFullBlock','PrintFullBlock']


def BlockedInverseOpen(A,B,full_matrix=False):
    r'''This function calculates the diagonal elements of the inverted matrix 
    using a blockwise inversion technique.

    The matrix is given by m lots of :math:`k\times k` blocks of ``A`` along the diagonal, and 
    the :math:`k\times k` block ``B`` on all of the first upper diagonal, and ``B*`` on all of the 
    first lower diagonal, making the matrix tridiagonal in total.
    
    **Returns:**
    If ``full_matrix`` is specified then a :math:`(mk)\times (mk)` matrix is returned, 
    otherwise a list of m :math:`k\times k` matrices is returned. 

    **Notes:**
    This inversion procedure scales as :math:`m^3 k` instead of the standard inversion scaling of :math:`(mk)^3`.
    '''

    from numpy import dot,conj,zeros
    from numpy.linalg import inv

    m = len(A)
    k = len(A[0])
    assert all(z.shape == (k,k) for z in A)
    assert B.shape == (k,k)

    # Firstly, transform the matrix so that the upper off-diagonal is only 
    # comprised of identity matrices.
    Binv = inv(B)
    Bdash = dot(Binv,conj(B))
    Adash = [dot(Binv,z) for z in A]

    # First recursion relationship (LU decomposition)
    Einv = [None] * m
    Einv[0] = inv(Adash[0])
    for i in range(1,m):
        Einv[i] = inv(Adash[i] - dot(Bdash,Einv[i-1]))

    # Second recursion relationship (backwards substitution)
    G = [None] * m
    G[-1] = Einv[-1]
    for i in reversed(range(m-1)):
        G[i] = Einv[i] + dot(Einv[i],dot(G[i+1],dot(Bdash,Einv[i])))

    if not full_matrix:
        # Undo the original Binv transformation.
        Gdash = [dot(z,Binv) for z in G]
        return Gdash
    else:
        Gfull = zeros((k*m,k*m),complex)
        # Define some convenience functions
        def getpart(mat,i,j):
            return mat[i*k:(i+1)*k, j*k:(j+1)*k]
        def setpart(mat,i,j,val):
            mat[i*k:(i+1)*k, j*k:(j+1)*k] = val

        # For each diagonal
        for i in reversed(range(m)):
            setpart(Gfull,i,i,G[i])

            # For every element above this diagonal
            temp = G[i]
            for j in reversed(range(i)):
                temp = -dot(Einv[j],temp)
                setpart(Gfull,j,i,temp)

            # For every element left of this diagonal
            temp = G[i]
            for j in reversed(range(i)):
                temp = -dot(temp,dot(Bdash,Einv[j]))
                setpart(Gfull,i,j,temp)

        # Undo the effect of the original Binv transformation.
        for i in xrange(m):
            for j in xrange(m):
                block = getpart(Gfull,i,j)
                block = dot(block,Binv)
                setpart(Gfull,i,j,block)

        return Gfull


def BlockedInversePeriodic(A,B,full_matrix=False):
    '''This function is identical to :func:`BlockedInverseOpen` but with 
    additional ``B`` and ``B*`` terms in the corners of the matrix.
    '''

    from numpy import dot,conj,identity,zeros
    from numpy.linalg import inv

    m = len(A)
    k = len(A[0])
    assert all(z.shape == (k,k) for z in A)
    assert B.shape == (k,k)

    # Firstly, transform the matrix so that the upper off-diagonal is only 
    # comprised of identity matrices.
    Binv = inv(B)
    Bdash = dot(Binv,conj(B))
    Bdashinv = Bdash.conj()
    Adash = [dot(Binv,z) for z in A]

    # First do all of the diagonal Us which are identical to the open BCs case.
    E = [None] * m
    Einv = [None] * m
    E[0] = (Adash[0])
    Einv[0] = inv(E[0])
    for i in range(1,m-1):
        E[i] = Adash[i] - dot(Bdash,Einv[i-1])
        Einv[i] = inv(E[i])

    # Wait for other components before working out E[-1]

    # Calculate the variables that are used in the recursion relations.
    U1i = [None]*m
    U1i[0] = -Einv[0]
    for i in range(1,m-1):
        U1i[i] = dot(U1i[i-1],-Einv[i])
    Ui1 = [None]*m
    Ui1[0] = -dot(Bdash,Einv[0])
    for i in range(1,m-1):
        Ui1[i] = dot(-Bdash,dot(Einv[i],Ui1[i-1]))
    M = [None]*m
    M[-1] = 0.
    M[-2] = U1i[-2]
    for i in reversed(range(m-2)):
        M[i] = -dot(M[i+1],dot(Bdash,Einv[i])) + U1i[i]
    MR = [None]*m
    MR[-1] = 0.
    MR[-2] = dot(Bdashinv,Ui1[-2])
    for i in reversed(range(m-2)):
        MR[i] = -dot(Einv[i],MR[i+1]) + dot(Bdashinv,Ui1[i])

    # Now we can calculate E[-1]
    E[-1] = Adash[-1] - dot(Bdash,Einv[-2]) + dot(M[0],Bdash) + U1i[-2] + dot(Ui1[-2],Bdash)
    Einv[-1] = inv(E[-1])

    # Fill in those elements that required Einv[-1]
    U1i[-1] = dot(U1i[-2],-Einv[-1])
    Ui1[-1] = dot(-Bdash,dot(Einv[-1],Ui1[-2]))

    UiN = [None]*m
    UiN[-1] = -Einv[-1]
    for i in reversed(range(0,m-1)):
        UiN[i] = dot(-Einv[i],UiN[i+1])
    UNi = [None]*m
    UNi[-1] = -dot(Bdash,Einv[-1])
    for i in reversed(range(0,m-1)):
        UNi[i] = dot(UNi[i+1],dot(Bdash,-Einv[i]))

    # We can finally apply the reverse recursion relation.
    Gii = [None]*m
    Gii[-1] = Einv[-1]
    for i in reversed(range(m-1)):
        Gii[i] = dot(Einv[i], dot(Gii[i+1],dot(Bdash,Einv[i])) + identity(k) + dot(UiN[i+1] - dot(MR[i+1],dot(Bdash,Einv[-1])), U1i[i])) + dot(dot(Bdashinv,Ui1[i]),dot(Bdash,dot(Einv[-1],M[i])) - UNi[i])
        #Gii[i] = Einv[i] + dot(Einv[i],dot(Gii[i+1],dot(Bdash,Einv[i]))) \
        #        - dot(Einv[i],dot(MR[i+1],dot(Bdash,dot(Einv[-1],U1i[i])))) \
        #        - dot(UiN[i],U1i[i]) \
        #        - dot(Bdashinv,dot(Ui1[i],UNi[i])) \
        #        + dot(Bdashinv,dot(Ui1[i],dot(Bdash,dot(Einv[-1],M[i]))))


    # Below are some tests that were used to identify problems with the 
    # recursion relation.
    #
    ## Test the left recursion
    #Gleft = dot(-Gii[-1],dot(Bdash,Einv[-2])) \
    #        - dot(MR[-1],dot(Bdash,dot(Einv[-1],U1i[-2]))) \
    #        - dot(UiN[-1],U1i[-2])
    #print('left is',Gleft)
    #Gleftleft = dot(-Gleft,dot(Bdash,Einv[-3])) \
    #        - dot(MR[-1],dot(Bdash,dot(Einv[-1],U1i[-3]))) \
    #        - dot(UiN[-1],U1i[-3])
    #print('leftleft is',Gleftleft)

    #Gup = -dot(Einv[-2],Gii[-1]) \
    #        + dot(Bdashinv,dot(Ui1[-2],dot(Bdash,dot(Einv[-1],M[-1])))) \
    #        + dot(Bdashinv,dot(Ui1[-2],dot(Bdash,Einv[-1])))
    #print("up is",Gup)
    #Gupup = -dot(Einv[-3],Gup) + dot(Bdashinv,dot(Ui1[-3],dot(Bdash,Einv[-1])))
    #print("upup is",Gupup)

    ##import pdb;pdb.set_trace()
    #Gupleft = dot(-Gup,dot(Bdash,Einv[-2])) \
    #        + Einv[-2] \
    #        + dot(MR[-2],dot(Bdash,dot(Einv[-1],U1i[-2]))) \
    #        - dot(UiN[-2],U1i[-2])
    #print("Gupleft is",Gupleft)
    #Gleftup = -dot(Einv[-2],Gleft) \
    #        + Einv[-2] \
    #        + dot(Bdashinv,dot(Ui1[-2],dot(Bdash,dot(Einv[-1],M[-2])))) \
    #        - dot(Bdashinv,dot(Ui1[-2],UNi[-2]))
    #print("Gleftup is",Gleftup)

    if not full_matrix:
        # Undo the effects of the original Binv transformation.
        Gii = [dot(z,Binv) for z in Gii]
        return Gii
    else:
        Gfull = zeros((k*m,k*m),complex)
        # Define some convenience functions.
        def getpart(mat,i,j):
            return mat[i*k:(i+1)*k, j*k:(j+1)*k]
        def setpart(mat,i,j,val):
            mat[i*k:(i+1)*k, j*k:(j+1)*k] = val

        # For each diagonal
        for i in reversed(range(m)):
            setpart(Gfull,i,i,Gii[i])

            # For every element above this diagonal
            temp = Gii[i]
            for j in reversed(range(i)):
                temp = -dot(Einv[j],temp)
                temp += dot(Bdashinv,dot(Ui1[j],
                    (dot(Bdash,dot(Einv[-1],M[i])) - UNi[i])))
                setpart(Gfull,j,i,temp)

            # For every element left of this diagonal
            temp = Gii[i]
            for j in reversed(range(i)):
                temp = -dot(temp,dot(Bdash,Einv[j]))
                temp += dot(   dot(MR[i],dot(Bdash,Einv[-1]))  - UiN[i]  ,U1i[j])
                setpart(Gfull,i,j,temp)

        # Undo the effect of the original Binv transformation.
        for i in xrange(m):
            for j in xrange(m):
                block = getpart(Gfull,i,j)
                block = dot(block,Binv)
                setpart(Gfull,i,j,block)

        return Gfull



def NormalInverse(A,B,periodic=False,B_inv=False):
    '''This function is a comparison function, that creates the complete matrix 
    used in :func:`BlockedInverseOpen` or :func:`BlockedInversePeriodic` 
    depending on the given value for ``peroidic`` from the given ``A`` and 
    ``B`` and then inverts this using the standard numpy inversion routines. It also returns the uninverted matrix.

    The ``B_inv`` parameter is for testing purposes and includes an additional transformation of :math:`B^{-1}`.

    **Returns:** ``mat,matinv``
    
        ``mat`` (matrix)
            The uninverted matrix.
        ``matinv`` (matrix)
            The inverted matrix.'''

    from numpy import conj,zeros,dot
    from numpy.linalg import inv

    m = len(A)
    k = len(A[0])
    assert all(z.shape == (k,k) for z in A)
    assert B.shape == (k,k)

    def setpart(mat,i,j,val):
        mat[i*k:(i+1)*k, j*k:(j+1)*k] = val

    Ginv = zeros((k*m,k*m),complex)
    for i in xrange(m):
        setpart(Ginv,i,i,A[i])
        if i > 0:
            setpart(Ginv,i,i-1,conj(B))
            setpart(Ginv,i-1,i,B)

    if periodic:
        setpart(Ginv,m-1,0,B)
        setpart(Ginv,0,m-1,conj(B))

    G = inv(Ginv)
    if B_inv:
        Binvmat = zeros((k*m,k*m),complex)
        for i in range(m):
            setpart(Binvmat,i,i,B)
        G = dot(G,Binvmat)

    return Ginv,G

def CompareWithExact(m,k,periodic=False):
    '''This function compares the exact inversion solution with the blockwise 
    calculation. It will randomly generate complex ``A`` and ``B`` matrices, 
    such that the blocks are of size ``k`` x ``k`` and there are a total of ``m`` 
    blocks on the diagonal. The code is then run for one of the above functions 
    depending on the value of ``periodic``. The calculations are also timed.''' 

    from numpy.random import rand
    from time import time

    A = [rand(k,k) + 1j*rand(k,k) for i in range(m)]
    B = rand(k,k) + 1j*rand(k,k)

    if periodic:
        start = time()
        Gdiag = BlockedInversePeriodic(A,B,False)
        Gdiag_time = time() - start
        start = time()
        Gfull = BlockedInversePeriodic(A,B,True)
        Gfull_time = time() - start
    else:
        start = time()
        Gdiag = BlockedInverseOpen(A,B,False)
        Gdiag_time = time() - start
        start = time()
        Gfull = BlockedInverseOpen(A,B,True)
        Gfull_time = time() - start

    start = time()
    Ginv,Gexact = NormalInverse(A,B,periodic)
    Gexact_time = time() - start

    Gexactdiag = [Gexact[i*k:(i+1)*k,i*k:(i+1)*k] for i in range(m)]

    Gdiff = abs(Gexact - Gfull).max()
    Gdiagdiff = max(abs(Gdiag[i] - Gexactdiag[i]).max() for i in range(m))

    print("Gdiag difference is",Gdiagdiff)
    print("Gfull difference is",Gdiff)
    print("Exact inversion time:",Gexact_time)
    print("Full inversion time: ",Gfull_time)
    print("Diag inversion time: ",Gdiag_time)



# Below are some utility functions for performing the LU decomposition in 
# blocks programmatically.
def BlockLU(mat):
    from numpy import identity, dot,zeros
    from numpy.linalg import inv
    import copy

    N = len(mat)
    k = len(mat[0][0])

    U = copy.deepcopy(mat)
    L = [[zeros((k,k)) for i in range(N)] for j in range(N)]
    for i in range(N):
        L[i][i] = identity(k)

    for i in range(1,N):
        # last row
        for j in range(i):
            L[-1][j] = L[-1][j] - dot(U[-1][i-1],dot(inv(U[i-1][i-1]),L[i-1][j]))
        U[-1][i] = U[-1][i] - dot(U[-1][i-1],inv(U[i-1][i-1]))
        U[-1][i-1] = zeros((k,k))

        if i < N-1:
            # normal part
            for j in range(i):
                L[i][j] = L[i][j] - dot(U[i][i-1],dot(inv(U[i-1][i-1]),L[i-1][j]))

            U[i][i] = U[i][i] - dot(U[i][i-1],inv(U[i-1][i-1]))
            U[i][i-1] = zeros((k,k))

    return L,U


def ABtoFullBlock(A,B,periodic=False):
    from numpy import identity, dot,zeros
    from numpy.linalg import inv
    Binv = inv(B)
    Bdash = dot(inv(B),B)

    N = len(A)
    k = len(A[0])

    mat = [[zeros((k,k)) for i in range(N)] for i in range(N)]

    for i in range(N):
        mat[i][i] = dot(Binv,A[i])
        if i > 0:
            mat[i-1][i] = Bdash
            mat[i][i-1] = identity(k)
        elif periodic:
            mat[-1][0] = identity(k)
            mat[0][-1] = Bdash

    return mat

def PrintFullBlock(mat):
    from numpy import array
    N = len(mat)
    k = len(mat[0][0])
    print(array(mat).transpose(0,2,1,3).reshape(N*k,N*k))
