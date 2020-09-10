/*! \file
 *
 * \brief Routines to handle the core work of updating the N matrix of the QMC 
 * simulation. 
 *
 * These functions make use of fast matrix updates, which are possible due to 
 * the rank-1 updates (i.e. simple insert of a row and column to the matrix).
 * 
 * The N matrix itself is defined through a block like structure, where each 
 * block:
 * \f[
 * \mathbf{N^{-1}}_{lm} = e^{V_l} \delta_{lm} - \mathbf{G}_{0,lm} (e^{V_l} - 
 * \mathbf{I})
 * \f]
 * contains a 2x2 block that describes the spin structure, where \f$e^{V_l}\f$ 
 * is a diagonal matrix with the elements \f$(e^{\gamma s_l},e^{-\gamma 
 * s_l})\f$.
 *
 */

cd sign(cd l)
{
	//if(l<0.0){return -1.0;}
    //else{return 1.0;}
	return l/abs(l);
}
//------------------------------------------------------------------------------
/*! A convenience function for indexing the large matrices, which are stored as 
 * 1D arrays.
 */
int index(int l, int m, int Size)//Transforming matrix indices to vector indices
{return l*Size+m;}
//------------------------------------------------------------------------------
/*! A convenience function to extract one 2x2 part of the total N matrix.
 */
void set_S_tilde(cd S_tilde[], cd N[], int j, int Size2)
//Setting S_tilde as submatrix j,j of the matrix N
{
    S_tilde[0]=N[index(2*j, 2*j, Size2*2)];
    S_tilde[1]=N[index(2*j, 2*j+1, Size2*2)];
    S_tilde[2]=N[index(2*j+1, 2*j, Size2*2)];
    S_tilde[3]=N[index(2*j+1, 2*j+1, Size2*2)];
}
//------------------------------------------------------------------------------
/*! A convenience function to invert a 2x2 matrix.
 */
void invert(cd M[], cd Det)
//Inverting the matrix M
{
    if(Det==0.){M[0]=M[1]=M[2]=M[3]=0.0;}
    else{
        cd dummy=M[0];
        M[0]=M[3]/Det;
        M[3]=dummy/Det;
        M[1]=-M[1]/Det;
        M[2]=-M[2]/Det;}
}
//------------------------------------------------------------------------------
/*! A convenience function for the determinant of a 2x2 matrix.
 */
cd det(cd M[])//Determinant of the matrix M
{return (M[0]*M[3]-M[1]*M[2]);}
//------------------------------------------------------------------------------
/*! Perform the removal update on the matrix N.
 *
 * This function uses the block matrix formula:
 * \f[
 * A^{-1} = P - QS^{-1}R
 * \f]
 * where
 * \f[
 * \left[ \begin{array}{cc}
 * A & B \\
 * C & D
 * \end{array} \right]
 *  =
 * \left[ \begin{array}{cc}
 * P & Q \\
 * R & S
 * \end{array} \right]^{-1}
 * \f]
 *
 * For optimization, the matrix \f$S^{-1}\f$ is supplied by the previous code, 
 * which has already used it to determine the removal weight.
 */
void mat_red(cd N[], double taus[], int spins[], cd S[], int j, int n, int Size2)
//Removal of column and row j from N
{
    cd *Q = new cd [4*(n-1)];
    cd *R = new cd [4*(n-1)];
    cd *P = new cd [4*(n-1)*(n-1)];

    int k=0;
    int l=0;

    for (int i=0; i<n; i++){
        if(i!=j){
            R[index(0,2*k,2*(n-1))]=N[index(2*j,2*i,Size2*2)];
            R[index(0,2*k+1,2*(n-1))]=N[index(2*j,2*i+1,Size2*2)];
            R[index(1,2*k,2*(n-1))]=N[index(2*j+1,2*i,Size2*2)];
            R[index(1,2*k+1,2*(n-1))]=N[index(2*j+1,2*i+1,Size2*2)];
            Q[index(2*k,0,2)]=N[index(2*i,2*j,Size2*2)];
            Q[index(2*k,1,2)]=N[index(2*i,2*j+1,Size2*2)];
            Q[index(2*k+1,0,2)]=N[index(2*i+1,2*j,Size2*2)];
            Q[index(2*k+1,1,2)]=N[index(2*i+1,2*j+1,Size2*2)];
            taus[k]=taus[i];
            spins[k]=spins[i];
            for (int m=0; m<n; m++){
                if(m!=j){
                P[index(2*k,2*l,2*(n-1))]=N[index(2*i,2*m,Size2*2)];
                P[index(2*k,2*l+1,2*(n-1))]=N[index(2*i,2*m+1,Size2*2)];
                P[index(2*k+1,2*l,2*(n-1))]=N[index(2*i+1,2*m,Size2*2)];
                P[index(2*k+1,2*l+1,2*(n-1))]=N[index(2*i+1,2*m+1,Size2*2)];
                l++;}}
            l=0;
            k++;}}

    for (int i=0; i<2*(n-1); i++){
        for (int m=0; m<2*(n-1); m++){
            N[index(i,m,Size2*2)]=P[index(i,m,2*(n-1))]-(Q[index(i,0,2)]*S[0]*R[index(0,m,2*(n-1))]+Q[index(i,0,2)]*S[1]*R[index(1,m,2*(n-1))]+Q[index(i,1,2)]*S[2]*R[index(0,m,2*(n-1))]+Q[index(i,1,2)]*S[3]*R[index(1,m,2*(n-1))]);}}

    delete [] Q;
    delete [] R;
    delete [] P;
}
//------------------------------------------------------------------------------
/*! Perform the actual matrix insertion on the matrix N.
 *
 * This function takes as arguments the new elements to be inserted in the 
 * matrix. These come from interpolation calls of decide_insertion().
 *
 * This function makes use of block matrix inversion in order to speed up the 
 * insertion process.
 */
void mat_inc(cd N[], double taus[], int spins[], cd Q[], cd R[], cd S[], cd S_tilde[], double Tau, int Spin, int n, int Size2)
//Insertion of time Tau and spin Spin to the configuration
{
    cd *Q_tilde=new cd[4*n];
    cd *R_tilde=new cd[4*n];

    for(int l=0; l<2*n; l++){
        Q_tilde[index(l,0,2)]=Q_tilde[index(l,1,2)]=0.0;
        R_tilde[index(0,l,2*n)]=R_tilde[index(1,l,2*n)]=0.0;
        for(int j=0; j<2*n; j++){
            Q_tilde[index(l,0,2)]-=N[index(l,j,Size2*2)]*(Q[index(j,0,2)]*S_tilde[0]+Q[index(j,1,2)]*S_tilde[2]);
            Q_tilde[index(l,1,2)]-=N[index(l,j,Size2*2)]*(Q[index(j,0,2)]*S_tilde[1]+Q[index(j,1,2)]*S_tilde[3]);
            R_tilde[index(0,l,2*n)]-=(S_tilde[0]*R[index(0,j,2*n)]+S_tilde[1]*R[index(1,j,2*n)])*N[index(j,l,Size2*2)];
            R_tilde[index(1,l,2*n)]-=(S_tilde[2]*R[index(0,j,2*n)]+S_tilde[3]*R[index(1,j,2*n)])*N[index(j,l,Size2*2)];}}

    for (int i=0; i<2*n; i++){
        N[index(i,2*n,Size2*2)]=Q_tilde[index(i,0,2)];
        N[index(i,2*n+1,Size2*2)]=Q_tilde[index(i,1,2)];
        N[index(2*n,i,Size2*2)]=R_tilde[index(0,i,2*n)];
        N[index(2*n+1,i,Size2*2)]=R_tilde[index(1,i,2*n)];
        for (int j=0; j<2*n; j++){
            N[index(i,j,Size2*2)]+=(Q_tilde[index(i,0,2)]*S[0]*R_tilde[index(0,j,2*n)]+Q_tilde[index(i,0,2)]*S[1]*R_tilde[index(1,j,2*n)]+Q_tilde[index(i,1,2)]*S[2]*R_tilde[index(0,j,2*n)]+Q_tilde[index(i,1,2)]*S[3]*R_tilde[index(1,j,2*n)]);}}

    N[index(2*n,2*n,Size2*2)]=S_tilde[0];
    N[index(2*n,2*n+1,Size2*2)]=S_tilde[1];
    N[index(2*n+1,2*n,Size2*2)]=S_tilde[2];
    N[index(2*n+1,2*n+1,Size2*2)]=S_tilde[3];

    taus[n]=Tau;
    spins[n]=Spin;

    delete [] Q_tilde;
    delete [] R_tilde;
}
//------------------------------------------------------------------------------
/*! Determine the probability to remove a spin and act out the removal.
 *
 * This function randomly determines a spin to remove using random_removal() 
 * and then calculates the ratio between the configuration probabilities for 
 * the system before and after removal.
 *
 * This probably is simply the ratio of the determinant of the N matrices 
 * associated to each configure. Due to the structure of these matrices, the 
 * ratio may be calculated simply as the inverted determinant of the 2x2 block 
 * that will be removed.
 *
 * In the case that the removal is successful, the inverted block is sent in 
 * the call to mat_red(), which does the actual work to remove the spin.
 */
int decide_removal(cd N[], double taus[], int spins[], double K, double Gamma[], int n, int Size2, long seed[], cd & Sign)
//Decide the removal of a spin
{
    int i=0;
    int j = random_removal(seed, n);
    cd *S_tilde = new cd[4];
    set_S_tilde(S_tilde, N, j, Size2);
    cd Det=det(S_tilde);
    cd p = (1.0*n)/K*Det;
	// Danny added
	//if(n == 1)
	//	p *= 2;
    if (acceptance(seed, min(abs(real(p)), 1.0))){
        Sign=sign(p);
        invert(S_tilde, Det);
        mat_red(N, taus, spins, S_tilde, j, n, Size2);
        i=1;}
    delete [] S_tilde;
    return i;
}
//------------------------------------------------------------------------------
/*! Determine the probability to insert a new spni and act out the insertion.
 *
 * This function randomly chooses a new time to insert a random spin and then 
 * calculates the ratio between the configuration probabilities for the system 
 * before and after insertion.
 *
 * The ratio is simply the ratio of determinants of the N matrices. Due to the 
 * structure of the matrices, this comes down to the Schur complement that 
 * arises in the block matrix inversion.
 *
 * As many interpolation routines must be called in this function, these values 
 * are saved and, in the case that the insertion is successful, are passed to 
 * the mat_inc() function that performs the actual insertion.
 */
int decide_insertion(cd N[], double taus[], int spins[], double Gamma[], cd G0_up[], cd G0_down[], cd F0[], double tau[], int n, int Size, int Size2, long seed[], double K, cd & Sign)
//Decide insertion of a Spin
{
    int l=0;
    double tau_test = random_tau(seed);
    int spin_test = random_spin(seed);
    cd *Q=new cd[4*n];
    cd *R=new cd[4*n];
    cd *S=new cd[4];
    cd *S_tilde=new cd[4];
    for (int i=0; i<n; i++){
        R[index(0,2*i,2*n)]=-(1.0-Gamma[spins[i]])*single_linear(tau, G0_up, tau_test-taus[i], Size);
        R[index(0,2*i+1,2*n)]=-(1.0-Gamma[1-spins[i]])*single_linear(tau, F0, tau_test-taus[i], Size);
		// Danny: Modified for complex F0.
        R[index(1,2*i,2*n)]=-(1.0-Gamma[spins[i]])*conj(single_linear(tau, F0, tau_test-taus[i], Size));
        R[index(1,2*i+1,2*n)]=-(1.0-Gamma[1-spins[i]])*single_linear(tau, G0_down, tau_test-taus[i], Size);
        Q[index(2*i,0,2)]=-(1.0-Gamma[spin_test])*single_linear(tau, G0_up, taus[i]-tau_test, Size);
        Q[index(2*i,1,2)]=-(1.0-Gamma[1-spin_test])*single_linear(tau, F0, taus[i]-tau_test, Size);
		// Danny: Modified for complex F0.
        Q[index(2*i+1,0,2)]=-(1.0-Gamma[spin_test])*conj(single_linear(tau, F0, taus[i]-tau_test, Size));
        Q[index(2*i+1,1,2)]=-(1.0-Gamma[1-spin_test])*single_linear(tau, G0_down, taus[i]-tau_test, Size);}

    S[0] = 1.-G0_up[0]*(1.0-Gamma[spin_test]);
    S[1] = -F0[0]*(1.0-Gamma[1-spin_test]);
	// Danny: Modified for complex F0.
    S[2] = -conj(F0[0])*(1.0-Gamma[spin_test]);
    S[3] = 1.-G0_down[0]*(1.0-Gamma[1-spin_test]);


    for (int i=0; i<2*n; i++){
        for (int j=0; j<2*n; j++){
            S[0]-=R[index(0,i,2*n)]*N[index(i,j,Size2*2)]*Q[index(j,0,2)];
            S[1]-=R[index(0,i,2*n)]*N[index(i,j,Size2*2)]*Q[index(j,1,2)];
            S[2]-=R[index(1,i,2*n)]*N[index(i,j,Size2*2)]*Q[index(j,0,2)];
            S[3]-=R[index(1,i,2*n)]*N[index(i,j,Size2*2)]*Q[index(j,1,2)];}}

    for (int i=0; i<4; i++){S_tilde[i]=S[i];}

    cd Det=det(S);
    cd p = K/(n+1.0)*Det;
	// Danny added
	//if(n == 0)
	//{
	//	static int count = 0;
	//	count++;
	//	printf("n==0 for the %dth time\n",count);
	//	n /= 2;
	//}
    if (acceptance(seed, min(1.0, abs(real(p))))){
        Sign=sign(p);
        invert(S_tilde, Det);
        mat_inc(N, taus, spins, Q, R, S, S_tilde, tau_test, spin_test, n, Size2);
        l=1;}

    delete [] R;
    delete [] Q;
    delete [] S_tilde;
    delete [] S;
    return l;
}
