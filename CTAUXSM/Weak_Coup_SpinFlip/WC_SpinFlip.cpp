/*! \file
 *
 * \brief The main file of the solver including the class.
 *
 * This file contains the majority of the control of the solver. All of the 
 * details are contained in a class WC, which keeps track of all important 
 * details. In order to use this, one should call the functions with something 
 * similar to:
 * 
 * @code
 * int Size = 20000;
 * WC * ImpSol=new WC(g0_up, g0_down, f0, U, Size, num_omega*2,num_sweeps);
 * ImpSol->K = K;
 * ImpSol->NIgreen();
 * ImpSol->sampling_proc();
 * ImpSol->save_and_det_Greentau(G_up, G_down, F, 0,Gw_up,Gw_down,Fw);
 * delete ImpSol;
 * @endcode
 *
 * where the appropriate Matsubara input Weiss green's functions are 
 * specified in g0_up, g0_down and f0 and the output is written to files 
 * and into the arrays supplied by G_up, G_down, F, Gw_up, Gw_down and Fw.
 *
 * Note that this code assumes that the inverse temperature is beta=1. This 
 * parameter should therefore be scaled appropriately for the input Weiss 
 * Green's functions and interaction parameters.
 */
#include "PythonSplines.cpp"

//------------------------------------------------------------------------------
/*! A convenience function. Effectively returns a/|a|.
 */
double sign(int l)
{
    if(l<0){return -1.0;}
    else{return 1.0;}
}
//------------------------------------------------------------------------------

/*! \brief The wrapper class that implements the CT-AUX solver.
 *
 */
class WC
{
      public:
             WC(cd g_0_up_1[], cd g0_down_1[], cd f0_1[], double U, int Size, int Size1, int NoIt);
             ~WC();

             int Size; //!< Number of Matsubara Frequencies in the sampling green function
             int Size1; //!< Number of Matsubara Frequencies for DMFT iteration
             cd *g0_up; //!< Non-interacting green functions for DMFT iteration
             cd *g0_down;
             cd *f0;
             double *tau; //!< Imaginary times for the sampling process
             cd *G0_up;//!< Interpolated green functions for the sampling process
             cd *G0_down;
             cd *F0;
             double pi;
             int n;//!< dynamical perturbation order (Size of the Matrix N)
             double *taus;//!< dynamical imaginary times during the sampling
             int *spins;//!< dynamical Ising spins during the sampling
             cd *N;//!< dynamical matrix N. The first nxn elements are used of this 1D array to emulate a 2D matrix.
             long *seed;//!< seed for random number generator
             double U;
             double K;//!< real parameter K for Hubbard-Stratonovich decoupling
             double* Gamma;//!< Vector Gamma for the sampling
             double phi;
             double *omega;
             cd Sign;//!< Sign, just to check
             int Size2;
             int NoIt;
             int NoSteps;
             int L;//!< Number of Measurements for the green functions
			 cd* G_meas;
             // Added by Danny
             cd * M_meas[4]; //!< Accumulation M matrix for the QMC.
             double* tau_meas;
			 double pert_order; //!< Average perturbation order

			 /*! Fourier transform the imaginary time Weiss Green's functions 
			  * to Matsubara frequency.
			  * 
			  * After Fourier transformation, this function will also spline 
			  * interpolate the points to a finer grid in order to allow a 
			  * better sampling procedure.
			  *
			  * This function will also save the Green's function to 
			  * "nigreen.txt".
			  */
             void NIgreen();

			 /*! Generate the initial configuration for the solver.
			  *
			  * This configuration will always be chosen with a single up 
			  * auxillary spin. The two values for gamma are then determined 
			  * and the N matrix is then configured.
			  */
             void start_conf();

			 /*! The main part of the sampling procedure, which controls the 
			  * QMC loops.
			  *
			  * In this function, each loop of the sampling procedure is 
			  * handled. For each iteration, a random choice is made to either 
			  * add a new aux spin, or to remove an existing one. The success 
			  * of this is then determined from the functions 
			  * decide_insertion() and decide_removal(). 
			  *
			  * Each 100 loops, the system will be measured. This number is 
			  * chosen so that the configuration has a sufficient chance to 
			  * change such that it is reasonably uncorrelated to the previous 
			  * measurement. Note that for very large average_orders, this 
			  * number should be made larger.
			  * 
			  * The measurements themselves are performed through 
			  * measure_times().
			  *
			  * As the size of the arrays are dynamically adjusted, this 
			  * function will check to see if we have reached the current 
			  * maximum size of the arrays and then call ReallocateArrays() in 
			  * order to increase the size by 30.
			  */
             void sampling_proc();

			 /*! Performs the measurement for a particular configuration.
			  *
			  * Both the imaginary time and Matsubara measurements are 
			  * performed in this function. 
			  * 
			  * The function first determines the Green's functions at all the 
			  * required times, that result from the subtraction of all of the 
			  * configuration times from one another. 
			  *
			  * The imaginary time measurements are performed with the 
			  * appropriate matrix operations, taking into account of the 2x2 
			  * substructure of the spinful terms. The formula for this is:
			  * \f[
			  * G_{c,\sigma,\sigma'}(\tau) = G_{0,\sigma,\sigma'}(\tau) + 
			  * \left[ \sum_{lm} \mathbf{G}_{0}(\tau,\tau_l) \mathbf{M}_{lm} 
			  * \mathbf{G}_{0}(\tau_m,0) \right]_{\sigma,\sigma'}
			  * \f]
			  * where the bold matrices indicate that these are indexed with 
			  * spin numbers. For example:
			  * \f[
			  * [\mathbf{G}_0]_{\sigma,\sigma'} = G_{0,\sigma\sigma'}
			  * \f]
			  * The matrix M is defined through:
			  * \f[
			  * \mathbf{M}_{lm} = (e^V - 1)\mathbf{N}^c_{lm}
			  * \f]
			  * where \f$e^V\f$ is a block diagonal matrix, with each block 
			  * containing a diagonal of \f$e^{V_\uparrow}, 
			  * e^{V_\downarrow}\f$. The matrix N is described in 
			  * decide_insertion().
			  *
			  * Note that the initial matrix \f$G_{0,\sigma,\sigma'}\f$ is not 
			  * measured as part of every loop and instead added in the final 
			  * step save_and_det_Greentau().
			  *
			  * The Matsubara measurements are more direct, and are more simply 
			  * expressed through:
			  * \f[
			  * G_{c,\sigma,\sigma'}(i\omega_n) = 
			  * G_{0,\sigma\sigma'}(i\omega_n) + [\sum_{lm} \exp{i\omega_n 
			  * (\tau_l - \tau_m)} \mathbf{G}_{0} \mathbf{M} \mathbf{G}_{0}]_{\sigma\sigma'}
			  * \f]
			  * where we can avoid much of the operation by simply accumulating 
			  * values of \f$\mathbf{M} \exp{i\omega_n (\tau_l - \tau_m}\f$ for 
			  * each iteration, and only performing the matrix product at the 
			  * end.
			  */
             void measure_times();

			 /*! Finalise the Green's functions and save them to the 
			  * appropriate files.
			  *
			  * This function performs the operations that could be left out in 
			  * the measure_times() function. It adds the non-interacting part 
			  * to the imaginary time measurements, and performs the 
			  * appropriate matrix multiplications for the Matsubara 
			  * measurements.
			  *
			  * The imaginary time measurements are then Fourier transformed 
			  * back into Matsubara frequencies, and then these are all saved 
			  * to the appropriate files. The output for each of these 
			  * functions are also saved into the supplied arguments G_up, 
			  * G_down, F, Gw_up, Gw_down and Fw.
			  *
			  * The argument m exists to allow an id to set for each iteration 
			  * of the solver in (for example) the DMFT loops. It can safely be 
			  * set to m=0 without an issues.
			  */
             void save_and_det_Greentau(cd G_up[], cd G_down[], cd F[], int m, cd Gw_up[], cd Gw_down[], cd Fw[]);

			 /*! Convenience function to save a particular Green's function to 
			  * a file.
			  *
			  * This function saves the Green's function, which is in G and of 
			  * size Size, to the file STR. The format is as rows for each 
			  * element of the Green's function, and each row contains the real 
			  * and imaginary parts as doubles in string format.
			  * 
			  */
             void save_green(cd G[], int Size, string STR);

			 /*! Reallocate the space for the configuration arrays and the N 
			  * matrix.
			  *
			  * This function increases the space for the configuration arrays 
			  * tau_meas, taus and spins as well as the N matrix. It is 
			  * necessary to be careful when copying the N matrix due to the 
			  * storage mechanism.
			  */
			 void ReallocateArrays(int new_size);
};
//------------------------------------------------------------------------------
//Konstruktor
WC::WC(cd g0_up_1[], cd g0_down_1[], cd f0_1[], double U_1, int Size_0, int Size1_1, int NoIt_1)
{
             Size=Size_0;
             Size1=Size1_1;
			 Size2=20;
             pi=3.1415926535;
             NoIt=NoIt_1;
             NoSteps=0;
             L=401;
             tau = new double[Size];
             g0_up = new cd[Size1];
             g0_down = new cd[Size1];
             f0 = new cd[Size1];
             for (int l=0; l<Size1; l++){
                 g0_up[l]=g0_up_1[l];
                 g0_down[l]=g0_down_1[l];
                 f0[l]=f0_1[l];}
             G0_up = new cd [Size];
             G0_down = new cd [Size];
             F0 = new cd [Size];
             U=U_1;
             seed= new long[1];
			 // Added by Danny
			 // Get seed from urandom
			 FILE * urandom = fopen("/dev/urandom","rb");
			 assert(fread(seed,sizeof(long),1,urandom) == 1);
			 //srandom(seed);
			 fclose(urandom);
			 if((*seed) > 0) (*seed) = -(*seed);
             //seed[0]=-(long)time(0);
             K=5.0;
             omega = new double [Size1];
             Gamma = new double [2];
             Gamma[0] = exp(-1.0*log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));
             Gamma[1] = exp(log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));
             taus = new double [Size2];
             spins = new int [Size2];
             N = new cd [Size2*Size2*4];
             Sign=1.0;
             n=1;
             tau_meas=new double[L];
             G_meas=new cd[3*L];
             for (int i=0; i<L; i++){
                G_meas[3*i]=G_meas[3*i+1]=G_meas[3*i+2]=0.0;
                tau_meas[i]=i*1.0/(L-1);}
             // Added by Danny
             for(int i=0;i < 4;i++)
             {
                 M_meas[i] = new cd[Size1];
                 memset(M_meas[i],0.,sizeof(cd)*Size1);
             }
}

void WC::ReallocateArrays(int new_size)
{
	cout << "Increasing size to " << new_size << endl;
	cout.flush();

	double *new_taus = new double[new_size];
	int *new_spins = new int[new_size];
	cd * new_N = new cd[new_size*new_size*4];

	//cout << "Increasing size to " << new_size << endl;
	//cout.flush();
	memcpy(new_taus,taus,sizeof(double)*Size2);
	memcpy(new_spins,spins,sizeof(int)*Size2);
	for(int i = 0;i < Size2*2;i++)
		memcpy(&new_N[new_size*2*i],&N[Size2*2*i],sizeof(cd)*Size2*2);

	//cout << "Increasing size to " << new_size << endl;
	//cout.flush();
	delete[] taus;
	delete[] spins;
	delete[] N;

	//cout << "Increasing size to " << new_size << endl;
	//cout.flush();
	taus = new_taus;
	spins = new_spins;
	N = new_N;

	//cout << "Increasing size to " << new_size << endl;
	//cout.flush();

	Size2 = new_size;
}


//------------------------------------------------------------------------------            
//Destruktor
WC::~WC()
{
             delete [ ] tau;
             delete [ ] taus;
             delete [ ] spins;
             delete [ ] N;
             delete [ ] G0_up;
             delete [ ] G0_down;
             delete [ ] F0;
             delete [ ] Gamma;
             delete [ ] g0_up;
             delete [ ] g0_down;
             delete [ ] f0;
             delete [ ] seed;
             delete [ ] omega;
             delete [ ] G_meas;
             delete [ ] tau_meas;
             // Added by Danny
             for(int i = 0;i < 4;i++)
                 delete[] M_meas[i];
}
//------------------------------------------------------------------------------
void WC::NIgreen()
//Fouriertransformation of frequencies to imaginary time and spline interpolation
{
    double *smalltau = new double [Size1+1];
    cd *gTreal = new cd [Size1+1];
    //double *NIG_prime = new double [Size1+1];//Ableitung von G, für Spline-Interpolation
	//double dummy;
    const cd i_imag = cd(0,1);

    for (int l=0; l<Size1; l++){
        omega[l]=pi*(2.0*(l-(Size1)/2)+1);
        smalltau[l]=1.0/Size1*l;}
        smalltau[Size1]=1.0;
        
    for (int l=0; l<Size; l++){tau[l]=1.0/(Size-1)*l;}
 
	// Danny:
	// save NI matsubara for testing the Fourier transforms.
	//FILE * file = fopen("nigreen_w.txt","wb");
	//for(int i = 0;i < Size1;i++)
	//	fprintf(file,"%.16g %.16g\n",real(g0_up[i]),imag(g0_up[i]));
	//fclose(file);

    fftMatsubara(gTreal, g0_up, Size1);
    save_green(gTreal, Size1+1, "nigreen.txt");

	// Temporary size increase with interpolation
    //ifftMatsubara(gTreal, g0_up, Size1);

	//file = fopen("nigreen_w2.txt","wb");
	//for(int i = 0;i < Size1;i++)
	//	fprintf(file,"%.16g %.16g\n",real(g0_up[i]),imag(g0_up[i]));
	//fclose(file);
    

    //spline_derivative(gTreal, Size1, NIG_prime);    
    //splines(smalltau, gTreal, NIG_prime, tau, G0_up, Size1+1, Size);    
	// TODO: If the G_upup and G_downdown functions are definitely real, then can 
	// perform only the real spline interpolation.
    PythonSplinesComplex(smalltau, gTreal, tau, G0_up, Size1+1, Size);    
    //-------------------------------------//
    
    fftMatsubara(gTreal, g0_down, Size1);
    
    //spline_derivative(gTreal, Size1, NIG_prime);    
    //splines(smalltau, gTreal, NIG_prime, tau, G0_down, Size1+1, Size);
	// TODO: If the G_upup and G_downdown functions are definitely real, then can 
	// perform only the real spline interpolation.
    PythonSplinesComplex(smalltau, gTreal, tau, G0_down, Size1+1, Size);
    
    //-------------------------------------//
    
    fftMatsubaraF(gTreal, f0, Size1);
    
    //spline_derivative(gTreal, Size1, NIG_prime);    
    //splines(smalltau, gTreal, NIG_prime, tau, F0, Size1+1, Size);
    PythonSplinesComplex(smalltau, gTreal, tau, F0, Size1+1, Size);
    
    //save_NIgreen(G0_down, Size);//------------------------------------------------  
    for (int l=0; l<Size; l++){tau[l]=double(l)/(Size-1.0);}
    
    delete [] smalltau;
    delete [] gTreal;
    //delete [] NIG_prime;
}
//------------------------------------------------------------------------------
void WC::start_conf()
//Start configuration for the sampling
{
	// Added by Danny since the option to change K came up
    Gamma[0] = exp(-1.0*log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));
    Gamma[1] = exp(log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));

    n=1;
    taus[0]=0.25;
    spins[0]=1;
    cd *S=new cd[4];
    S[0]=1.0-G0_up[0]*(1.0-Gamma[spins[0]]);
    S[1]=-F0[0]*(1.0-Gamma[1-spins[0]]);
	// Danny: Modified for complex F0.
    S[2]=-conj(F0[0])*(1.0-Gamma[spins[0]]);
    S[3]=1.0-G0_down[0]*(1.0-Gamma[1-spins[0]]);
    
    invert(S, det(S));
    
    N[0]=S[0];
    N[index(0,1,Size2*2)]=S[1];
    N[index(1,0,Size2*2)]=S[2];
    N[index(1,1,Size2*2)]=S[3];
    delete [] S;
} 
//------------------------------------------------------------------------------
void WC::sampling_proc()
//Sampling procedure of the QMC algorithm
{
    start_conf();
    cd Sine=0.0;
    long Np=0;

    for (int l=0; l<NoIt; l++){
		// Danny changed here
        if(l%100==9){
            measure_times();
            NoSteps++;}
		// Temp Danny debugging
		if(l%100000 == 0)
		{
			printf("Iteration number %dk and pert_order %g\n",l/1000,((double)Np)/l);
			fflush(stdout);
		}
        Np+=n;
        Sine+=Sign;
		// Danny changed
		if(acceptance(seed, 0.5)){
		//if(n == 0 || acceptance(seed, 0.5)){
            n+=decide_insertion(N, taus, spins, Gamma, G0_up, G0_down, F0, tau, n, Size, Size2, seed, K, Sign);

			// Danny added
			if(n == Size2)
				ReallocateArrays(Size2 + 30);
		}
        else{
			// Danny changed
			if(n>1){
			//if(n>0){
                n-=decide_removal(N, taus, spins, K, Gamma, n, Size2, seed, Sign);}}}             
	pert_order = ((double)Np)/NoIt;
    cout<<"average perturbation order: "<<pert_order<<endl;
    cout<<"average sign: "<<real(Sine)/NoIt << " + i" << imag(Sine)/NoIt <<endl;

    // Note that here this is setting Sign to be the global value, not the 
    // local value anymore.
	Sign = Sine / double(NoIt);

}
//------------------------------------------------------------------------------
void WC::measure_times()
//Subfunction of the measurement of the Matrix M during the sampling
{
    cd dummy1;
    cd dummy2;
    cd*g_11f=new cd[n];
    cd*g_22f=new cd[n];
    cd*g_12f=new cd[n];
    cd*g_21f=new cd[n];
    cd*g_11b=new cd[n];
    cd*g_22b=new cd[n];
    cd*g_12b=new cd[n];
    cd*g_21b=new cd[n];
    for (int i=0; i<n; i++){
        g_11b[i]=single_linear(tau, G0_up, taus[i], Size);
        g_12b[i]=single_linear(tau, F0, taus[i], Size);
		// Danny: Modified for complex F0.
        g_21b[i]=conj(g_12b[i]);
        g_22b[i]=single_linear(tau, G0_down, taus[i], Size);}
            
    for (int l=0; l<L; l++){
        for (int i=0; i<n; i++){
            g_11f[i]=single_linear(tau, G0_up, tau_meas[l]-taus[i], Size);
            g_12f[i]=single_linear(tau, F0, tau_meas[l]-taus[i], Size);
			// Danny: Modified for complex F0.
            g_21f[i]=conj(g_12f[i]);
            g_22f[i]=single_linear(tau, G0_down, tau_meas[l]-taus[i], Size);}
        
        for (int i=0; i<n; i++){
            dummy1=(1.0-Gamma[spins[i]]);
            dummy2=(1.0-Gamma[1-spins[i]]);
            for(int j=0; j<n; j++){
                G_meas[3*l]+=(dummy1*g_11f[i]*(N[index(2*i,2*j,Size2*2)]*g_11b[j]+N[index(2*i,2*j+1,Size2*2)]*g_21b[j])+dummy2*g_12f[i]*(N[index(2*i+1,2*j,Size2*2)]*g_11b[j]+N[index(2*i+1,2*j+1,Size2*2)]*g_21b[j]));
                G_meas[3*l+1]+=(dummy1*g_21f[i]*(N[index(2*i,2*j,Size2*2)]*g_12b[j]+N[index(2*i,2*j+1,Size2*2)]*g_22b[j])+dummy2*g_22f[i]*(N[index(2*i+1,2*j,Size2*2)]*g_12b[j]+N[index(2*i+1,2*j+1,Size2*2)]*g_22b[j]));
                G_meas[3*l+2]+=(dummy1*g_11f[i]*(N[index(2*i,2*j,Size2*2)]*g_12b[j]+N[index(2*i,2*j+1,Size2*2)]*g_22b[j])+dummy2*g_12f[i]*(N[index(2*i+1,2*j,Size2*2)]*g_12b[j]+N[index(2*i+1,2*j+1,Size2*2)]*g_22b[j]));
            }}}

    // Added by Danny
    // This part is just
    // M = (e^V - 1) N exp(-i\omega_n\tau_b) exp(i\omega_n\tau_f)
    // (i.e. a matrix that must be stuck between G_0s later.
    for(int w=0;w < Size1;w++)
    {
        for(int i=0;i < n;i++)
        {
            for(int j=0;j < n;j++)
            {
                const cd i_imag = cd(0,1);
                cd exp_dummy = exp(i_imag * omega[w] * (taus[i] - taus[j]));
                M_meas[0][w] += (1.0 - Gamma[spins[i]]) * N[index(2*i,2*j,Size2*2)] * exp_dummy;
                M_meas[1][w] += (1.0 - Gamma[spins[i]]) * N[index(2*i,2*j+1,Size2*2)] * exp_dummy;
                M_meas[2][w] += (1.0 - Gamma[1-spins[i]]) * N[index(2*i+1,2*j,Size2*2)] * exp_dummy;
                M_meas[3][w] += (1.0 - Gamma[1-spins[i]]) * N[index(2*i+1,2*j+1,Size2*2)] * exp_dummy;
                //M_meas[0][w] += (1.0 - Gamma[spins[i]]) * N[index(2*i,2*j,Size2*2)] * exp_dummy;
                //M_meas[3][w] += (1.0 - Gamma[1-spins[i]]) * N[index(2*i+1,2*j+1,Size2*2)] * exp_dummy;
            }
        }
    }

    delete [] g_11f;
    delete [] g_12f;
    delete [] g_21f;
    delete [] g_22f;
    delete [] g_11b;
    delete [] g_12b;
    delete [] g_21b;
    delete [] g_22b;
}
//------------------------------------------------------------------------------
void WC::save_and_det_Greentau(cd G_up[], cd G_down[], cd F[], int m, cd Gw_up[], cd Gw_down[], cd Fw[])
//Determining the greenfunction in frequency space, transformation to imaginary times and saving
{
    cd *G_up1=new cd[L];
    cd *F1=new cd[L];
    cd *G_down1=new cd[L];
    cd *GreenTau_up=new cd[Size1+1];
    cd *GreenTau_down=new cd[Size1+1];
    cd *FTau=new cd[Size1+1];
    
    stringstream M;
    M<<m;

    for(int i=0; i<L; i++){
        G_up1[i]=single_linear(tau, G0_up, tau_meas[i], Size)+G_meas[3*i]*1.0/double(NoSteps);
        G_down1[i]=single_linear(tau, G0_down, tau_meas[i], Size)+G_meas[3*i+1]*1.0/double(NoSteps);
        F1[i]=single_linear(tau, F0, tau_meas[i], Size)+G_meas[3*i+2]*1.0/double(NoSteps);}
        //G_up1[i]=single_linear(tau, G0_up, tau_meas[i], Size);
        //G_down1[i]=single_linear(tau, G0_down, tau_meas[i], Size);
        //F1[i]=single_linear(tau, F0, tau_meas[i], Size);}

    // Added by Danny
    // Reconstructing the Green's function from the Matsubara measurements 
    // requires a matrix multiplication
    for(int w = 0;w < Size1;w++)
    {
        // Explicit 2x2x2 matrix multiplcatin - ugly!
        // Note that beta = 1 here.
        cd temp, temp2; // RHS col (i.e. [M G_0]_{:,i} )
		// Danny: Modified for complex F0.
        temp = M_meas[0][w]*g0_up[w] + M_meas[1][w]*conj(f0[w]);
        temp2 = M_meas[2][w]*g0_up[w] + M_meas[3][w]*conj(f0[w]);
        temp /= NoSteps;
        temp2 /= NoSteps;
        Gw_up[w] = g0_up[w] - (g0_up[w]*temp + f0[w]*temp2);
        //Lower left term would go here.
        temp = M_meas[0][w]*f0[w] + M_meas[1][w]*g0_down[w];
        temp2 = M_meas[2][w]*f0[w] + M_meas[3][w]*g0_down[w];
        temp /= NoSteps;
        temp2 /= NoSteps;
		// TODO: Check to see if this is right!! Discovered a mistake
        Fw[w] = f0[w] - (g0_up[w]*temp + f0[w]*temp2);
		// Danny: Modified for complex F0.
        Gw_down[w] = g0_down[w] - (conj(f0[w])*temp + g0_down[w]*temp2);
        
        //Gw_up[w] = g0_up[w] - g0_up[w]*g0_up[w] * M_meas[0][w] * (1./NoSteps);
        //Gw_down[w] = g0_down[w] - g0_down[w]*g0_down[w] * M_meas[3][w] * (1./NoSteps);
        
        //Gw_up[w] = M_meas[0][w];
        //Gw_down[w] = M_meas[3][w];
    }


    double *tau1 = new double [Size1+1];
    //double *DER2 = new double [L];
    for (int l=0; l<Size1+1; l++){tau1[l]=1.0/(Size1)*l;}
    //spline_derivative(G_up1, L-1, DER2);
    //splines(tau_meas, G_up1, DER2, tau1, GreenTau_up, L, Size1+1);
    PythonSplinesComplex(tau_meas, G_up1, tau1, GreenTau_up, L, Size1+1);
    save_green(GreenTau_up, Size1+1, "GreenTauup"+M.str()+".txt");
    ifftMatsubara(GreenTau_up, G_up, Size1);

    //spline_derivative(G_down1, L-1, DER2);    
    //splines(tau_meas, G_down1, DER2, tau1, GreenTau_down, L, Size1+1);
    PythonSplinesComplex(tau_meas, G_down1, tau1, GreenTau_down, L, Size1+1);
    save_green(GreenTau_down, Size1+1, "GreenTaudown"+M.str()+".txt");
    ifftMatsubara(GreenTau_down, G_down, Size1);
    
    
    //spline_derivative(F1, L-1, DER2);    
    //splines(tau_meas, F1, DER2, tau1, FTau, L, Size1+1);
    PythonSplinesComplex(tau_meas, F1, tau1, FTau, L, Size1+1);
    save_green(FTau, Size1+1, "FTau"+M.str()+".txt");
    ifftMatsubaraF(FTau, F, Size1);
    
    delete [] GreenTau_up;
    delete [] GreenTau_down;
    delete [] FTau;
    delete [] G_up1;
    delete [] G_down1;
    delete [] F1;
    //delete [] DER2;
    delete [] tau1;
}
//------------------------------------------------------------------------------
void WC::save_green(cd G[], int Size, string STR)
//saving green function
{
     FILE*NIgreen_writefile=fopen (STR.c_str(), "w");
     for (int i=0; i<Size; i++){
             fprintf(NIgreen_writefile, "%g %g\n", real(G[i]),imag(G[i]));}
     fprintf(NIgreen_writefile, "\n");
     fclose(NIgreen_writefile);
}
//------------------------------------------------------------------------------    
