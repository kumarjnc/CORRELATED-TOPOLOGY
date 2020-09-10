//------------------------------------------------------------------------------
double sign(int l)
{
    if(l<0){return -1.0;}
    else{return 1.0;}
}
//------------------------------------------------------------------------------
class WC
{
      public:
             WC(cd g_0_up_1[], cd g0_down_1[], cd f0_1[], double U, int Size, int Size1, int NoIt);
             ~WC();

             int Size; //Number of Matsubara Frequencies in the sampling green function
             int Size1; //Number of Matsubara Frequencies for DMFT iteration
             cd *g0_up; //Non-interacting green functions for DMFT iteration
             cd *g0_down;
             cd *f0;
             double *tau; //Imaginary times for the sampling process
             double *G0_up;//Interpolated green functions for the sampling process
             double *G0_down;
             double *F0;
             double pi;
             int n;//dynamical perturbation order (Size of the Matrix N)
             double *taus;//dynamical imaginary times during the sampling
             int *spins;//dynamical Ising spins during the sampling
             double *N;//dynamical matrix N
             long *seed;//seed for random number generator
             double U;
             double K;//real parameter K for Hubbard-Stratonovich decoupling
             double* Gamma;//Vector Gamma for the sampling
             double phi;
             double *omega;
             double *Sign;//Sign, just to check
             int Size2;
             int NoIt;
             int NoSteps;
             int L;//Number of Measurements for the green functions
             double* G_meas;
             double* tau_meas;

             void NIgreen();
             void start_conf();
             void sampling_proc();
             void save_pertorder();
             void green_measurement(double tau_test, int l);
             void save_MeasuredGreen();
             void save_Green_Measure(double Measurement_green[], int L);
             void measure_times();
             void save_Var_Measure(double Measurement_var[], int L);
             void save_and_det_Greentau(cd G_up[], cd G_down[], cd F[], int m);
             void save_green(double G[], int Size, string STR);
};
//------------------------------------------------------------------------------
//Konstruktor
WC::WC(cd g0_up_1[], cd g0_down_1[], cd f0_1[], double U_1, int Size_0, int Size1_1, int NoIt_1)
{
             Size=Size_0;
             Size1=Size1_1;
             Size2=500;
             pi=3.1415926535;
             NoIt=NoIt_1;
             NoSteps=0;
             L=201;
             tau = new double[Size];
             g0_up = new cd[Size1];
             g0_down = new cd[Size1];
             f0 = new cd[Size1];
             for (int l=0; l<Size1; l++){
                 g0_up[l]=g0_up_1[l];
                 g0_down[l]=g0_down_1[l];
                 f0[l]=f0_1[l];}
             G0_up = new double [Size];
             G0_down = new double [Size];
             F0 = new double [Size];
             U=U_1;
             seed= new long[1];
             seed[0]=-(long)time(0);
             K=1.0;
             omega = new double [Size1];
             Gamma = new double [2];
             Gamma[0] = exp(-1.0*log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));
             Gamma[1] = exp(log(1.0+U/(2.0*K)+sqrt(pow(1.0+U/(2.0*K),2.0)-1.0)));
             taus = new double [Size2];
             spins = new int [Size2];
             N = new double [Size2*Size2];
             Sign= new double [1];
             Sign[0]=1.0;
             n=1;
             tau_meas=new double[L];
             G_meas=new double[3*L];
             for (int i=0; i<L; i++){
                G_meas[3*i]=G_meas[3*i+1]=G_meas[3*i+2]=0.0;
                tau_meas[i]=i*1.0/(L-1);}
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
             delete [ ] Sign;
             delete [ ] G_meas;
             delete [ ] tau_meas;
}
//------------------------------------------------------------------------------
void WC::NIgreen()
//Fouriertransformation of frequencies to imaginary time and spline interpolation
{
    double *smalltau = new double [Size1+1];
    double *gTreal = new double [Size1+1];
    double *NIG_prime = new double [Size1+1];//Ableitung von G, für Spline-Interpolation
    double dummy;
    const cd i_imag = cd(0,1);

    for (int l=0; l<Size1; l++){
        omega[l]=pi*(2.0*(l-(Size1)/2)+1);
        smalltau[l]=1.0/Size1*l;}
        smalltau[Size1]=1.0;
        
    for (int l=0; l<Size; l++){tau[l]=1.0/(Size-1)*l;}
 
    fftMatsubara(gTreal, g0_up, Size1);
    save_green(gTreal, Size1+1, "nigreen.txt");
    

    spline_derivative(gTreal, Size1, NIG_prime);    
    splines(smalltau, gTreal, NIG_prime, tau, G0_up, Size1+1, Size);    
    //-------------------------------------//
    
    fftMatsubara(gTreal, g0_down, Size1);
    
    spline_derivative(gTreal, Size1, NIG_prime);    
    splines(smalltau, gTreal, NIG_prime, tau, G0_down, Size1+1, Size);
    
    //-------------------------------------//
    
    fftMatsubaraF(gTreal, f0, Size1);
    
    spline_derivative(gTreal, Size1, NIG_prime);    
    splines(smalltau, gTreal, NIG_prime, tau, F0, Size1+1, Size);
    
    //save_NIgreen(G0_down, Size);//------------------------------------------------  
    for (int l=0; l<Size; l++){tau[l]=double(l)/(Size-1.0);}
    
    delete [] smalltau;
    delete [] gTreal;
    delete [] NIG_prime;
}
//------------------------------------------------------------------------------
void WC::start_conf()
//Start configuration for the sampling
{
    n=1;
    taus[0]=0.25;
    spins[0]=1;
    double *S=new double[4];
    S[0]=1.0-G0_up[0]*(1.0-Gamma[spins[0]]);
    S[1]=-F0[0]*(1.0-Gamma[1-spins[0]]);
    S[2]=-F0[0]*(1.0-Gamma[spins[0]]);
    S[3]=1.0-G0_down[0]*(1.0-Gamma[1-spins[0]]);
    
    invert(S, det(S));
    
    N[0]=S[0];
    N[index(0,1,Size2)]=S[1];
    N[index(1,0,Size2)]=S[2];
    N[index(1,1,Size2)]=S[3];
    delete [] S;
} 
//------------------------------------------------------------------------------
void WC::sampling_proc()
//Sampling procedure of the QMC algorithm
{
    start_conf();
    double Sine=0.0;
    int Np=0;

    for (int l=0; l<NoIt; l++){
        if(l%10==9){
            measure_times();
            NoSteps++;}
        Np+=n;
        Sine+=Sign[0];
        if(acceptance(seed, 0.5)){
            n+=decide_insertion(N, taus, spins, Gamma, G0_up, G0_down, F0, tau, n, Size, Size2, seed, K, Sign);}
        else{
            if(n>1){
                n-=decide_removal(N, taus, spins, K, Gamma, n, Size2, seed, Sign);}}}             
    cout<<"average perturbation order: "<<((double)Np)/NoIt<<endl;
    cout<<"average sign: "<<Sine/NoIt<<endl;

}
//------------------------------------------------------------------------------
void WC::measure_times()
//Subfunction of the measurement of the Matrix M during the sampling
{
    double dummy1;
    double dummy2;
    double*g_11f=new double[n];
    double*g_22f=new double[n];
    double*g_12f=new double[n];
    double*g_21f=new double[n];
    double*g_11b=new double[n];
    double*g_22b=new double[n];
    double*g_12b=new double[n];
    double*g_21b=new double[n];
    for (int i=0; i<n; i++){
        g_11b[i]=single_linear(tau, G0_up, taus[i], Size);
        g_12b[i]=single_linear(tau, F0, taus[i], Size);
        g_21b[i]=g_12b[i];
        g_22b[i]=single_linear(tau, G0_down, taus[i], Size);}
            
    for (int l=0; l<L; l++){
        for (int i=0; i<n; i++){
            g_11f[i]=single_linear(tau, G0_up, tau_meas[l]-taus[i], Size);
            g_12f[i]=single_linear(tau, F0, tau_meas[l]-taus[i], Size);
            g_21f[i]=g_12f[i];
            g_22f[i]=single_linear(tau, G0_down, tau_meas[l]-taus[i], Size);}
        
        for (int i=0; i<n; i++){
            dummy1=(1.0-Gamma[spins[i]]);
            dummy2=(1.0-Gamma[1-spins[i]]);
            for(int j=0; j<n; j++){
                G_meas[3*l]+=(dummy1*g_11f[i]*(N[index(2*i,2*j,Size2)]*g_11b[j]+N[index(2*i,2*j+1,Size2)]*g_21b[j])+dummy2*g_12f[i]*(N[index(2*i+1,2*j,Size2)]*g_11b[j]+N[index(2*i+1,2*j+1,Size2)]*g_21b[j]));
                G_meas[3*l+1]+=(dummy1*g_21f[i]*(N[index(2*i,2*j,Size2)]*g_12b[j]+N[index(2*i,2*j+1,Size2)]*g_22b[j])+dummy2*g_22f[i]*(N[index(2*i+1,2*j,Size2)]*g_12b[j]+N[index(2*i+1,2*j+1,Size2)]*g_22b[j]));
                G_meas[3*l+2]+=(dummy1*g_11f[i]*(N[index(2*i,2*j,Size2)]*g_12b[j]+N[index(2*i,2*j+1,Size2)]*g_22b[j])+dummy2*g_12f[i]*(N[index(2*i+1,2*j,Size2)]*g_12b[j]+N[index(2*i+1,2*j+1,Size2)]*g_22b[j]));}}}
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
void WC::save_and_det_Greentau(cd G_up[], cd G_down[], cd F[], int m)
//Determining the greenfunction in frequency space, transformation to imaginary times and saving
{
    double *G_up1=new double[L];
    double *F1=new double[L];
    double *G_down1=new double[L];
    double *GreenTau_up=new double[Size1+1];
    double *GreenTau_down=new double[Size1+1];
    double *FTau=new double[Size1+1];
    
    stringstream M;
    M<<m;

    for(int i=0; i<L; i++){
        G_up1[i]=single_linear(tau, G0_up, tau_meas[i], Size)+G_meas[3*i]*1.0/NoSteps;
        G_down1[i]=single_linear(tau, G0_down, tau_meas[i], Size)+G_meas[3*i+1]*1.0/NoSteps;
        F1[i]=single_linear(tau, F0, tau_meas[i], Size)+G_meas[3*i+2]*1.0/NoSteps;}


    double *tau1 = new double [Size1+1];
    double *DER2 = new double [L];
    for (int l=0; l<Size1+1; l++){tau1[l]=1.0/(Size1)*l;}
    spline_derivative(G_up1, L-1, DER2);
    splines(tau_meas, G_up1, DER2, tau1, GreenTau_up, L, Size1+1);
    save_green(GreenTau_up, Size1+1, "GreenTauup"+M.str()+".txt");
    ifftMatsubara(GreenTau_up, G_up, Size1);

    spline_derivative(G_down1, L-1, DER2);    
    splines(tau_meas, G_down1, DER2, tau1, GreenTau_down, L, Size1+1);
    save_green(GreenTau_down, Size1+1, "GreenTaudown"+M.str()+".txt");
    ifftMatsubara(GreenTau_down, G_down, Size1);
    
    
    spline_derivative(F1, L-1, DER2);    
    splines(tau_meas, F1, DER2, tau1, FTau, L, Size1+1);
    save_green(FTau, Size1+1, "FTau"+M.str()+".txt");
    ifftMatsubaraF(FTau, F, Size1);
    
    delete [] GreenTau_up;
    delete [] GreenTau_down;
    delete [] FTau;
    delete [] G_up1;
    delete [] G_down1;
    delete [] F1;
    delete [] DER2;
    delete [] tau1;
}
//------------------------------------------------------------------------------
void WC::save_green(double G[], int Size, string STR)
//saving green function
{
     FILE*NIgreen_writefile=fopen (STR.c_str(), "w");
     for (int i=0; i<Size; i++){
             fprintf(NIgreen_writefile, "%g ", G[i]);}
     fprintf(NIgreen_writefile, "\n");
     fclose(NIgreen_writefile);
}
//------------------------------------------------------------------------------    
