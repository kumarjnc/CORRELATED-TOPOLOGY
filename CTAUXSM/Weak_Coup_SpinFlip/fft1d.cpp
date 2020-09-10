/*! \file
 *
 * \brief Routines to perform the Fourier transforms.
 *
 * The routines here use the GSL (GNU Scientific Library) to perform a FFT on 
 * the Green's functions, from and to Matsubara frequency and imaginary time.
 */
#include <gsl/gsl_fft_complex.h>
     
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

/*! A wrapped around the GSL Fourier transform library.
 */ 
void fft1d(cd in_fct[], cd out_fct[], int dim1)
//1-dimensional fast fourier transform
{
       int i;
       const cd i_imag=cd(0,1);
       double data[2*dim1];
     
       gsl_fft_complex_wavetable * wavetable;
       gsl_fft_complex_workspace * workspace;
     
       for (i = 0; i < dim1; i++)
         {
           REAL(data,i) = real(in_fct[i]);
           IMAG(data,i) = imag(in_fct[i]);
         }
         
       wavetable = gsl_fft_complex_wavetable_alloc (dim1);
       workspace = gsl_fft_complex_workspace_alloc (dim1);
     

       gsl_fft_complex_forward (data, 1, dim1, wavetable, workspace);
     
       for (i = 0; i < dim1; i++){
       out_fct[i]=REAL(data,i)+i_imag*(IMAG(data,i));
       }


       gsl_fft_complex_wavetable_free (wavetable);
       gsl_fft_complex_workspace_free (workspace);
}

/*! Fourier transform the supplied Matsubara frequency function into imaginary 
 * time. This specific function is for the functions diagonal in spin.
 *
 * Because the Fourier transform is not well suited (numerically) to 
 * transforming the leading \f$1/(i\omega_N)\f$ factor, this is treated as a 
 * separate term in the transform, and manually transformed. This is possible, 
 * as one knows that this term must have a coefficient of 1.
 */
void fftMatsubara(cd G_Tau[], cd G_omega[], int Size)
//Fourier Transformation of G(omega) to imaginary time
{
    double pi=3.1415926535;
    const cd i_imag = cd(0,1);
    cd* g_omega=new cd [Size];
    double* tau=new double[Size];
    double* omega=new double [Size];
    cd* gT = new cd [Size];
    
    for (int l=0; l<Size; l++){
        omega[l]=pi*(2.0*(l-(Size)/2)+1);
        tau[l]=double(l)/(Size)*pi*(Size-1);}
        
    for (int i=0; i<Size; i++){
        g_omega[i]=G_omega[i]+i_imag*1.0/(omega[i]);}
            
    fft1d(g_omega, gT, Size);
        
    for (int l=0; l<Size; l++){
		//G_Tau[l]=0.5-real(gT[l]*(cos(tau[l])+i_imag*sin(tau[l])));}
		G_Tau[l]=0.5-gT[l]*(cos(tau[l])+i_imag*sin(tau[l]));}
    G_Tau[Size]=1.0-G_Tau[0];
    
    delete [] g_omega;
    delete [] gT;
    delete [] tau;
    delete [] omega;
}

/*! Perform the Fourier transform from Matsubara to imaginary time, for the 
 * cross term Green's functions.
 *
 * In this case we do not need to worry about the \f$1/(i\omega_N)\f$ factor, 
 * as it is not present here.
 */
void fftMatsubaraF(cd G_Tau[], cd G_omega[], int Size)
//Fourier Transformation of F(omega) to imaginary time
{
    double pi=3.1415926535;
    const cd i_imag = cd(0,1);
    cd* g_omega=new cd [Size];
    double* tau=new double[Size];
    cd* gT = new cd [Size];
    for (int l=0; l<Size; l++){
        tau[l]=double(l)/(Size)*pi*(Size-1);}

    for (int i=0; i<Size; i++){
        g_omega[i]=G_omega[i];}
                   
    fft1d(g_omega, gT, Size);
    for (int l=0; l<Size; l++){
		//G_Tau[l]=-real(gT[l]*(cos(tau[l])+i_imag*sin(tau[l])));}
		G_Tau[l]=-gT[l]*(cos(tau[l])+i_imag*sin(tau[l]));}
    G_Tau[Size]=-G_Tau[0];
    delete [] g_omega;
    delete [] gT;
    delete [] tau;
}

/*! The inverse transform of fftMatsubara(). That is, from imaginary time to 
 * Matsubara frequency.
 *
 * The leading order term is also accounted for here.
 */
void ifftMatsubara(cd G_Tau[], cd G_omega[], int Size)
//Fouriertransformation from g(tau)->g(iomega)
{
 double pi=3.1415926535;
    const cd i_imag = cd(0,1);
    double* tau=new double[Size];
    double* omega=new double [Size];
    cd* gT = new cd [Size];
    cd f;
    
    for (int l=0; l<Size; l++){
        omega[l]=pi*(2.0*(l-(Size)/2)+1);
        tau[l]=double(l)/(Size)*pi*(Size-1);}
        
    for (int i=0; i<Size; i++){
        f=(cos(tau[i])-i_imag*sin(tau[i]));
        G_omega[i]=-1.0/Size*(G_Tau[i]-0.5)*f;}
            
    fft1d(G_omega, gT, Size);
        
    for (int l=1; l<Size; l++){
        G_omega[l]=gT[Size-l]-i_imag/omega[l];}
    G_omega[0]=gT[0]-i_imag/omega[0];

    delete [] gT;
    delete [] tau;
    delete [] omega;
}

/*! The inverse transform of fftMatsubaraF(). That is, from imaginary time to 
 * Matsubara frequency.
 */
void ifftMatsubaraF(cd G_Tau[], cd G_omega[], int Size)
//Fourier Transformation of F(tau)->F(iomega)
{
    double pi=3.1415926535;
    const cd i_imag = cd(0,1);
    double* tau=new double[Size];
    cd* gT = new cd [Size];
    for (int l=0; l<Size; l++){
        tau[l]=double(l+1)/(Size)*pi*(Size-1);}

    for (int i=0; i<Size; i++){
        G_omega[i]=-1.0/Size*(cos(tau[i])+i_imag*sin(tau[i]))*G_Tau[i];}
                   
    fft1d(G_omega, gT, Size);
    for (int l=1; l<Size; l++){
        G_omega[l]=gT[Size-l];}
    G_omega[0]=gT[0];
    delete [] gT;
    delete [] tau;
}
