#include <complex>
#include <iostream>
#include <fstream>

#define cd complex<double>

using namespace std;

#include <gsl/gsl_fft_complex.h>
     
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

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

int main(void)
{
	int Size = 40;

	cd G_Tau[Size];
	cd G_omega[Size];

	// OLD
    double pi=3.1415926535;
    const cd i_imag = cd(0,1);
    cd* g_omega=new cd [Size];
    double* tau=new double[Size];
    double* omega=new double [Size];
    cd* gT = new cd [Size];
    
    for (int l=0; l<Size; l++){
        omega[l]=pi*(2.0*(l-(Size)/2)+1);
        tau[l]=double(l)/(Size)*pi*(Size-1);}

	// NEW
	for(int i = 0;i < Size;i++)
	{
		if(i < Size/2)
			G_omega[i] = 1./(i_imag*omega[i]) + 1./pow(i_imag*omega[i],3);
		else
			G_omega[i] = 1./(i_imag*omega[i]) + 1./pow(i_imag*omega[i],3);
	}
	
        
	if(true)
	{
			// OLD
			for (int i=0; i<Size; i++){
				g_omega[i]=G_omega[i]+i_imag*1.0/(omega[i]);}
					
			fft1d(g_omega, gT, Size);
				
			for (int l=0; l<Size; l++){
				//G_Tau[l]=0.5-real(gT[l]*(cos(tau[l])+i_imag*sin(tau[l])));}
				G_Tau[l]=0.5-gT[l]*(cos(tau[l])+i_imag*sin(tau[l]));}
			G_Tau[Size]=1.0-G_Tau[0];
			
			ofstream file("output.txt");

			for(int i = 0;i < Size;i++)
			{
				file << real(G_Tau[i]) << " " << imag(G_Tau[i]) << endl;
			}
			file.close();
	}
	else
	{
		delete[] tau;
		tau=new double[Size+1];
		for(int l=0; l<Size+1; l++)
			tau[l]=double(l)/(Size);
		for(int i=0;i < Size;i++)
			omega[i]=pi*(2.0*(i-(Size)/2)+1);

		for(int i=0;i < Size+1;i++)
		{
			G_Tau[i] = 0.;
			//for(int j=Size/2;j < Size;j++)
			for(int j=0;j < Size;j++)
			{
				//G_Tau[i] += 2*real((G_omega[j] + i_imag*1.0/(omega[j])) * exp(i_imag*omega[j]*tau[i]));
				G_Tau[i] += (G_omega[j] - 1.0/(i_imag*omega[j])) * exp(-i_imag*omega[j]*tau[i]);
				// Negative here to change sign convention for the solver.
				G_Tau[i] = -G_Tau[i];

				//cd temp = 0.;
				//temp = G_omega[j] + i_imag*1.0/(omega[j]);
				//printf("%g\n",real(temp));
				//temp *= exp(i_imag*omega[j]*tau[i]);
				//printf("%g\n",real(temp));
				//temp = 2*real(temp);
				//printf("%g\n",real(temp));
				//exit(1);
			}
			G_Tau[i] += 0.5;
		}

		ofstream file("output2.txt");

		for(int i = 0;i < Size;i++)
		{
			file << real(G_Tau[i]) << " " << imag(G_Tau[i]) << endl;
		}
		file.close();
	}


    delete [] g_omega;
    delete [] gT;
    delete [] tau;
    delete [] omega;
}
