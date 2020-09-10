/*! \file
 *
 * \brief The interface to the main CT-AUX procedure.
 *
 * This code was written by Danny in order to interface the CT-AUX code to 
 * python. It takes several arguments via the command line, as well as possibly 
 * reading some data via stdin.
 *
 * The output is then given back in either designated files, or via stdout.
 *
 * The arguments to the function must be:
 * -# \c input_filename: The filename of the input data. Can be "-", see 
 *  comments.
 * -# \c output_filename: The filename where to outptu to. Can be "-", see 
 *  comments.
 * -# \c U: The interaction strength.
 * -# \c num_omega: The number of Matsubara frequencies.
 * -# \c num_sweeps: The number of QMC iterations to perform.
 *
 * It is also possible to supply an argument \c --K which will set the optional 
 * parameter K. If \c --binary is given, then the data will be saved in direct 
 * binary format (danger for portability!) or otherwise as plain text format.
 *
 * When \c input_filename or \c output_filename are set to "-", then the data 
 * will be read or written to or from stdin and stdout respectively.
 *
 * The input data should be given as either a series of rows of 7 numbers in 
 * text mode, which specify omega, real and imaginary parts of G0_up, real and 
 * imaginary parts of G0_down and real and imaginary parts of F0. Or as a 
 * direct binary dump of G0_up, G0_down and F0 in binary mode.
 *
 * On exit, for text mode, the perturbation order and average sign of the 
 * solver will be saved, followed by rows identical to the input data, but for 
 * the Green's functions. For binary mode, the following will be saved:
 *
 * -# \c pert_order (double)
 * -# \c Sign (complex double)
 * -# \c G_up (complex double array)
 * -# \c G_down (complex double array)
 * -# \c F (complex double array)
 * -# \c Gw_up (complex double array)
 * -# \c Gw_down (complex double array)
 * -# \c Fw (complex double array)
 *
 *  The quantities which include w in their names result from Matsubara 
 *  measurements, whereas the other quantities results from imaginary time 
 *  measurements (yet are still given in Matsubara frequencies).
 */
using namespace std;
#define cd complex<double>

//#include <ctime>
//#include <stdlib.h>
//#include <time.h>
//#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <fstream>
#include <iostream>
#include <complex>
//#include <algorithm>

#include <assert.h>
#include <string.h>
const cd I(0,1);

//#pragma GCC diagnostic warning "-Wall"
#include "Weak_Coup_SpinFlip/fft1d.cpp"//Fouriertrafo functions
#include "Weak_Coup_SpinFlip/random_fcts.cpp"//Random number generator and stochastic functions
#include "Weak_Coup_SpinFlip/Splines.cpp"//Spline interpolation functions
#include "Weak_Coup_SpinFlip/matrix_manip.cpp"//Matrix update functions
#include "Weak_Coup_SpinFlip/WC_SpinFlip.cpp"//Impurity solver

//#define DEBUG

#ifdef DEBUG
void LOG(const char * a,...)
{
    va_list args;
    va_start (args, a);
    vfprintf (stderr, a, args);
    va_end (args);
}
#define LOGDONE() fprintf(stderr," ...done\n")
#else
#define LOG(...)
#define LOGDONE()
#endif

int main(int argc,char ** argv)
{
    LOG("Parsing arguments...");
	const int req_args = 6;
	assert(argc >= req_args);
	// Arguments:
	// input_filename
	// output_filename
	// U
	// num_omega
	// num_sweeps	
	
    char ** cur_argv = &argv[1];
    char * input_filename = *cur_argv; cur_argv++;
    char * output_filename = *cur_argv; cur_argv++;
	double U;
	assert(sscanf(*cur_argv,"%lg",&U) == 1); cur_argv++;
	int num_omega;
	assert(sscanf(*cur_argv,"%d",&num_omega) == 1); cur_argv++;
	int num_sweeps;
	double dbl_temp;
	assert(sscanf(*cur_argv,"%lg",&dbl_temp) == 1); cur_argv++;
	num_sweeps = int(dbl_temp);

	bool binary_files = false;
	bool show_help = false;
	double K = 5.;

	for(int remaining = argc - req_args;remaining > 0;remaining--)
	{
		if(strcmp(*cur_argv,"--binary") == 0)
			binary_files = true;
		else if(strcmp(*cur_argv,"--K") == 0)
		{
			cur_argv++; remaining--;
			assert(sscanf(*cur_argv,"%lg",&K) == 1);
		}
		else
		{
			printf("Unknown argument %s\n",*cur_argv);
			show_help = true;
		}

		cur_argv++;
	}
	
	if(show_help)
	{
		printf("Help goes here!\n");
		exit(1);
	}
    LOGDONE();

	LOG("Read in data");
    FILE * file;
	cd * g0_up = new cd[num_omega*2];
	cd * g0_down = new cd[num_omega*2];
	cd * f0 = new cd[num_omega*2];
	if(strcmp(input_filename,"NONE") == 0)
	{
        // No functions provided, treat as zero.
		memset(g0_up,0,sizeof(cd)*num_omega*2);
		memset(g0_down,0,sizeof(cd)*num_omega*2);
		memset(f0,0,sizeof(cd)*num_omega*2);
	}
	else
	{
		if(strcmp(input_filename,"-") == 0)
			file = stdin;
		else
			file = fopen(input_filename,"rb");
		assert(file != NULL);

        // Recently changed this to accept only the complete frequency range 
        // (i.e. both postive and negative frequencies).
		if(binary_files)
		{
			int ret = fread(g0_up,sizeof(cd),num_omega*2,file);
			assert(ret == num_omega*2 || (ret == 0 && feof(file)));
			ret = fread(g0_down,sizeof(cd),num_omega*2,file);
			assert(ret == num_omega*2 || (ret == 0 && feof(file)));
			ret = fread(f0,sizeof(cd),num_omega*2,file);
			assert(ret == num_omega*2 || (ret == 0 && feof(file)));
		}
		else
		{
			for(int i = 0;i < num_omega*2;i++)
			{
				assert(!feof(file));
				double temp[6];
				//assert(1 == fscanf(file,"%*g %lg %lg %lg\n",&g0_up[i],&g0_down[i],&f0[i]));
				assert(6 == fscanf(file,"%*g %lg %lg %lg %lg %lg %lg\n",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4],&temp[5]));
                //g0_up[num_omega-i-1] = temp[0] - I*temp[1];
				//g0_up[num_omega+i] = temp[0] + I*temp[1];
				//g0_down[num_omega-i-1] = temp[2] - I*temp[3];
				//g0_down[num_omega+i] = temp[2] + I*temp[3];
				//f0[num_omega-i-1] = temp[4] - I*temp[5];
				//f0[num_omega+i] = temp[4] + I*temp[5];
                //
                g0_up[i] = temp[0] + I*temp[1];
                g0_down[i] = temp[2] + I*temp[3];
                f0[i] = temp[4] + I*temp[5];
			}
		}

        // A check here, to make sure the up and down are real (in 
        // imaginary time)
        for(int i = 0;i < num_omega;i++)
        {
			//assert(g0_up[num_omega-i-1] == conj(g0_up[num_omega+i]));
            //assert(g0_down[num_omega-i-1] == conj(g0_down[num_omega+i]));
            //f0[num_omega-i-1] = conj(f0[num_omega+i]);
        }
	}
    LOGDONE();

	cd * G_up = new cd[num_omega*2];
	cd * G_down = new cd[num_omega*2];
	cd * F = new cd[num_omega*2];

    cd * Gw_up = new cd[num_omega*2];
    cd * Gw_down = new cd[num_omega*2];
    cd * Fw = new cd[num_omega*2];

	//for(int i = 0;i < num_omega;i++)
	//	printf("%d %lg %lg\n",i,real(g0_up[i]),imag(g0_up[i]));


	// Size of grid to interpolate with.
	int Size = 20000;
    WC * ImpSol=new WC(g0_up, g0_down, f0, U, Size, num_omega*2,num_sweeps);
	// Set the K value
	ImpSol->K = K;
	printf("K is %g\n",K);
    ImpSol->NIgreen();
    ImpSol->sampling_proc();
    ImpSol->save_and_det_Greentau(G_up, G_down, F, 0,Gw_up,Gw_down,Fw);
    delete ImpSol;

	printf("Writing data\n");
    if(strcmp(output_filename,"-") == 0)
        file = stdout;
    else
    {
        file = fopen(output_filename,"wb");
        assert(file != NULL);
    }

	if(binary_files)
	{
		// Because of bad programming, need to check the sizes.
		assert(sizeof(double) == 8);
		assert(sizeof(cd) == 16);
		fwrite(&ImpSol->pert_order,sizeof(double),1,file);
        fwrite(&ImpSol->Sign,sizeof(cd),1,file);
		fwrite(G_up,sizeof(cd),num_omega*2,file);
		fwrite(G_down,sizeof(cd),num_omega*2,file);
		fwrite(F,sizeof(cd),num_omega*2,file);
		fwrite(Gw_up,sizeof(cd),num_omega*2,file);
		fwrite(Gw_down,sizeof(cd),num_omega*2,file);
		fwrite(Fw,sizeof(cd),num_omega*2,file);
	}
	else
	{
		fprintf(file,"Pert_order %.16lg\n",ImpSol->pert_order);
		fprintf(file,"Sign %.16lg %.16lg\n",real(ImpSol->Sign),imag(ImpSol->Sign));
        //for(int i = num_omega;i < 2*num_omega;i++)
        for(int i = 0;i < 2*num_omega;i++)
			fprintf(file,"%d %.16lg %.16lg %.16lg %.16lg %.16lg %.16lg\n",i-num_omega,real(G_up[i]),imag(G_up[i]),real(G_down[i]),imag(G_down[i]),real(F[i]),imag(F[i]));
	}

	fclose(file);

	delete[] G_up;
	delete[] G_down;
	delete[] F;
    delete[] Gw_up;
    delete[] Gw_down;
    delete[] Fw;

	return EXIT_SUCCESS;
}
