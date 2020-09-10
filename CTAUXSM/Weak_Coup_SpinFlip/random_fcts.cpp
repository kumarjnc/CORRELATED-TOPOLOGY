/*! \file
 *
 * \brief Random number routines.
 *
 * This file contains some random number routines, as implemented in Numerical 
 * Recipes.
 *
 * The Numerical Recipes routine here was replaced by Danny for the glibc 
 * routine. This was done in the process of testing for bugs, and I found it to 
 * be more reliable in its even distribution of generated numbers.
 */

//long ran_seed=-(long)time(0);             // seed for random number generator
//double box_rand=num_recipes_ran1(ran_seed);         //rand_seed changed in every run
/* 
----------------------------------------------------------------------------------------------------*/
//random number generator ran1 from numerical recipes in c
//good for up to 100 000 000 random numbers, thereafter use extended version
//returns uniformly distributed numbers between 0 and 1
//initialize with negative idum, do not change within sequence

/*! The old Numerical Recipes random number generator, superceeded by 
 * num_recipes_ran1().
 */
double num_recipes_ran1_old(long idum[])
{
    const long IA=16807,IM=2147483647,IQ=127773,IR=2836,NTAB=32;
    const long NDIV=(1+(IM-1)/NTAB);
    const double EPS=3.0e-16,AM=1.0/IM,RNMX=(1.0-EPS);
    static long iy=0;
    static long iv[NTAB];
    int j;
    long k;
    double temp;

    if (idum[0] <= 0 || !iy) {
        if (-idum[0] < 1) idum[0]=1;
        else idum[0] = -idum[0];
        for (j=NTAB+7;j>=0;j--) {
            k=idum[0]/IQ;
            idum[0]=IA*(idum[0]-k*IQ)-IR*k;
            if (idum[0] < 0) idum[0] += IM;
            if (j < NTAB) iv[j] = idum[0];
        }
        iy=iv[0];
    }
    k=idum[0]/IQ;
    idum[0]=IA*(idum[0]-k*IQ)-IR*k;
    if (idum[0] < 0) idum[0] += IM;
    j=iy/NDIV;
    iy=iv[j];
    iv[j] = idum[0];
    if ((temp=AM*iy) > RNMX) return RNMX;
    else return temp;
}
/*! Replacement random generator using the glibc function.
 * 
 * Note that the argument is required as an artifact of the Numerical Recipes 
 * function. This argument should be a negative number in the first call of 
 * this function, at which time the random number generator will be seeded with 
 * the absolute value of that number.
 *
 * It is recommended to assign a value to this number using a truely random 
 * source, for example /dev/urandom (see WC::WC()). Note that using the current 
 * time is a bad choice, as multiprocessing can cause many runs of this code to 
 * use the same seed for the random number generator.
 */
#include <stdlib.h>
double num_recipes_ran1(long idum[])
{
	if(idum[0] <= 0)
	{
		srandom(-idum[0]);
		idum[0] = 1;
	}

	return double(random())/RAND_MAX;
}
//------------------------------------------------------------------
/*! Generate a random time for the configuration.
 */
double random_tau(long tau_seed[])
{return num_recipes_ran1(tau_seed);}
//------------------------------------------------------------------
/*! Generate a random up or down spin.
 */
int random_spin(long spin_seed[])
{
    int spin;
    double rd_spin=num_recipes_ran1(spin_seed);
    if(rd_spin<=0.5)
    {spin=0;}
    else
    {spin=1;}
    
    return spin;
}
//-------------------------------------------------------------------
/*! Choose a random spin from the current set of spins in the configuration.  
 *
 * This is equivalent to a random integer modulo n.
 */
int random_removal(long rem_seed[], int n)
{
    double rd_removal=num_recipes_ran1(rem_seed);
    int rd_removal_1=(int)(n*rd_removal);
    if (rd_removal_1==n)
    {rd_removal_1=0;}
    
    return rd_removal_1;
}
//--------------------------------------------------------------------
/*! Choose whether to accept or reject a configuration update, based on the 
 * propability prop.
 *
 * This is equivalent to asking the question x < prop, where x is a random 
 * number linearly distributed between 0 and 1.
 */
bool acceptance(long acc_seed[], double prop)
{
    bool acc;
    double rd_acc=num_recipes_ran1(acc_seed);
    if(rd_acc<=prop)
    {acc=true;}
    else 
    {acc=false;}
    return acc;
}
    
