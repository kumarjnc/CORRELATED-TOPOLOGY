
/*! \file
 *
 * \brief This file contains some spline interpolation routines, that (I 
 * believe) are from Numerical Recipes.
 *
 * The routines here have been replaced in the main program by Danny, because 
 * they suffered from some issues of derivative choice at the endpoints of the 
 * function.
 */

void spline_derivative(double y[], int n, double y2[])
{
    for (int i=1; i<n; i++){
        y2[i]=(y[i+1]+y[i-1]-2.0*y[i])*n*n;}
       y2[0]=y2[1]; 
       y2[n]=y2[n];
    //y2[0]=(y[1]-y[0])*n*n;
    //y2[n]=(y[n]-y[n-1])*n*n;	
}

//-------------------------------------------------------------------------------------
double splint(double xa[], double ya[], double y2a[], int n, double x)
{
	int klo,khi,k;
	double h,b,a;

	klo=0;
	khi=n-1;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	double y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return y;
}
//-------------------------------------------------------------------------------------
void splines(double x_in[], double y_in[], double y_prime[], double x_out[], double y_out[], int Size, int n)
{
    for (int i=0; i<n; i++){
        y_out[i]=splint(x_in, y_in, y_prime, Size, x_out[i]);}
}
//------------------------------------------------------------------------------------------
void linear(double x_in[], double y_in[], double x_out[], double y_out[], int n, int Size)
{
    double tau_1;
    double tau_2;
    int l;
    for (int i=0; i<n; i++)
    {
        l=(int)(x_out[i]*(Size-1));
        tau_1=x_in[l];
        if(tau_1<=x_out[i])
        {
            if(l==Size-1){y_out[i]=y_in[Size-1];}
            else{
            tau_2=x_in[l+1];
            y_out[i]=y_in[l]+(y_in[l+1]-y_in[l])*Size*(x_out[i]-x_in[l]);}
        }
        else
        {
            tau_2=x_in[l-1];
            y_out[i]=y_in[l]-(y_in[l+1]-y_in[l])*Size*(x_out[i]-x_in[l]);
        }
    }
}
//------------------------------------------------------------------------------
cd single_linear(double x_in[], cd y_in[], double x_out, int Size)
{
    double tau_1;
	cd y_out;
    if(x_out<0.0){
        x_out=1.0+x_out;
        int l=(int)(x_out*(Size-1));
        tau_1=x_in[l];
        
        if(tau_1<=x_out){y_out=-(y_in[l]+(y_in[l+1]-y_in[l])*double(Size)*(x_out-tau_1));}
        else{y_out=-(y_in[l]+(y_in[l]-y_in[l-1])*double(Size)*(x_out-tau_1));}}
    else{
        int l=(int)(x_out*(Size-1));
        tau_1=x_in[l];
        
        if(tau_1<=x_out){y_out=(y_in[l]+(y_in[l+1]-y_in[l])*double(Size)*(x_out-tau_1));}
        else{y_out=(y_in[l]+(y_in[l]-y_in[l-1])*double(Size)*(x_out-tau_1));}}
    return y_out;
}



