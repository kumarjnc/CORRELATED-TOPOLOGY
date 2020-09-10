/*! \file
 *
 * \brief Spline interpolation routines, using python.
 *
 */

/*! A convenience function for creating a safe temporary file.
 *
 * By default, the function will use the location "/tmp". However, if the 
 * TMPDIR environment variable is set, then this will be used. A file in this 
 * location will then be created, using "template_name" as the filename.  
 * "template_name" should contain "XXXXXX", such that the function mkstemp() 
 * can properly function.
 */
char * CreateTempFile(const char * template_name)
{
	// The template_name here must have XXXXXX within, for mkstemp to work.
	// The returned filename should be delete[]ed when finished with.
	char * tmp_filename = new char[1024];
	tmp_filename[0] = '\0';
	strcat(tmp_filename,"/tmp");

	// Find the appropriate dir (use TMPDIR if it is set)
	char ** ptr = environ;
	while(*ptr != NULL)
	{
		if(strncmp(*ptr,"TMPDIR=",7) == 0)
		{
			strcpy(tmp_filename,(*ptr)+7);
			break;
		}
		ptr++;
	}
    strcat(tmp_filename,"/");
	strcat(tmp_filename,template_name);

    int fd = mkstemp(tmp_filename);
    assert(fd != -1);
    close(fd);

	return tmp_filename;
}

/*! This function performs a spline interpolation, using the python class 
 * UnivariateSpline.
 *
 * This function was written to better handle the derivatives at the endpoints 
 * of the imaginary time Green's functions.
 */
void PythonSplines(double * x_in, double y_in[], double x_out[], double y_out[], int Size, int n)
{
    // Use a tmp file for passing the info output
    char * input_filename = CreateTempFile("input_XXXXXX");
    char * output_filename = CreateTempFile("output_XXXXXX");
    //char input_filename[] = "/home/cocks/asdf_input";
    //char output_filename[] = "/home/cocks/asdf_output";

	// TODO: This is dodgy - need to fix (with proper C implementation)
	FILE * file = fopen(input_filename,"wb");
	assert(fwrite(&Size,sizeof(int),1,file) == 1);
	assert(fwrite(&n,sizeof(int),1,file) == 1);
	assert(fwrite(x_in,sizeof(double),Size,file) == (unsigned int)(Size));
	assert(fwrite(y_in,sizeof(double),Size,file) == (unsigned int)(Size));
	assert(fwrite(x_out,sizeof(double),n,file) == (unsigned int)(n));
	fclose(file);

	char python_code[5000] = "";
	strcat(python_code,"python -c 'from scipy.interpolate import UnivariateSpline; from numpy import fromfile\n");
	strcat(python_code,"file = open(\""); strcat(python_code,input_filename); strcat(python_code,"\")\n");
	strcat(python_code,"Size = fromfile(file,\"int32\",1)\n"); 
	strcat(python_code,"n = fromfile(file,\"int32\",1)\n"); 
	strcat(python_code,"x_in = fromfile(file,\"float64\",Size)\n"); 
	strcat(python_code,"y_in = fromfile(file,\"float64\",Size)\n"); 
	strcat(python_code,"x_out = fromfile(file,\"float64\",n)\n"); 
	strcat(python_code,"y_out = UnivariateSpline(x_in,y_in,s=0)(x_out)\n");
	strcat(python_code,"file.close()\n");
	strcat(python_code,"y_out.tofile(\""); strcat(python_code,output_filename); strcat(python_code,"\")\n");
	strcat(python_code,"'");

	assert(system(python_code) == 0);

	file = fopen(output_filename,"rb");
	assert(fread(y_out,sizeof(double),n,file) == (unsigned int)(n));
	fclose(file);

	unlink(input_filename);
	unlink(output_filename);

	delete[] input_filename;
	delete[] output_filename;
}

/*! An interpolation as for PythonSplines(), but allowing complex functions.
 */ 
void PythonSplinesComplex(double * x_in, cd y_in[], double x_out[], cd y_out[], int Size, int n)
{
	// This function just applies the real spline interpolation to both the 
	// real and imaginary parts separately.
	
	double * temp_in = new double[Size];
	double * temp_out = new double[n];

	// Real parts first
	for(int i = 0;i < Size;i++)
		temp_in[i] = real(y_in[i]);

	PythonSplines(x_in,temp_in,x_out,temp_out,Size,n);

	for(int i = 0;i < n;i++)
		y_out[i] = temp_out[i];

	// Imag parts
	for(int i = 0;i < Size;i++)
		temp_in[i] = imag(y_in[i]);

	PythonSplines(x_in,temp_in,x_out,temp_out,Size,n);

	cd i_imag = cd(0,1);
	for(int i = 0;i < n;i++)
		y_out[i] = y_out[i] + i_imag*temp_out[i];
}
