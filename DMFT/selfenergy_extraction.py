# This the analysis function to extract the self-energy
# at zero frequency and the infinite frequency (large frequency)
# when self-energy is flat
# first column is the staggering potential, second column is interpolated dynamical self-energy at w=0 and third Hartree Fock self-energy. The value is written in selfenergy.dat for given U value 

import numpy as np
import shutil
import os
from numpy import load
import RDMFT_TISM
data=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.00_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.00_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x = info['lambda_x']
os.remove('info.dat')

data1=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.25_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.25_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)

info = RDMFT_TISM.ReadInfo()
lambda_x1 = info['lambda_x']
os.remove('info.dat')

data2=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.5_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.5_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)

info = RDMFT_TISM.ReadInfo()
lambda_x2 = info['lambda_x']
os.remove('info.dat')

data3=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.75_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_0.75_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x3 = info['lambda_x']
os.remove('info.dat')

data4=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.0_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.0_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x4 = info['lambda_x']
os.remove('info.dat')

data5=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.25_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.25_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x5 = info['lambda_x']
os.remove('info.dat')

data6=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.5_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.5_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x6 = info['lambda_x']
os.remove('info.dat')

data7=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.75_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_1.75_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x7 = info['lambda_x']
os.remove('info.dat')

data8=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.00_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.00_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x8 = info['lambda_x']
os.remove('info.dat')

data9=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.25_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.25_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x9 = info['lambda_x']
os.remove('info.dat')

data10=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.50_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.50_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x10 = info['lambda_x']
os.remove('info.dat')

data11=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.75_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_2.75_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x11 = info['lambda_x']
os.remove('info.dat')

data12=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.00_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.00_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x12 = info['lambda_x']
os.remove('info.dat')

data13=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.25_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.25_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x13 = info['lambda_x']
os.remove('info.dat')

data14=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.50_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.50_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x14 = info['lambda_x']
os.remove('info.dat')

data15=load('/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.75_gama_0.25_OBC/everything_iter29.npz')

src = '/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.25/U_4.0/lambda_3.75_gama_0.25_OBC/info.dat'
dst = 'info.dat'
shutil.copyfile(src, dst)
info = RDMFT_TISM.ReadInfo()
lambda_x15 = info['lambda_x']
os.remove('info.dat')

SE=data['SE']
SE1=data1['SE']
SE2=data2['SE']
SE3=data3['SE']
SE4=data4['SE']
SE5=data5['SE']
SE6=data6['SE']
SE7=data7['SE']
SE8=data8['SE']
SE9=data9['SE']
SE10=data10['SE']
SE11=data11['SE']
SE12=data12['SE']
SE13=data13['SE']
SE14=data14['SE']
SE15=data15['SE']

#print 'The self-energy at zero omega'

SEw = abs(SE[0,400,0].real)
SEw1 = abs(SE1[0,400,0].real)
SEw2 = abs(SE2[0,400,0].real)
SEw3 = abs(SE3[0,400,0].real)
SEw4 = abs(SE4[0,400,0].real)
SEw5 = abs(SE5[0,400,0].real)
SEw6 = abs(SE6[0,400,0].real)
SEw7 = abs(SE7[0,400,0].real)
SEw8 = abs(SE8[0,400,0].real)
SEw9 = abs(SE9[0,400,0].real)
SEw10= abs(SE10[0,400,0].real)
SEw11= abs(SE11[0,400,0].real)
SEw12= abs(SE12[0,400,0].real)
SEw13= abs(SE13[0,400,0].real)
SEw14= abs(SE14[0,400,0].real)
SEw15= abs(SE15[0,400,0].real)

SE_w=np.vstack([SEw, SEw1, SEw2, SEw3, SEw4, SEw5, SEw6, SEw7, SEw8, SEw9, SEw10, SEw11, SEw12, SEw13, SEw14, SEw15])
#print 'The self-energy at infinite omega'
SE0 = abs(SE[0,1,0].real)
SE01 = abs(SE1[0,1,0].real)
SE02 = abs(SE2[0,1,0].real)
SE03 = abs(SE3[0,1,0].real)
SE04 = abs(SE4[0,1,0].real)
SE05 = abs(SE5[0,1,0].real)
SE06 = abs(SE6[0,1,0].real)
SE07 = abs(SE7[0,1,0].real)
SE08 = abs(SE8[0,1,0].real)
SE09 = abs(SE9[0,1,0].real)
SE010= abs(SE10[0,1,0].real)
SE011= abs(SE11[0,1,0].real)
SE012= abs(SE12[0,1,0].real)
SE013= abs(SE13[0,1,0].real)
SE014= abs(SE14[0,1,0].real)
SE015= abs(SE15[0,1,0].real)

stag_pot = np.vstack([lambda_x, lambda_x1, lambda_x2, lambda_x3, lambda_x4, lambda_x5, lambda_x6, lambda_x7, lambda_x8, lambda_x9, lambda_x10, lambda_x11, lambda_x12, lambda_x13, lambda_x14, lambda_x15])
SE_whf=np.vstack([SE0, SE01, SE02, SE03, SE04, SE05, SE06, SE07, SE08, SE09, SE010, SE011, SE012, SE013, SE014, SE015])

SE_tot = np.concatenate((stag_pot, SE_w, SE_whf), axis=1)

#print SE_tot
stag_pot = SE_tot[:,0]
SE_w=SE_tot[:,1]
SE_whf=SE_tot[:,2]

#Following can be used with updated version of python
#np.savetxt('selfenergy.dat', np.c_[stag_pot, SE_ww, SE_www], fmt = '%1.3f', header="lambda_x, SE_w, SE_w0" ,delimiter= ' ')

np.savetxt('selfenergy.dat', np.c_[stag_pot, SE_w, SE_whf], fmt = '%1.3f' ,delimiter= ' ')


