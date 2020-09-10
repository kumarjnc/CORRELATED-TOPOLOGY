#This function reads the data of outcome of DMFT+CTAUX and defines the staggered occupancy
# according to as mentioned in the draft.
# n_s = n_{\lambda_x}- n_{-\lambda_x}
# where n_{\lambda_x}= n_{\uparrow \lambda_x}+n_{\downarrow \lambda_x}


import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import figure,imshow,colorbar,title,savefig,quiver
from pylab import cm
from pylab import *
from numpy import load


data1=load('U_6.0/lambda_.25/data.npz')

data2=load('U_6.0/lambda_.50/data.npz')

data3=load('U_6.0/lambda_.75/data.npz')

data4=load('U_6.0/lambda_1.00/data.npz')

data5=load('U_6.0/lambda_1.25/data.npz')

data6=load('U_6.0/lambda_1.50/data.npz')

data7=load('U_6.0/lambda_1.75/data.npz')

data8=load('U_6.0/lambda_2.00/data.npz')

data9=load('U_6.0/lambda_2.10/data.npz')

data10=load('U_6.0/lambda_2.20/data.npz')

data11=load('U_6.0/lambda_2.25/data.npz')

data12=load('U_6.0/lambda_2.30/data.npz')

data13=load('U_6.0/lambda_2.40/data.npz')

data14=load('U_6.0/lambda_2.50/data.npz')

data15=load('U_6.0/lambda_2.60/data.npz')

data16=load('U_6.0/lambda_2.75/data.npz')

data17=load('U_6.0/lambda_2.80/data.npz')

data18=load('U_6.0/lambda_2.90/data.npz')

data19=load('U_6.0/lambda_3.00/data.npz')


print(data1.files)
print(data2.files)
print(data3.files)
print(data4.files)
print(data5.files)
print(data6.files)
print(data7.files)
print(data8.files)
print(data9.files)
print(data10.files)
print(data11.files)
print(data12.files)
print(data13.files)
print(data14.files)
print(data15.files)
print(data16.files)
print(data17.files)
print(data18.files)
print(data19.files)

SE1=data1['N_dbl']
SE2=data2['N_dbl']
SE3=data3['N_dbl']
SE4=data4['N_dbl']
SE5=data5['N_dbl']
SE6=data6['N_dbl']
SE7=data7['N_dbl']
SE8=data8['N_dbl']
SE9=data9['N_dbl']
SE10=data10['N_dbl']
SE11=data11['N_dbl']
SE12=data12['N_dbl']
SE13=data13['N_dbl']
SE14=data14['N_dbl']
SE15=data15['N_dbl']
SE16=data16['N_dbl']
SE17=data17['N_dbl']
SE18=data18['N_dbl']
SE19=data19['N_dbl']

Nup1=data1['N_up']
Nup2=data2['N_up']
Nup3=data3['N_up']
Nup4=data4['N_up']
Nup5=data5['N_up']
Nup6=data6['N_up']
Nup7=data7['N_up']
Nup8=data8['N_up']
Nup9=data9['N_up']
Nup10=data10['N_up']
Nup11=data11['N_up']
Nup12=data12['N_up']
Nup13=data13['N_up']
Nup14=data14['N_up']
Nup15=data15['N_up']
Nup16=data16['N_up']
Nup17=data17['N_up']
Nup18=data18['N_up']
Nup19=data19['N_up']

Ndw1=data1['N_down']
Ndw2=data2['N_down']
Ndw3=data3['N_down']
Ndw4=data4['N_down']
Ndw5=data5['N_down']
Ndw6=data6['N_down']
Ndw7=data7['N_down']
Ndw8=data8['N_down']
Ndw9=data9['N_down']
Ndw10=data10['N_down']
Ndw11=data11['N_down']
Ndw12=data12['N_down']
Ndw13=data13['N_down']
Ndw14=data14['N_down']
Ndw15=data15['N_down']
Ndw16=data16['N_down']
Ndw17=data17['N_down']
Ndw18=data18['N_down']
Ndw19=data19['N_down']

print 'this the full self-energy contribution'
print SE1
print SE2
print SE3
print SE4
print SE5
print SE6
print SE7
print SE8
print SE9
print SE10
print SE11
print SE12

U=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.10, 2.20, 2.25, 2.30, 2.40, 2.50, 2.60, 2.75, 2.80, 2.90, 3.00]

print U
double=np.vstack([SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8, SE9, SE10, SE11, SE12, SE13, SE14, SE15, SE16, SE17, SE18, SE19])


Nup=np.vstack([Nup1, Nup2, Nup3, Nup4, Nup5, Nup6, Nup7, Nup8, Nup9, Nup10, Nup11, Nup12, Nup13, Nup14, Nup15, Nup16, Nup17, Nup18, Nup19])

Ndw=np.vstack([Ndw1, Ndw2, Ndw3, Ndw4, Ndw5, Ndw6, Ndw7, Ndw8, Ndw9, Ndw10, Ndw11, Ndw12, Ndw13, Ndw14, Ndw15, Ndw16, Ndw17, Ndw18, Ndw19])

print Nup.shape
rc('axes', linewidth=2.5)
#plt.plot(double[:,1],'ro-',lw=2.5)

#plt.plot(double[:,0],'ro-',lw=2.5)


#plt.plot(Nup[:,0]+Ndw[:,0],'bo-',lw=2.5)
#plt.plot(Nup[:,1]+Ndw[:,1],'bo-',lw=2.5)

#plt.plot(abs(Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
#plt.plot(abs(Nup[:,1]-Ndw[:,1]),'go-',lw=2.5)
U6, = plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'rs-',label='U6',lw=2.5)

#plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
print 'done'

data1=load('U_4.0/lambda_0.25_gama_0.0_OBC/data.npz')

data2=load('U_4.0/lambda_0.50_gama_0.0_OBC/data.npz')

data3=load('U_4.0/lambda_0.75_gama_0.0_OBC/data.npz')

data4=load('U_4.0/lambda_0.90/data.npz')
data5=load('U_4.0/lambda_1.00_gama_0.0_OBC/data.npz')
data6=load('U_4.0/lambda_1.10/data.npz')

data7=load('U_4.0/lambda_1.20/data.npz')

data8=load('U_4.0/lambda_1.25_gama_0.0_OBC/data.npz')

data9=load('U_4.0/lambda_1.30/data.npz')

data10=load('U_4.0/lambda_1.40/data.npz')

data11=load('U_4.0/lambda_1.50_gama_0.0_OBC/data.npz')

data12=load('U_4.0/lambda_1.60/data.npz')

data13=load('U_4.0/lambda_1.70/data.npz')

data14=load('U_4.0/lambda_1.75_gama_0.0_OBC/data.npz')

data15=load('U_4.0/lambda_1.80/data.npz')

data16=load('U_4.0/lambda_2.00_gama_0.0_OBC/data.npz')

data17=load('U_4.0/lambda_2.25_gama_0.0_OBC/data.npz')

data18=load('U_4.0/lambda_2.50_gama_0.0_OBC/data.npz')

data19=load('U_4.0/lambda_2.75_gama_0.0_OBC/data.npz')

data20=load('U_4.0/lambda_3.00/data.npz')


print(data1.files)
print(data2.files)
print(data3.files)
print(data4.files)
print(data5.files)
print(data6.files)
print(data7.files)
print(data8.files)
print(data9.files)
print(data10.files)
print(data11.files)
print(data12.files)
print(data13.files)
print(data14.files)
print(data15.files)
print(data16.files)
print(data17.files)
print(data18.files)
print(data19.files)
print(data20.files)

SE1=data1['N_dbl']
SE2=data2['N_dbl']
SE3=data3['N_dbl']
SE4=data4['N_dbl']
SE5=data5['N_dbl']
SE6=data6['N_dbl']
SE7=data7['N_dbl']
SE8=data8['N_dbl']
SE9=data9['N_dbl']
SE10=data10['N_dbl']
SE11=data11['N_dbl']
SE12=data12['N_dbl']
SE13=data13['N_dbl']
SE14=data14['N_dbl']
SE15=data15['N_dbl']
SE16=data16['N_dbl']
SE17=data17['N_dbl']
SE18=data18['N_dbl']
SE19=data19['N_dbl']
SE20=data20['N_dbl']

Nup1=data1['N_up']
Nup2=data2['N_up']
Nup3=data3['N_up']
Nup4=data4['N_up']
Nup5=data5['N_up']
Nup6=data6['N_up']
Nup7=data7['N_up']
Nup8=data8['N_up']
Nup9=data9['N_up']
Nup10=data10['N_up']
Nup11=data11['N_up']
Nup12=data12['N_up']
Nup13=data13['N_up']
Nup14=data14['N_up']
Nup15=data15['N_up']
Nup16=data16['N_up']
Nup17=data17['N_up']
Nup18=data18['N_up']
Nup19=data19['N_up']
Nup20=data20['N_up']

Ndw1=data1['N_down']
Ndw2=data2['N_down']
Ndw3=data3['N_down']
Ndw4=data4['N_down']
Ndw5=data5['N_down']
Ndw6=data6['N_down']
Ndw7=data7['N_down']
Ndw8=data8['N_down']
Ndw9=data9['N_down']
Ndw10=data10['N_down']
Ndw11=data11['N_down']
Ndw12=data12['N_down']
Ndw13=data13['N_down']
Ndw14=data14['N_down']
Ndw15=data15['N_down']
Ndw16=data16['N_down']
Ndw17=data17['N_down']
Ndw18=data18['N_down']
Ndw19=data19['N_down']
Ndw20=data20['N_down']
print 'this the full self-energy contribution'
print SE1
print SE2
print SE3
print SE4
print SE5
print SE6
print SE7
print SE8
print SE9
print SE10
print SE11
print SE12

U=[0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.70, 1.75, 1.80, 2.00, 2.25, 2.50, 2.75, 3.00]

double=np.vstack([SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8, SE9, SE10, SE11, SE12, SE13, SE14, SE15, SE16, SE17, SE18, SE19, SE20])


Nup=np.vstack([Nup1, Nup2, Nup3, Nup4, Nup5, Nup6, Nup7, Nup8, Nup9, Nup10, Nup11, Nup12, Nup13, Nup14, Nup15, Nup16, Nup17, Nup18, Nup19, Nup20])

Ndw=np.vstack([Ndw1, Ndw2, Ndw3, Ndw4, Ndw5, Ndw6, Ndw7, Ndw8, Ndw9, Ndw10, Ndw11, Ndw12, Ndw13, Ndw14, Ndw15, Ndw16, Ndw17, Ndw18, Ndw19, Ndw20])

print Nup.shape

#plt.plot(double[:,1],'ro-',lw=2.5)

#plt.plot(double[:,0],'ro-',lw=2.5)


#plt.plot(Nup[:,0]+Ndw[:,0],'bo-',lw=2.5)
#plt.plot(Nup[:,1]+Ndw[:,1],'bo-',lw=2.5)

#plt.plot(abs(Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
#plt.plot(abs(Nup[:,1]-Ndw[:,1]),'go-',lw=2.5)
U4, = plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'gs-',label='U4',lw=2.5)

#plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import figure,imshow,colorbar,title,savefig,quiver
from pylab import cm

from numpy import load
'''
data1=load('U_1.0/lambda_0.25_gama_0.0_OBC/data.npz')

data2=load('U_1.0/lambda_0.50_gama_0.0_OBC/data.npz')

data3=load('U_1.0/lambda_0.75_gama_0.0_OBC/data.npz')

data4=load('U_1.0/lambda_1.00_gama_0.0_OBC/data.npz')

data5=load('U_1.0/lambda_1.25_gama_0.0_OBC/data.npz')

data6=load('U_1.0/lambda_1.50_gama_0.0_OBC/data.npz')

data7=load('U_1.0/lambda_1.75_gama_0.0_OBC/data.npz')

data8=load('U_1.0/lambda_2.00_gama_0.0_OBC/data.npz')

data9=load('U_1.0/lambda_2.25_gama_0.0_OBC/data.npz')

data10=load('U_1.0/lambda_2.50_gama_0.0_OBC/data.npz')

data11=load('U_1.0/lambda_2.75_gama_0.0_OBC/data.npz')

data12=load('U_1.0/lambda_3.00/data.npz')


print(data1.files)
print(data2.files)
print(data3.files)
print(data4.files)
print(data5.files)
print(data6.files)
print(data7.files)
print(data8.files)
print(data9.files)
print(data10.files)
print(data11.files)
print(data12.files)

SE1=data1['N_dbl']
SE2=data2['N_dbl']
SE3=data3['N_dbl']
SE4=data4['N_dbl']
SE5=data5['N_dbl']
SE6=data6['N_dbl']
SE7=data7['N_dbl']
SE8=data8['N_dbl']
SE9=data9['N_dbl']
SE10=data10['N_dbl']
SE11=data11['N_dbl']
SE12=data12['N_dbl']

Nup1=data1['N_up']
Nup2=data2['N_up']
Nup3=data3['N_up']
Nup4=data4['N_up']
Nup5=data5['N_up']
Nup6=data6['N_up']
Nup7=data7['N_up']
Nup8=data8['N_up']
Nup9=data9['N_up']
Nup10=data10['N_up']
Nup11=data11['N_up']
Nup12=data12['N_up']

Ndw1=data1['N_down']
Ndw2=data2['N_down']
Ndw3=data3['N_down']
Ndw4=data4['N_down']
Ndw5=data5['N_down']
Ndw6=data6['N_down']
Ndw7=data7['N_down']
Ndw8=data8['N_down']
Ndw9=data9['N_down']
Ndw10=data10['N_down']
Ndw11=data11['N_down']
Ndw12=data12['N_down']
print 'this the full self-energy contribution'
print SE1
print SE2
print SE3
print SE4
print SE5
print SE6
print SE7
print SE8
print SE9
print SE10
print SE11
print SE12
U=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
double=np.vstack([SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8, SE9, SE10, SE11, SE12])


Nup=np.vstack([Nup1, Nup2, Nup3, Nup4, Nup5, Nup6, Nup7, Nup8, Nup9, Nup10, Nup11, Nup12])

Ndw=np.vstack([Ndw1, Ndw2, Ndw3, Ndw4, Ndw5, Ndw6, Ndw7, Ndw8, Ndw9, Ndw10, Ndw11, Ndw12])

print Nup.shape

#plt.plot(double[:,1],'ro-',lw=2.5)

#plt.plot(double[:,0],'ro-',lw=2.5)


#plt.plot(Nup[:,0]+Ndw[:,0],'bo-',lw=2.5)
#plt.plot(Nup[:,1]+Ndw[:,1],'bo-',lw=2.5)

#plt.plot(abs(Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
#plt.plot(abs(Nup[:,1]-Ndw[:,1]),'go-',lw=2.5)

U1, = plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'bs-',label='U1',lw=2.5)

#plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)

data1=load('U_0.5/lambda_0.25_gama_0.0_OBC/data.npz')

data2=load('U_0.5/lambda_0.50_gama_0.0_OBC/data.npz')

data3=load('U_0.5/lambda_0.75_gama_0.0_OBC/data.npz')

data4=load('U_0.5/lambda_1.00_gama_0.0_OBC/data.npz')

data5=load('U_0.5/lambda_1.25_gama_0.0_OBC/data.npz')

data6=load('U_0.5/lambda_1.50_gama_0.0_OBC/data.npz')

data7=load('U_0.5/lambda_1.75_gama_0.0_OBC/data.npz')

data8=load('U_0.5/lambda_2.00_gama_0.0_OBC/data.npz')

data9=load('U_0.5/lambda_2.25_gama_0.0_OBC/data.npz')

data10=load('U_0.5/lambda_2.50_gama_0.0_OBC/data.npz')

data11=load('U_0.5/lambda_2.75_gama_0.0_OBC/data.npz')

data12=load('U_0.5/lambda_3.00_gama_0.0_OBC/data.npz')


print(data1.files)
print(data2.files)
print(data3.files)
print(data4.files)
print(data5.files)
print(data6.files)
print(data7.files)
print(data8.files)
print(data9.files)
print(data10.files)
print(data11.files)
print(data12.files)

SE1=data1['N_dbl']
SE2=data2['N_dbl']
SE3=data3['N_dbl']
SE4=data4['N_dbl']
SE5=data5['N_dbl']
SE6=data6['N_dbl']
SE7=data7['N_dbl']
SE8=data8['N_dbl']
SE9=data9['N_dbl']
SE10=data10['N_dbl']
SE11=data11['N_dbl']
SE12=data12['N_dbl']

Nup1=data1['N_up']
Nup2=data2['N_up']
Nup3=data3['N_up']
Nup4=data4['N_up']
Nup5=data5['N_up']
Nup6=data6['N_up']
Nup7=data7['N_up']
Nup8=data8['N_up']
Nup9=data9['N_up']
Nup10=data10['N_up']
Nup11=data11['N_up']
Nup12=data12['N_up']

Ndw1=data1['N_down']
Ndw2=data2['N_down']
Ndw3=data3['N_down']
Ndw4=data4['N_down']
Ndw5=data5['N_down']
Ndw6=data6['N_down']
Ndw7=data7['N_down']
Ndw8=data8['N_down']
Ndw9=data9['N_down']
Ndw10=data10['N_down']
Ndw11=data11['N_down']
Ndw12=data12['N_down']
print 'this the full self-energy contribution'
print SE1
print SE2
print SE3
print SE4
print SE5
print SE6
print SE7
print SE8
print SE9
print SE10
print SE11
print SE12
U=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]

double=np.vstack([SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8, SE9, SE10, SE11, SE12])


Nup=np.vstack([Nup1, Nup2, Nup3, Nup4, Nup5, Nup6, Nup7, Nup8, Nup9, Nup10, Nup11, Nup12])

Ndw=np.vstack([Ndw1, Ndw2, Ndw3, Ndw4, Ndw5, Ndw6, Ndw7, Ndw8, Ndw9, Ndw10, Ndw11, Ndw12])

print Nup.shape

#plt.plot(double[:,1],'ro-',lw=2.5)

#plt.plot(double[:,0],'ro-',lw=2.5)


#plt.plot(Nup[:,0]+Ndw[:,0],'bo-',lw=2.5)
#plt.plot(Nup[:,1]+Ndw[:,1],'bo-',lw=2.5)

#plt.plot(abs(Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
#plt.plot(abs(Nup[:,1]-Ndw[:,1]),'go-',lw=2.5)

U05, = plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'cs-',label='U05',lw=2.5)

#plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)

data1=load('U_5.0/lambda_0.25_gama_0.0_OBC/data.npz')

data2=load('U_5.0/lambda_0.50_gama_0.0_OBC/data.npz')

data3=load('U_5.0/lambda_0.75_gama_0.0_OBC/data.npz')

data4=load('U_5.0/lambda_1.00_gama_0.0_OBC/data.npz')

data5=load('U_5.0/lambda_1.25_gama_0.0_OBC/data.npz')

data6=load('U_5.0/lambda_1.40/data.npz')

data7=load('U_5.0/lambda_1.50_gama_0.0_OBC/data.npz')

data8=load('U_5.0/lambda_1.60/data.npz')

data9=load('U_5.0/lambda_1.70/data.npz')
data10=load('U_5.0/lambda_1.75_gama_0.0_OBC/data.npz')

data11=load('U_5.0/lambda_1.80/data.npz')

data12=load('U_5.0/lambda_1.90/data.npz')

data13=load('U_5.0/lambda_2.00_gama_0.0_OBC/data.npz')

data14=load('U_5.0/lambda_2.10/data.npz')

data15=load('U_5.0/lambda_2.20/data.npz')

data16=load('U_5.0/lambda_2.25_gama_0.0_OBC/data.npz')

data17=load('U_5.0/lambda_2.30/data.npz')

data18=load('U_5.0/lambda_2.40/data.npz')

data19=load('U_5.0/lambda_2.50_gama_0.0_OBC/data.npz')

data20=load('U_5.0/lambda_2.60/data.npz')

data21=load('U_5.0/lambda_2.70/data.npz')

data22=load('U_5.0/lambda_2.75_gama_0.0_OBC/data.npz')

data23=load('U_5.0/lambda_2.80/data.npz')

data24=load('U_5.0/lambda_3.00/data.npz')


print(data1.files)
print(data2.files)
print(data3.files)
print(data4.files)
print(data5.files)
print(data6.files)
print(data7.files)
print(data8.files)
print(data9.files)
print(data10.files)
print(data11.files)
print(data12.files)

SE1=data1['N_dbl']
SE2=data2['N_dbl']
SE3=data3['N_dbl']
SE4=data4['N_dbl']
SE5=data5['N_dbl']
SE6=data6['N_dbl']
SE7=data7['N_dbl']
SE8=data8['N_dbl']
SE9=data9['N_dbl']
SE10=data10['N_dbl']
SE11=data11['N_dbl']
SE12=data12['N_dbl']
SE13=data13['N_dbl']
SE14=data14['N_dbl']
SE15=data15['N_dbl']
SE16=data16['N_dbl']
SE17=data17['N_dbl']
SE18=data18['N_dbl']
SE19=data19['N_dbl']
SE20=data20['N_dbl']
SE21=data21['N_dbl']
SE22=data22['N_dbl']
SE23=data23['N_dbl']
SE24=data24['N_dbl']

Nup1=data1['N_up']
Nup2=data2['N_up']
Nup3=data3['N_up']
Nup4=data4['N_up']
Nup5=data5['N_up']
Nup6=data6['N_up']
Nup7=data7['N_up']
Nup8=data8['N_up']
Nup9=data9['N_up']
Nup10=data10['N_up']
Nup11=data11['N_up']
Nup12=data12['N_up']
Nup13=data13['N_up']
Nup14=data14['N_up']
Nup15=data15['N_up']
Nup16=data16['N_up']
Nup17=data17['N_up']
Nup18=data18['N_up']
Nup19=data19['N_up']
Nup20=data20['N_up']
Nup21=data21['N_up']
Nup22=data22['N_up']
Nup23=data23['N_up']
Nup24=data24['N_up']

Ndw1=data1['N_down']
Ndw2=data2['N_down']
Ndw3=data3['N_down']
Ndw4=data4['N_down']
Ndw5=data5['N_down']
Ndw6=data6['N_down']
Ndw7=data7['N_down']
Ndw8=data8['N_down']
Ndw9=data9['N_down']
Ndw10=data10['N_down']
Ndw11=data11['N_down']
Ndw12=data12['N_down']
Ndw13=data13['N_down']
Ndw14=data14['N_down']
Ndw15=data15['N_down']
Ndw16=data16['N_down']
Ndw17=data17['N_down']
Ndw18=data18['N_down']
Ndw19=data19['N_down']
Ndw20=data20['N_down']
Ndw21=data21['N_down']
Ndw22=data22['N_down']
Ndw23=data23['N_down']
Ndw24=data24['N_down']
print 'this the full self-energy contribution'
print SE1
print SE2
print SE3
print SE4
print SE5
print SE6
print SE7
print SE8
print SE9
print SE10
print SE11
print SE12

U=[0.25, 0.50, 0.75, 1.00, 1.25, 1.40, 1.50, 1.60, 1.70, 1.75, 1.80, 1.90, 2.00, 2.10, 2.20, 2.25, 2.30, 2.40, 2.50, 2.60, 2.70, 2.75, 2.80,3.00]
double=np.vstack([SE1, SE2, SE3, SE4, SE5, SE6, SE7, SE8, SE9, SE10, SE11, SE12, SE13, SE14, SE15, SE16, SE17, SE18, SE19, SE20, SE21, SE22, SE23, SE24])



Nup=np.vstack([Nup1, Nup2, Nup3, Nup4, Nup5, Nup6, Nup7, Nup8, Nup9, Nup10, Nup11, Nup12, Nup13, Nup14, Nup15, Nup16, Nup17, Nup18, Nup19, Nup20, Nup21, Nup22, Nup23, Nup24])

Ndw=np.vstack([Ndw1, Ndw2, Ndw3, Ndw4, Ndw5, Ndw6, Ndw7, Ndw8, Ndw9, Ndw10, Ndw11, Ndw12, Ndw13, Ndw14, Ndw15, Ndw16, Ndw17, Ndw18, Ndw19, Ndw20, Ndw21, Ndw22, Ndw23, Ndw24 ])

print Nup.shape

#plt.plot(double[:,1],'ro-',lw=2.5)

#plt.plot(double[:,0],'ro-',lw=2.5)


#plt.plot(Nup[:,0]+Ndw[:,0],'bo-',lw=2.5)
#plt.plot(Nup[:,1]+Ndw[:,1],'bo-',lw=2.5)

#plt.plot(abs(Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
#plt.plot(abs(Nup[:,1]-Ndw[:,1]),'go-',lw=2.5)

U5, = plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'ms-',label='U5',lw=2.5)

#plt.plot(U,abs(Nup[:,1]+Ndw[:,1]-Nup[:,0]-Ndw[:,0]),'go-',lw=2.5)
plt.legend([U05, U1, U4, U5, U6], ['U=0.5','U=1.0','U=4.0','U=5.0','U=6.0'],loc='upper left', fontsize=18)
plt.ylim(-0.1,1.8)
plt.xlim(0.25,3.00)
plt.ylabel(r'$n_s$', fontsize=30)
plt.xlabel(r'$\lambda_x/t$', fontsize=30)
#plt.grid(True)
plt.tight_layout()
#plt.show()

plt.savefig('stag_occ_v_f.png')

