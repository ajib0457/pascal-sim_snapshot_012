import h5py
import numpy as np
from sympy import *
import pickle
from matplotlib import cm
import pandas as pd

#start_time = time.time()   
sim_sz=500000#kpc
grid_nodes=2000
#RECON_IMG CODE
#Symbolize variables and specify function
X,Y,Z,s=symbols('X Y Z s')
h=(1/sqrt(2*pi*s*s))**(3)*exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
#Toggle the next two lines to either implement my_den or DTFE density fields
#a=np.loadtxt('/import/oth3/ajib0457/my_den/density_grid/den_grid_%d_snapshot_012'%grid_nodes)#density field
#used to read binary file
#a=np.fromfile('/scratch/GAMNSCM2/DTFE_1.1.1_rep/DTFE_out_%d_snpsht_012_binary.den'%grid_nodes,count=grid_nodes**3,dtype=np.dtype(np.float32))
f1=h5py.File("/scratch/GAMNSCM2/DTFE_1.1.1_rep/bnry_h5/output_files/DTFE_out_%d_spsht_012_halos_mvir.a_den.h5" %grid_nodes, 'r')
a=f1['/DTFE'][:]
   # print("--- %s seconds ---" % (time.time() - start_time))
#a=np.loadtxt('../output_files/DTFE_out_300_snpsht_012.den')
#a=np.loadtxt('../data_files/test_1.den')
#a=np.loadtxt('../data_files/AbsemFile.txt')


#Toggle the next two lines to either implement my_den or DTFE density field
grid=a#for my density field, unhash line 15,21 and 22
del a
#grid=np.matrix.transpose(a)#this is essentially inverse transpose to make it flat again, done because it otherwise wouldn't save to file
image=np.reshape(grid,(grid_nodes,grid_nodes,grid_nodes))
#image=ndimage.filters.gaussian_filter(image,4)
del grid
img_x,img_y,img_z=np.shape(image)
#img_x,img_y,img_z=70,70,70
   
#take second partial derivatives as per Hessian 
hprimexx=h.diff(X,X)
#hprimexy=h.diff(X,Y)
#hprimeyy=h.diff(Y,Y)
#hprimezz=h.diff(Z,Z)
#hprimezx=h.diff(Z,X)
#hprimezy=h.diff(Z,Y)
#Lambdify i.e make them variables once again
fxx=lambdify((X,Y,Z,s),hprimexx,'numpy')
#fxy=lambdify((X,Y,Z,s),hprimexy,'numpy')
#fyy=lambdify((X,Y,Z,s),hprimeyy,'numpy')
#fzz=lambdify((X,Y,Z,s),hprimezz,'numpy')
#fzx=lambdify((X,Y,Z,s),hprimezx,'numpy')
#fzy=lambdify((X,Y,Z,s),hprimezy,'numpy')
#3d meshgrid for each kernel and evaluate 6 partial derivatives to generate 9 kernels
   
    
#Kernal settings
#b=0.5#strength of classification; lower than 0.5 will make filter more strict
#n=5#Change this value to increase/decrease smoothing scale
s=1.96 #standard deviation of kernal

kern_x,kern_y,kern_z=img_x,img_y,img_z #kernel size
in_val,fnl_val=-140,140  #value range of kernel 
#in_val,fnl_val=-35,35
#Kernel generator
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kern_y),np.linspace(in_val,fnl_val,kern_x),np.linspace(in_val,fnl_val,kern_z))
#X=X.astype(np.float32)       
#Y=Y.astype(np.float32)
#Z=Z.astype(np.float32)
dxx=fxx(X,Y,Z,s)
#dxy=fxy(X,Y,Z,s)
#dyy=fyy(X,Y,Z,s)
#dzz=fzz(X,Y,Z,s)
#dzx=fzx(X,Y,Z,s)
#dzy=fzy(X,Y,Z,s)
del X
del Y
del Z


##Kernel padding
#K_xx=np.zeros([img_x,img_y,img_z])
#K_xy=np.zeros([img_x,img_y,img_z])
#K_yy=np.zeros([img_x,img_y,img_z])
#K_zz=np.zeros([img_x,img_y,img_z])
#K_zx=np.zeros([img_x,img_y,img_z])
#K_zy=np.zeros([img_x,img_y,img_z])
##Imbed kernel into padding matrices
#K_xx[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dxx
#K_xy[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dxy
#K_yy[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dyy
#K_zz[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dzz
#K_zx[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dzx
#K_zy[int(img_x/2-kern_x/2):0+int(img_x/2+kern_x/2),int(img_y/2-kern_y/2):0+int(img_y/2+kern_y/2),int(img_z/2-kern_z/2):0+int(img_z/2+kern_z/2)]=dzy
##redefining dxx etc. so that they are now padded
#dxx=K_xx
#dxy=K_xy
#dyy=K_yy
#dzz=K_zz
#dzx=K_zx
#dzy=K_zy.


#shift kernel to 0,0
dxx=np.roll(dxx,int(img_x/2),axis=0)
dxx=np.roll(dxx,int(img_y/2),axis=1)
dxx=np.roll(dxx,int(img_z/2),axis=2)
#dxy=np.roll(dxy,int(img_x/2),axis=0)
#dxy=np.roll(dxy,int(img_y/2),axis=1)
#dxy=np.roll(dxy,int(img_z/2),axis=2)
#dyy=np.roll(dyy,int(img_x/2),axis=0)
#dyy=np.roll(dyy,int(img_y/2),axis=1)
#dyy=np.roll(dyy,int(img_z/2),axis=2)
#dzz=np.roll(dzz,int(img_x/2),axis=0)
#dzz=np.roll(dzz,int(img_y/2),axis=1)
#dzz=np.roll(dzz,int(img_z/2),axis=2)
#dzx=np.roll(dzx,int(img_x/2),axis=0)
#dzx=np.roll(dzx,int(img_y/2),axis=1)
#dzx=np.roll(dzx,int(img_z/2),axis=2)
#dzy=np.roll(dzy,int(img_x/2),axis=0)
#dzy=np.roll(dzy,int(img_y/2),axis=1)
#dzy=np.roll(dzy,int(img_z/2),axis=2)
#fft 6 kernels

fft_dxx=np.fft.fftn(dxx)
#fft_dxy=np.fft.fftn(dxy)
#fft_dyy=np.fft.fftn(dyy)
#fft_dzz=np.fft.fftn(dzz)
#fft_dzx=np.fft.fftn(dzx)
#fft_dzy=np.fft.fftn(dzy)
fft_db=np.fft.fftn(image)

del dxx
#del dxy
#del dyy
#del dzz
#del dzx
#del dzy
#convolution of kernels with density field & inverse transform
ifft_dxx=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real
#ifft_dxy=np.fft.ifftn(np.multiply(fft_dxy,fft_db)).real
#ifft_dyy=np.fft.ifftn(np.multiply(fft_dyy,fft_db)).real
#ifft_dzz=np.fft.ifftn(np.multiply(fft_dzz,fft_db)).real
#ifft_dzx=np.fft.ifftn(np.multiply(fft_dzx,fft_db)).real
#ifft_dzy=np.fft.ifftn(np.multiply(fft_dzy,fft_db)).real

del fft_dxx
#del fft_dxy
#del fft_dyy
#del fft_dzz
#del fft_dzx
#del fft_dzy
del fft_db    
#reshape into column matrices
ifft_dxx=np.reshape(ifft_dxx,(np.size(ifft_dxx),1))      
#ifft_dxy=np.reshape(ifft_dxy,(np.size(ifft_dxy),1))
#ifft_dyy=np.reshape(ifft_dyy,(np.size(ifft_dyy),1))
#ifft_dzz=np.reshape(ifft_dzz,(np.size(ifft_dzz),1))
#ifft_dzx=np.reshape(ifft_dzx,(np.size(ifft_dzx),1))
#ifft_dzy=np.reshape(ifft_dzy,(np.size(ifft_dzy),1))
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dxx.h5" %(grid_nodes,std_dev_phys), 'w')
#group=f.create_group('hessian_ingr')
f.create_dataset('/hessian_ingr/ifft_dxx',data=ifft_dxx)
f.close()
