import numpy as np
from sympy import *
import pickle
from matplotlib import cm
import pandas as pd
import h5py
#I should make 5 codes to calculate
#225GB+
part=0#Change for each file. START FROM 0

tot_parts=8
sim_sz=500000#kpc
in_val,fnl_val=-140,140
s=1.96
grid_nodes=2000
nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
grid_phys=sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f1=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dxx.h5" %(grid_nodes,std_dev_phys), 'r')
f2=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dxy.h5" %(grid_nodes,std_dev_phys), 'r')
f3=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dyy.h5" %(grid_nodes,std_dev_phys), 'r')
f4=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dzz.h5" %(grid_nodes,std_dev_phys), 'r')
f5=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dzx.h5" %(grid_nodes,std_dev_phys), 'r')
f6=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%skpc_ifft_dzy.h5" %(grid_nodes,std_dev_phys), 'r')
ifft_dxx=f1['/hessian_ingr/ifft_dxx'][nrows_in:nrows_fn]
ifft_dxy=f2['/hessian_ingr/ifft_dxy'][nrows_in:nrows_fn]
ifft_dyy=f3['/hessian_ingr/ifft_dyy'][nrows_in:nrows_fn]
ifft_dzz=f4['/hessian_ingr/ifft_dzz'][nrows_in:nrows_fn]
ifft_dzx=f5['/hessian_ingr/ifft_dzx'][nrows_in:nrows_fn]
ifft_dzy=f6['/hessian_ingr/ifft_dzy'][nrows_in:nrows_fn]

#grid_nodes is translated depending upon the rows called for hessian compilation
grid_nodes_true=int(round(np.power((nrows_fn-nrows_in),1.*1/3)))
hessian=np.column_stack((ifft_dxx,ifft_dxy,ifft_dzx,ifft_dxy,ifft_dyy,ifft_dzy,ifft_dzx,ifft_dzy,ifft_dzz))
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
del ifft_dxx
del ifft_dxy
del ifft_dyy
del ifft_dzz
del ifft_dzx
del ifft_dzy
#hessian=np.split(hessian,grid_nodes_true**3)
hessian=np.reshape(hessian,(grid_nodes_true**3,3,3))#change to 3,3 for 3d and 1,1 for 1d
#calculate eigenvalues and eigenvectors
eig_vals_vecs=np.linalg.eig(hessian)
del hessian
#create unique randomness to each eigenvalue    
#np.random.seed(1)
#unq_rnd=np.round(np.random.rand(grid_nodes**3,3),3).astype(np.float32)
#unq_rnd=unq_rnd*10e-13
#extract eigenvalues
eigvals_unsorted=eig_vals_vecs[0]
eigvals=np.sort(eigvals_unsorted)
#extract eigenvectors
eigvecs=eig_vals_vecs[1]
eig_one=eigvals[:,2]
eig_two=eigvals[:,1]
eig_three=eigvals[:,0]
del eigvals

#link eigenvalues as keys to eigenvectors as values inside dictionary    
vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector
del eigvecs
eigvals_unsorted=eigvals_unsorted.flatten()
vecsvals=np.column_stack((eigvals_unsorted,values))
del eigvals_unsorted
del values

####Classifier#### 
recon_img=np.zeros([grid_nodes_true**3])
#NEW FILAMENT FILTER#
recon_filt_one=np.where(eig_three<0)
recon_filt_two=np.where(eig_two<0)
recon_filt_three=np.where(eig_one>=0)
del eig_three
del eig_two
del eig_one
recon_img[recon_filt_one]=1
recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  
del recon_filt_one
del recon_filt_two
recon_img=recon_img.flatten()
recon_img=recon_img.astype(np.int8)
mask=(recon_img !=3)
del recon_img
vecsvals=np.reshape(vecsvals,(grid_nodes_true**3,3,4))
vecsvals[mask,:,:]=np.ones((3,4))*-9
del mask
#Find the empty pixels indices
eig_fil_fnl=np.zeros((grid_nodes_true**3,4))#to compile after finding non filament rows and filament rows
fnd_pxls=np.where(vecsvals==-9)
fnd_pxls=np.unique(fnd_pxls[0])
#Find eig 1 pairs
fnd_prs=np.where(vecsvals[:,:,0]>=0)
#Compile eig_fil_fnl
eig_fil_fnl[fnd_pxls,0:]=9
del fnd_pxls
eig_fil_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
del fnd_prs
del vecsvals

#recon_vecs=eig_fil_fnl[:,0]
recon_vecs_x=eig_fil_fnl[:,1]
recon_vecs_y=eig_fil_fnl[:,2]
recon_vecs_z=eig_fil_fnl[:,3]
del eig_fil_fnl

f=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%skpc_%d.h5" %(grid_nodes,std_dev_phys,part), 'w')
f.create_dataset('/group%d/x'%part,data=recon_vecs_x)
f.create_dataset('/group%d/y'%part,data=recon_vecs_y)
f.create_dataset('/group%d/z'%part,data=recon_vecs_z)
f.close()
