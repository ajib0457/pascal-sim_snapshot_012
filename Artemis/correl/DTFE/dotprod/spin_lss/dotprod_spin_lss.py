import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import math as mth
import pandas as pd
import pickle
import sklearn.preprocessing as skl
import h5py
#Initial data from density field and classification
grid_nodes=2000
mass_bin=4     # 0 to 4
tot_mass_bins=5
sim_sz=500000#kpc
in_val,fnl_val=-140,140
tot_parts=8
s=1.96
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))

for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%skpc_%d.h5" %(grid_nodes,std_dev_phys,part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
'''
   
#FILAMENT AXIS -----
eig_x=open("/import/oth3/ajib0457/DTFE/eigvecs/800_grid_nodes/fil_recon_vecs_x_DTFE_gd%d_smth%skpc.pkl" %(grid_nodes,std_dev_phys), 'rb')
recon_vecs_x=pickle.load(eig_x)
eig_y=open("/import/oth3/ajib0457/DTFE/eigvecs/800_grid_nodes/fil_recon_vecs_y_DTFE_gd%d_smth%skpc.pkl" %(grid_nodes,std_dev_phys), 'rb')
recon_vecs_y=pickle.load(eig_y)
eig_z=open("/import/oth3/ajib0457/DTFE/eigvecs/800_grid_nodes/fil_recon_vecs_z_DTFE_gd%d_smth%skpc.pkl" %(grid_nodes,std_dev_phys), 'rb')
recon_vecs_z=pickle.load(eig_z)
'''
#reshape into 4D array so that they can be dot producted 
#recon_vecs_x_flt=np.reshape(recon_vecs_x,(grid_nodes**3))
#recon_vecs_y_flt=np.reshape(recon_vecs_y,(grid_nodes**3))
#recon_vecs_z_flt=np.reshape(recon_vecs_z,(grid_nodes**3))
recon_vecs_flt_unnorm=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z

recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#I should not normalize becauase they are already normalized and also the classifier (9) mask will be ruined
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Three for the 3 vector components
del recon_vecs_flt_norm
recon_vecs_unnorm=np.reshape(recon_vecs_flt_unnorm,(grid_nodes,grid_nodes,grid_nodes,3))#raw eigenvectors along with (9)-filled rows which represent blank vectors
del recon_vecs_flt_unnorm
# -----------------

#HALO mass filters ---------------------------------------------------------------------------------------------
inputfileall=open("/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/input_files/store_allprop.pkl",'rb')
all_prop=pickle.load(inputfileall)
halo_mass=all_prop.get('Mvir')#1e10 solar masses
#npart=all_prop.get('npart')
#100 particle filter
#partcl_100=np.where(npart>=20)#filter out halos with <100 particles
#halo_mass=halo_mass[partcl_100]
#mass bin filter
log_halo_mass=np.log10(halo_mass*10**10)#convert into log(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins
low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin
hi_int_mass=low_int_mass+mass_intvl
mass_mask=np.zeros(len(log_halo_mass))

low_int_mass=12.165#302106
hi_int_mass=12.165235#THESE ARE TESTING DOTPROD> DELETE WHEN RUNNING PROPER JOB

loint=np.where(log_halo_mass>=low_int_mass)#Change these two numbers as according to the above intervals
hiint=np.where(log_halo_mass<hi_int_mass)#Change these two numbers as according to the above intervals
mass_mask[loint]=1
mass_mask[hiint]=mass_mask[hiint]+1
mass_indx=np.where(mass_mask==2)
#-----------------------------------------------------------------------------------------------------------------

#Angular momentum
Lx=all_prop.get('Lx')
Lx=np.asarray(Lx).astype(float)
#Lx=Lx[partcl_100]
Lx=Lx[mass_indx]
Ly=all_prop.get('Ly')
Ly=np.asarray(Ly).astype(float)
#Ly=Ly[partcl_100]
Ly=Ly[mass_indx]
Lz=all_prop.get('Lz')
Lz=np.asarray(Lz).astype(float)
#Lz=Lz[partcl_100]
Lz=Lz[mass_indx]
#Positions
Xc=all_prop.get('Xc')
Xc=np.asarray(Xc).astype(float)
#Xc=Xc[partcl_100]
Xc=Xc[mass_indx]
Yc=all_prop.get('Yc')
Yc=np.asarray(Yc).astype(float)
#Yc=Yc[partcl_100]
Yc=Yc[mass_indx]
Zc=all_prop.get('Zc')
Zc=np.asarray(Zc).astype(float)
#Zc=Zc[partcl_100]
Zc=Zc[mass_indx]
#change to proper coordinates
hub_val=0.67
Xc=Xc*hub_val 
Yc=Yc*hub_val
Zc=Zc*hub_val

#Correct for periodicity
for i in range(len(Xc)):
    if (Xc[i]>500000 or Xc[i]<0):
        Xc[i]=abs(abs(Xc[i])-500000)
        
    if (Yc[i]>500000 or Yc[i]<0):
        Yc[i]=abs(abs(Yc[i])-500000)
        
    if (Zc[i]>500000 or Zc[i]<0):
        Zc[i]=abs(abs(Zc[i])-500000)
#normalized angular momentum vectors v1
halos_mom=np.column_stack((Lx,Ly,Lz))
norm_halos_mom=skl.normalize(halos_mom)
halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))

# -----------------

#pre-binning for Halos ----------
Xc_min=0.47166899999137968
Xc_max=499999.98283700005
Yc_min=2.7157000000006519
Yc_max=499996.69595100009
Zc_min=4.7269982810500002 
Zc_max=499996.66814600001

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#--------------------------------

#dot product details
n_dot_prod=0



#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_spin=[]
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
    #calculate dot product and bin
    if (recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
        spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
        store_spin.append(spin_dot)
del recon_vecs_unnorm
del recon_vecs
del halos
store_spin=np.asarray(store_spin)      
      
         

#plt.close('all')
#plot the dot products
#spin_bins=np.linspace(-1,1,n_dot_prod)
#width=0.015 #width of bars
#plt.bar(spin_bins,store_spin,width)
#plt.yscale('log')
spin_array=open('/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/dotproduct/spin_lss/DTFE_grid%d_spin_store_%dbins_fil_Log%s-%s_smth%skpc_binr.pkl'%(grid_nodes,n_dot_prod,round(low_int_mass,2),round(hi_int_mass,2),std_dev_phys),'wb')
#spin_array=open('/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/dotproduct/spin_store_%dbins_fnl.pkl'%n_dot_prod,'wb')
pickle.dump(store_spin,spin_array)
spin_array.close()
