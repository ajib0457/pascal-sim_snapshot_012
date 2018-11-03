import h5py
import numpy as np
import pynbody
import math as mth
import sys

sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution
smooth_scl=sys.argv[5]       #Smoothing scale in physical units Mpc/h
total_lss_parts=8    #Total amount of lss_class parts
cosmology=sys.argv[2]     #'lcdm'  'cde0'  'wdm2'
snapshot=sys.argv[3]          #'12  '11'
sim_type=sys.argv[1]   #'dm_only' 'dm_gas' 
den_type=sys.argv[4]      #'DTFE' 'my_den'
#load mask
recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))
mask=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):

    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d_mask.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()

#f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_box_%scubed_xyz_vxyz_jxyz_m_r.h5"%sim_sz, 'r')#xyz vxvyvz jxjyjz & Rmass & Rvir: Halo radius (kpc/h comoving).
f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
data=f['/halo'][:]#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()
#Prebinning for dotproduct binning within loop ------
Xc_min=np.min(data[:,0])
Xc_max=np.max(data[:,0])
Yc_min=np.min(data[:,1])
Yc_max=np.max(data[:,1])
Zc_min=np.min(data[:,2])
Zc_max=np.max(data[:,2])

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#----------------------------------------------------
recon_vecs_flt_unnorm=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z
mask=np.reshape(mask,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs=np.reshape(recon_vecs_flt_unnorm,(grid_nodes,grid_nodes,grid_nodes,3))#Reshape eigenvectors

color_cd=np.zeros((len(data),1))
norm_eigvecs=np.zeros((len(data),3))
data=np.hstack((data,color_cd,norm_eigvecs))
for i in range(len(data)):
   #Create index from halo coordinates
    grid_index_x=mth.trunc(data[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(data[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(data[i,2]*Zc_mult-Zc_minus) 
    data[i,12]=mask[grid_index_x,grid_index_y,grid_index_z]
    data[i,13:16]=recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]
#halos    
f=h5py.File("/scratch/GAMNSCM2/halo_halo_lss/%s/%s/%s_%s_snap%s_smth%s_%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_mask_exyz.h5"%(sim_type,cosmology,sim_type,cosmology,snapshot,smooth_scl,den_type), 'w')
f.create_dataset('/halo_lss',data=data)
f.close()
