import numpy as np
import pandas as pd
import h5py
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math as mth
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), 
#(tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure),ID(1) ,
#ID_mbp(2), hostHaloID(3) and numSubStruct(4)
sim_sz=500
sim_type=sys.argv[1]   #'dm_only' 
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
smooth_scl=sys.argv[3]
den_type=sys.argv[4]
lss_type=int(sys.argv[5])
particles_flt=100
grid_nodes=1250
total_lss_parts=8    #Total amount of lss_class parts
snapshot=11      #snapshot to keep the mask constant on to track halo migration
filehandler = open('/scratch/GAMNSCM2/Treefrog/%s/%s/treefrog_dict_%s_%s_smth%s_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct_mask_exyz.pkl'%(sim_type,cosmology,sim_type,cosmology,smooth_scl),"rb")
diction=pickle.load(filehandler)#[xyz, vxyz, jxyz, mtot, r, npart, id, idmbp, hosthaloid, numsubstruct, mask, exyz]...
filehandler.close() 

mask=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):
    
    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f2=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d_mask.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
mask=np.reshape(mask(grid_nodes,grid_nodes,grid_nodes))

f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snap%s_smth%s_%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct_mask_exyz.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot,smooth_scl,den_type), 'r')
cat_11=f['/halo_lss'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()    
#Prebinning for dotproduct binning within loop ------
Xc_min=np.min(cat_11[:,0])
Xc_max=np.max(cat_11[:,0])
Yc_min=np.min(cat_11[:,1])
Yc_max=np.max(cat_11[:,1])
Zc_min=np.min(cat_11[:,2])
Zc_max=np.max(cat_11[:,2])

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001

data=[]
hist_data=np.zeros((7,4))#(snapshots(z=0,z=0.26...),LSS_type(0,1,2,3))
#then to get the histogram
for i in diction.keys():
    if np.asarray(diction[i])[0,13]==lss_type and np.asarray(diction[i])[0,8]>particles_flt:#filter out for LSS or low dens region
    
        matrix=np.asarray(diction[i])
        for j in range(len(matrix)):#run through each progentior inc. parent
            if matrix[j,8]>particles_flt:
                z=int(j)#redshift is tied to the row position of halo within array
                #Now I have the positions, I can bin them and find the z=0 mask.
                grid_index_x=mth.trunc(matrix[j,0]*Xc_mult-Xc_minus)      
                grid_index_y=mth.trunc(matrix[j,1]*Yc_mult-Yc_minus) 
                grid_index_z=mth.trunc(matrix[j,2]*Zc_mult-Zc_minus) 
                lss=int(mask[grid_index_x,grid_index_y,grid_index_z])
                
                hist_data[z,lss]+=1#tally for histogram
            
# data to plot
n_groups = 4
bars_pg=7
 
# create plot
fig, ax = plt.subplots(figsize=(11,8))
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8
 
#z=['z=2.98','z=2.16','z=1.51','z=1.00','z=0.59','z=0.26','z=0.00'] #Redshift
z=['z=0.00','z=0.26','z=0.59','z=1.00','z=1.51','z=2.16','z=2.98']
colors=['grey','black','orange','red','brown','indigo','blue']
for i in range(bars_pg):
    rects1 = plt.bar(index+i*bar_width, np.log10(hist_data[i,:]), bar_width,alpha=opacity,color=colors[i],label=z[i])

plt.xlabel('LSS Type')
plt.ylabel('$log_{10}$[Qty]')
plt.xticks(index + (bars_pg-1)*bar_width/2.0, ('Void', 'Sheet', 'Filament', 'Cluster'))
plt.legend()
 
plt.tight_layout()
plt.show()
plt.savefig('/scratch/GAMNSCM2/Treefrog/tally_plts/%s/ft_hist_%s_%s_smth%s_begin_%s_prtflt%s.png'%(sim_type,sim_type,cosmology,smooth_scl,lss_type,particles_flt))

