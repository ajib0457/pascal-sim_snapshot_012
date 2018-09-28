import numpy as np
import matplotlib
import h5py
from scipy.spatial import distance
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sim_type='dm_only'   #'dm_only' 'DTFE'
cosmology='lcdm'     #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot='09'          #'12  '11'...
den_type='DTFE'      #'DTFE' 'my_den'
smooth_scl=2       #Smoothing scale in physical units Mpc/h. 2  3.5  5
sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution

f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
data=f['/halo'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()
#no_halos=len(data)

dist_sp=distance.pdist(data[:,0:3], 'euclidean')
bn=np.logical_and(dist_sp>=0.06,dist_sp<0.1)
a=np.where(bn==True)

x,y=np.shape(a)
min_dis=np.min(dist_sp)
plt.hist(dist_sp,bins=200,label='%s__min%s'%(y,round(min_dis,3)))
plt.xlabel('halo distance (Mpc/h)')
plt.ylabel('qty')
plt.legend()
plt.savefig('halo_halo_alignment.png')
