import numpy as np
import math as mth
import pandas as pd
import sklearn.preprocessing as skl
import h5py
from scipy.spatial import distance

sim_type='dm_only'   #'dm_only' 'DTFE'
cosmology='lcdm'     #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot='09'          #'12  '11'...
den_type='DTFE'      #'DTFE' 'my_den'
smooth_scl=2       #Smoothing scale in physical units Mpc/h. 2  3.5  5
sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution

f=h5py.File("%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart.h5"%(sim_type,cosmology,snapshot), 'r')
data=f['/halo'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()
no_halos=len(data)-224000
#dist=np.zeros(((no_halos-1)**2+no_halos-1)/2)
#no_dist=no_halos*(no_halos+1)/2.0
#print "{:,}".format(no_dist)
dist_sp=distance.pdist(data[0:50000,0:3], 'euclidean')

'''
k=0
for i in range(no_halos-1):
     for j in range(i+1,no_halos):
         
         dist[k]=np.sqrt((data[i,0]-data[j,0])**2+(data[i,1]-data[j,1])**2+(data[i,2]-data[j,2])**2)
         k+=1
'''
