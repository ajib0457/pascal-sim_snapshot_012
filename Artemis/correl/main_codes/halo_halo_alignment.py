import numpy as np
import h5py
from scipy.spatial import distance
import matplotlib.pyplot as plt

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
#no_halos=len(data)
a_len=5000
dist_sp=distance.pdist(data[0:a_len,0:3], 'euclidean')

bn=np.logical_and(dist_sp>=8,dist_sp<15)

a=np.where(bn==True)
a=np.asarray(a)
dist_sp=dist_sp[a]
l,w=np.shape(a)
indxs=np.zeros((w))
indxd=np.zeros((w))

for j in range(w): 
    i=0
    div=a[0,j]-(a_len-1)    
    while div>=0:
        i+=1
        div=div-(a_len-1-i)
    indxs[j]=i    
    indxd[j]=indxs[j]+1+div+(a_len-1-i)


#test distances
dist_sp_test=np.zeros((len(indxs)))
for i in range(len(indxs)):
    
    dist_sp_test[i]=np.sqrt((data[int(indxs[i]),0]-data[int(indxd[i]),0])**2+(data[int(indxs[i]),1]-data[int(indxd[i]),1])**2+(data[int(indxs[i]),2]-data[int(indxd[i]),2])**2)

print w
print np.sum(dist_sp_test-dist_sp)
        
'''
x,y=np.shape(a)
min_dis=np.min(dist_sp)
plt.hist(dist_sp,bins=200,label='%s__min%s'%(y,round(min_dis,3)))
plt.xlabel('halo distance (Mpc/h)')
plt.ylabel('qty')
plt.legend()
plt.savefig('halo_halo_alignment.png')
'''


