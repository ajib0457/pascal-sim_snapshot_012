import numpy as np
import pandas as pd
import h5py
sim_type='dm_only'
cosmology='lcdm'
snapshot='10'
f=h5py.File("%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot), 'r')
cat=f['/halo'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()

#data = pd.read_csv('/import/oth3/ajib0457/pascal_sims/treefrog/out_test',header = None)
data = pd.read_csv('treefrog_dm_only_005_011_tree',header=None)

print data
no_halos=int(len(data)-2)
data=data[0][3:no_halos+3]
data=np.asarray(data)

a=np.ones((len(data),3))*-1
for i in range(len(data)):
    
    val=np.fromstring(data[i],  sep="\t")
    a[i,0:len(val)]=val

filt_lbls=np.where(a[:,2]<0)
flt_assoc=np.where(a[:,2]>=0)

fnd_snaphalos=np.where(a[:,1]==len(cat))
