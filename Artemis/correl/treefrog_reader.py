import numpy as np
import pandas as pd
import h5py
sim_type='dm_only'
cosmology='lcdm'
snapshot='10'
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure),ID(1) ,ID_mbp(2), hostHaloID(3) and numSubStruct(4)
f=h5py.File("%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot), 'r')
cat_10=f['/halo'][:]
f.close()
sim_type='dm_only'
cosmology='lcdm'
snapshot='11'
f=h5py.File("%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot), 'r')
cat_11=f['/halo'][:]
f.close()

#data = pd.read_csv('/import/oth3/ajib0457/pascal_sims/treefrog/out_test',header = None)
data = pd.read_csv('treefrog_dm_only_005_011_tree',header=None)

#to skip header
no_halos=int(len(data)-2)
data=data[0][3:no_halos+3]
data=np.asarray(data)

#rid of space and \t within data
a=np.ones((len(data),3))*-1
for i in range(len(data)):
    
    val=np.fromstring(data[i],  sep="\t")
    a[i,0:len(val)]=val

#seperate merit rows and 'label' rows
filt_lbls=np.where(a[:,2]<0)#halo counts and labels
filt_lbls_a=a[filt_lbls]
filt_lbls_a=np.column_stack((filt_lbls_a,np.asarray(filt_lbls).transpose()))

filt_assoc=np.where(a[:,2]>=0)#the comparing halos and merits
filt_assoc_a=a[filt_assoc]
filt_assoc_a=np.column_stack((filt_assoc_a,np.asarray(filt_assoc).transpose()))

#halo to find, as an example. here I am assuming the list of tree will contain this halo.
blobsy=cat_11[0,12]#halo ID
blobsy_mass=cat_11[0,9]#halo mass
begin_snap=int(np.asarray(np.where(a[:,1]==len(cat_10))))#I should look after this point for finding it in snap10
end_snap=int(np.asarray(np.where(a[:,1]==len(cat_11)))-1)#and before this point


fnd_halo=np.where(a[begin_snap:end_snap,0]==blobsy)

fnd_halo=np.asarray(fnd_halo)+begin_snap
