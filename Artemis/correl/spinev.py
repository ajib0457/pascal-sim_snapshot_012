import numpy as np
import pandas as pd
import h5py
import sklearn.preprocessing as skl
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), 
#(tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure),ID(1) ,
#ID_mbp(2), hostHaloID(3) and numSubStruct(4)
sim_type=sys.argv[1]   #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
smooth_scl=sys.argv[3]
den_type=sys.argv[4]
lss_type=int(sys.argv[5])
particles_flt=100
filehandler = open('/scratch/GAMNSCM2/Treefrog/%s/%s/treefrog_dict_%s_%s_smth%s_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct_mask_exyz.pkl'%(sim_type,cosmology,sim_type,cosmology,smooth_scl),"rb")
diction=pickle.load(filehandler)#[vxyz, jxyz, mtot, r, npart, id, idmbp, hosthaloid, numsubstruct, mask, exyz]...
filehandler.close() 

data=[]
hist_data=np.zeros((7,4))#(snapshots(z=0,z=0.26...),LSS_type(0,1,2,3))
   
store_dpl=[]
for i in diction.keys():
    if np.asarray(diction[i])[0,13]==lss_type and np.asarray(diction[i])[0,8]>particles_flt:#filter out for LSS or low dens region
        matrix=np.asarray(diction[i])
        store_dp=np.zeros(len(matrix))
        for j in range(len(matrix)):#run through each progentior inc. parent
               vecs_spin=skl.normalize(data[j,3:6])
               vecs_eigv=skl.normalize(data[j,13:16])                           
               store_dp[j]=np.inner(vecs_spin,vecs_eigv)#take the dot product between vecs, row by row 
        store_dpl.append(store_dp)
            #also calc the difference between spins
        
   
