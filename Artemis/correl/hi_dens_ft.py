import numpy as np
import pandas as pd
import h5py
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), 
#(tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure),ID(1) ,
#ID_mbp(2), hostHaloID(3) and numSubStruct(4)
sim_type=sys.argv[1]   #'dm_only' 
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
#then to get the histogram
for i in diction.keys():
    if np.asarray(diction[i])[0,13]==lss_type and np.asarray(diction[i])[0,8]>particles_flt:#filter out for LSS or low dens region
        matrix=np.asarray(diction[i])
        for j in range(len(matrix)):#run through each progentior inc. parent
            if matrix[j,8]>particles_flt:
                z=int(j)#redshift is tied to the row position of halo within array
                lss=int(matrix[j,13])#lss code (void-0,sheet-1...) tied to array column index
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

