import numpy as np
import pandas as pd
import math as mth
import sklearn.preprocessing as skl
from scipy.spatial import distance
import h5py
import sys

'''
This code takes the distance measure between all sampling points and haloes, then finds the first minimum distance
for each halo. I say first becuase sometimes filaments have two identical sampling points, thus using np.argmin()
I take the first incidentally. This shouldn't matter since post smoothing the filaments should be aligned.

Furthermore I take the dp of 2 segments whenever I find the closest sampling point to be not at the extremes of the
filament, otherwise I take dp of 1 segment only.
'''

sim_sz=int(sys.argv[1])           # box size kpc/h
sim_type=sys.argv[2]      #'dm_only' 'DTFE'
cosmology=sys.argv[3]     #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=sys.argv[4]      #'12  '11'...
den_type=sys.argv[5]      #'DTFE' 'my_den'
smooth_scl=sys.argv[6]    #Smoothing scale in physical units Mpc/h
spine_dist=int(sys.argv[7])         #kpc
particles_filt=int(sys.argv[8])
f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snap%s_smth%s_%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct_mask_exyz.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot,smooth_scl,den_type), 'r')
cat_data=f['/halo_lss'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()
x_min=0
x_max=250000
y_min=0
y_max=250000
z_min=0
z_max=250000

#Filter out halos with N particles
partcl_halo_flt=np.where(cat_data[:,11]>=particles_filt)#filter for halos with <N particles
cat_data=cat_data[partcl_halo_flt]#Filter out halos with <N particles
cat_data[:,0:3]=cat_data[:,0:3]*1000.0
mask=np.zeros((len(cat_data)))
mask_xmin=np.where(cat_data[:,0]>x_min)
mask[mask_xmin]=1
mask_xmax=np.where(cat_data[:,0]<x_max)
mask[mask_xmax]+=1
mask_ymin=np.where(cat_data[:,1]>y_min)
mask[mask_ymin]+=1
mask_ymax=np.where(cat_data[:,1]<y_max)
mask[mask_ymax]+=1
mask_zmin=np.where(cat_data[:,2]>z_min)
mask[mask_zmin]+=1
mask_zmax=np.where(cat_data[:,2]<z_max)
mask[mask_zmax]+=1
cat_data=cat_data[np.where(mask==6)]

haloes_pos=cat_data[:,0:3]
halo_mass=cat_data[:,9]
spin_norm=skl.normalize(cat_data[:,6:9])  

#import skeleton
#data = pd.read_csv('/scratch/GAMNSCM2/disperse/%s/%s/snapshot_0%s/snapshot_011.NDnet_s5.up.NDskl.a.NDskl'%(sim_type,cosmology,snapshot), sep='delimiter',engine='python')#error_bad_lines=False
data = pd.read_csv('/scratch/GAMNSCM2/disperse/%s/%s/snapshot_0%s/snapshot_011.NDnet_s3.5.up.NDskl.rmB.BRK.S005.NDskl.a.NDskl'%(sim_type,cosmology,snapshot), sep='delimiter',engine='python')#error_bad_lines=False
data_arr=np.asarray(data)

#Find distance between all relevant sampling points ===================================
fil_loc=np.where(data_arr=="[FILAMENTS]")[0]#these seperate distinct datasets
strt_indx_fil=fil_loc+1
no_fil=','.join(data.values[int(strt_indx_fil)]).split()
no_fil=int(np.asarray(map(int,list(no_fil))))
smp_arr={}#dictionary with filament id as keys and no subkeys
test_smp_arr=[]

fil_test_id=[]
for i in range(no_fil): 
    strt_indx_fil+=1
    no_smp=int(','.join(data.values[int(strt_indx_fil)]).split()[2])   
    fil_det_row=np.zeros((1,3))
    fil_det_row[0,:]=','.join(data.values[int(strt_indx_fil)]).split()

    #save sample points into array
    smp_pos=np.zeros((no_smp,3))
    for j in range(no_smp):
        
        strt_indx_fil+=1
        smp_pos[j,:]=','.join(data.values[int(strt_indx_fil)]).split()
        if x_min<smp_pos[j,0]<x_max and y_min<smp_pos[j,1]<y_max and z_min<smp_pos[j,2]<z_max: 
            if j==0 or j==no_smp-1: 
                test_smp_arr.append(np.hstack((smp_pos[j,:],i)))#will hstack slow me down?
    smp_arr[i]=smp_pos

test_smp_arr=np.array(test_smp_arr)
test_halo_smp_dist=distance.cdist(haloes_pos,test_smp_arr[:,0:3],metric='euclidean')#2D distance matrix
min_indx=np.argmin(test_halo_smp_dist,axis=1)#I can use np.argmin() but for now stick with this

dp_1=np.zeros((len(haloes_pos),3))-1   

for i in range(len(haloes_pos)):
    
    fil=smp_arr[test_smp_arr[min_indx[i],3]]

    halo_smp_dist=distance.cdist(fil,np.reshape(haloes_pos[i],(1,3)),metric='euclidean')
    min_dist_indx=np.argmin(halo_smp_dist)
    
    if halo_smp_dist[min_dist_indx]<spine_dist:
        dp_1[i,0]=halo_mass[i]
        
        if min_dist_indx==0:#beginning of filament
            
            seg_1=fil[min_dist_indx]
            seg_2=fil[min_dist_indx+1]
            sepvecs=np.array([seg_1-seg_2])
            sepvecs=skl.normalize(sepvecs)
            dp_val=abs(np.inner(sepvecs,spin_norm[i]))
            dp_1[i,1]=dp_val
            
        elif min_dist_indx==len(fil)-1:#end of filament
            
            seg_1=fil[min_dist_indx]
            seg_2=fil[min_dist_indx-1]
            sepvecs=np.array([seg_1-seg_2])
            sepvecs=skl.normalize(sepvecs)
            dp_val=abs(np.inner(sepvecs,spin_norm[i]))
            dp_1[i,1]=dp_val
            
        else:#within somehwere of filament
            seg_1=fil[min_dist_indx]
            seg_2=fil[min_dist_indx-1]
            seg_3=fil[min_dist_indx+1]
            sepvecs=np.vstack((seg_1-seg_2,seg_1-seg_3))
            sepvecs=skl.normalize(sepvecs)
            dp_val=abs(np.inner(sepvecs,spin_norm[i]))
            dp_1[i,1:3]=dp_val
            
dp_1=dp_1[np.where(dp_1[:,1]>=0)]
exc_indx=np.where(dp_1[:,2]>=0)
dp_2=np.zeros((len(exc_indx[0]),2))
dp_2[:,1]=dp_1[exc_indx,2]
dp_2[:,0]=dp_1[exc_indx,0]  
dp_1_sub=dp_1[:,0:2] 
dist_alsub=np.vstack((dp_1_sub,dp_2))

f2=h5py.File("/scratch/GAMNSCM2/disperse/%s/%s/snapshot_0%s/disp_3.5sig_dp_%s_%s_snap%s_dist%smpch.rmB.BRK.S005.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot,spine_dist), 'w')
f2.create_dataset('/test',data=dist_alsub)#[[[hmass,spin],[hmass2,spin2],...]]
f2.close()   

