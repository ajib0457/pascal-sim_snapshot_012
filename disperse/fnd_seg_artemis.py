import numpy as np
import pandas as pd
import math as mth
import sklearn.preprocessing as skl
from scipy.spatial import distance
import collections
import h5py
import sys
'''
This code generates a dictionary which includes all the sampling points (of filaments) connected to 
the cp's which are closest to each halo in provided catalog. 
'''
def dist(halo_pos,smp_points,halo_id,sim_sz):
    '''
    This function will produce: distance=[halo_qty,halo_ids,seg_index1,min_dist1,seg_index2,min_dist2,...]
    '''
    if int(np.shape(np.where(smp_points>sim_sz))[1])>0:
        a=np.where(smp_points>sim_sz)
        smp_points[a]=abs(abs(smp_points[a])-sim_sz)
    
    if int(np.shape(np.where(smp_points<0.0))[1])>0:
        a=np.where(smp_points<0.0)
        smp_points[a]=abs(abs(smp_points[a])-sim_sz)
            
    len_haloes=np.shape(halo_pos)[0]
    len_smp=np.shape(smp_points)[0]
    distance=np.zeros((1+3*len_haloes))
    distance[0]=len_haloes
    distance[1:len_haloes+1]=halo_id
    for i in range(len_haloes):
        all_dist=np.zeros((len_smp))        
        for j in range (len_smp):
        
            all_dist[j]=mth.sqrt((halo_pos[i,0]-smp_points[j,0])**2+(halo_pos[i,1]-smp_points[j,1])**2 +(halo_pos[i,2]-smp_points[j,2])**2)
    
        distance[1+2*i+len_haloes]=np.argmin(all_dist)        
        distance[1+2*i+len_haloes+1]=np.min(all_dist)
        
    return distance,smp_points

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
#Filter out halos with N particles
partcl_halo_flt=np.where(cat_data[:,11]>=particles_filt)#filter for halos with <N particles
cat_data=cat_data[partcl_halo_flt]#Filter out halos with <N particles
halo_id=cat_data[:,12]
haloes_pos=cat_data[:,0:3]*1000.0 #convert back to kpc/h
halo_mass=cat_data[:,9]
spin_norm=skl.normalize(cat_data[:,6:9])  
catlog=np.column_stack((halo_id,spin_norm,halo_mass))

#import skeleton
data = pd.read_csv('/scratch/GAMNSCM2/disperse/%s/%s/snapshot_0%s/snapshot_011.NDnet_s3.5.up.NDskl.a.NDskl'%(sim_type,cosmology,snapshot), sep='delimiter',engine='python')#error_bad_lines=False
data_arr=np.asarray(data)
#indices of data
sim_dim=','.join(data_arr[2]).split()

#Find cp of type 3 (maxima) closest to haloes==========================================
strt_indx=0
cp_loc=np.where(data_arr=="[CRITICAL POINTS]")[0]#these seperate distinct datasets
no_cp=','.join(data.values[int(cp_loc+1)]).split()#number of critical points 
no_cp=int(np.asarray(map(int,list(no_cp))))
strt_indx=cp_loc+2#first cp
cp_store=np.zeros((no_cp,4))-1
for i in range(no_cp):
    #check that it has connecting filaments
    fil_index=strt_indx+1
    no_fil=','.join(data.values[int(fil_index)]).split()
    no_fil=int(np.asarray(map(int,list(no_fil))))
    #store the cp type of each cp and the cp ID
    cp_type=','.join(data.values[int(strt_indx)]).split()[0]    
    
    if int(cp_type)==3:    
    
        cp_pos=','.join(data.values[int(strt_indx)]).split()[1:4]
        cp_id=','.join(data.values[int(strt_indx)]).split()[5]
        #Then store in np array
        cp_store[i,0:3]=cp_pos
        cp_store[i,3]=int(cp_id)
        #get to next cp by reading filament data as to skip        
        strt_indx=strt_indx+int(no_fil)+2
#Take distance b/w haloes and cp ======================================================
cp_store_fnl=cp_store[np.where(cp_store[:,3]>=0)]#filter out empty rows
halo_cp_dist=distance.cdist(haloes_pos,cp_store_fnl[:,0:3],metric='euclidean')#2D distance matrix
min_dist=np.min(halo_cp_dist,axis=1,keepdims=True)#Find the closest cp to each halo
min_indx=np.where(halo_cp_dist==min_dist)[1]#I can use np.argmin() but for now stick with this
del halo_cp_dist
cp_store_filt=cp_store_fnl[min_indx,3]#the cp id for each halo, in order of halo catalog
#Find distance between all relevant sampling points ===================================
fil_loc=np.where(data_arr=="[FILAMENTS]")[0]#these seperate distinct datasets
strt_indx_fil=fil_loc+1
no_fil=','.join(data.values[int(strt_indx_fil)]).split()
no_fil=int(np.asarray(map(int,list(no_fil))))
smp_dict=collections.defaultdict(dict)#dictionary with filament id as keys and no subkeys
dist_dict=collections.defaultdict(dict)#halo id as keys and filament id as subkeys

#fil_extr=np.zeros((no_fil,2))#used for diagnostic purposes
for i in range(no_fil): 
    strt_indx_fil+=1
    cp_fil=','.join(data.values[int(strt_indx_fil)]).split()[0:2] 
    no_smp=int(','.join(data.values[int(strt_indx_fil)]).split()[2])
    cp_ex1_indx=np.where(cp_store_filt==int(cp_fil[0]))#find where the first fil cp hits clos halo cp
    cp_ex2_indx=np.where(cp_store_filt==int(cp_fil[1]))#and the second.
    cp_ex1=np.shape(cp_ex1_indx)[1]#find how many hits there was
    cp_ex2=np.shape(cp_ex2_indx)[1]#same as above for cp2
#    fil_extr[i]=cp_fil#used for diagnostic purposes
    if cp_ex1>0 or cp_ex2>0:#if a filament has common cp for either       
        #save sample points into array
        smp_pos=np.zeros((no_smp,3))
        for j in range(no_smp):            
            strt_indx_fil+=1
            smp_pos[j,:]=','.join(data.values[int(strt_indx_fil)]).split()
        
        #find which halo spins corresp. w/ filament cp
        if cp_ex1>0: 
            
            halo_ex1_pos=haloes_pos[cp_ex1_indx]
            halo_ex1_id=halo_id[cp_ex1_indx]
            #save the min dist and smp_point index in dictionary
            fil_halo_dist,smp_pos_fnl=dist(halo_ex1_pos,smp_pos,halo_ex1_id,sim_sz)
            no_haloes=int(fil_halo_dist[0])

            for k in range(no_haloes):            
                dist_dict[int(fil_halo_dist[k+1])][i]=fil_halo_dist[(no_haloes+1+2*k):(no_haloes+1+2*k+2)]

        if cp_ex2>0:
            
            halo_ex2_pos=haloes_pos[cp_ex2_indx]
            halo_ex2_id=halo_id[cp_ex2_indx]
            #save the min dist and smp_point index in dictionary
            fil_halo_dist,smp_pos_fnl=dist(halo_ex2_pos,smp_pos,halo_ex2_id,sim_sz)
            no_haloes=int(fil_halo_dist[0])

            for k in range(no_haloes):            
                dist_dict[int(fil_halo_dist[k+1])][i]=fil_halo_dist[(no_haloes+1+k):(no_haloes+1+k+2)]
        smp_dict[i]=smp_pos_fnl#save all relevant filament sampling points after periodicity corrections
         
    else:#if filament not related, go to next filament.
         strt_indx_fil+=no_smp 
del data                                         
#find closest segment/s and take dp ================================================
dp_1=np.zeros((len(halo_id),3))-1
j=0
for key in halo_id: 
    distance=np.zeros((len(dist_dict[key]),4))#[halo_id,fil_id,seg_id,min_dist]
    i=0
    for subkey in dist_dict[key].keys():
        distance[i,0]=key
        distance[i,1]=subkey
        distance[i,2:4]=dist_dict[key][subkey]
        i+=1
    min_dist=distance[np.argmin(distance[:,3]),:]#find the min distance of the options in the form: [haloid,filid,smppntid,min_dist]
    halo=catlog[np.where(catlog[:,0]==min_dist[0])]
    hmass=halo[0][4]
    hspin=halo[0][1:4]
    dp_1[j,0]=hmass
    if min_dist[2]==0 and min_dist[3]<spine_dist:#If closest smp point is at the beginning of filament
        #This takes dp of two segments, one from each filament       
        segs=dist_dict[min_dist[0]].keys()        
        seg_1=smp_dict[segs[0]][0]
        seg_2=smp_dict[segs[0]][1]
        seg_3=smp_dict[segs[1]][1]
        sepvecs=np.vstack((seg_1-seg_2,seg_1-seg_3))
        sepvecs=skl.normalize(sepvecs)
        dp_val=abs(np.inner(sepvecs,hspin))
        dp_1[j,1:3]=dp_val
                
    elif min_dist[2]==len(smp_dict[min_dist[1]])-1 and min_dist[3]<spine_dist:#if closest smp point is at the end of filament
        #note: this only takes one segment. figure out second segment if count is high!
         seg_1=smp_dict[int(min_dist[1])][int(min_dist[2])]
         seg_2=smp_dict[int(min_dist[1])][int(min_dist[2]-1)]
         sepvecs=np.array([seg_1-seg_2])
         sepvecs=skl.normalize(sepvecs)
         dp_val=abs(np.inner(sepvecs,hspin))
         dp_1[j,1]=dp_val
         
    elif min_dist[3]<spine_dist:# if in the middle somehwere, simple.
        #This scenario takes the dp of two adjacent segments
        seg_1=smp_dict[int(min_dist[1])][int(min_dist[2])]
        seg_2=smp_dict[int(min_dist[1])][int(min_dist[2]+1)]
        seg_3=smp_dict[int(min_dist[1])][int(min_dist[2]-1)]        
        sepvecs=np.vstack((seg_1-seg_2,seg_1-seg_3))        
        sepvecs=skl.normalize(sepvecs)
        dp_val=abs(np.inner(sepvecs,hspin))
        dp_1[j,1:3]=dp_val
    j+=1
del dist_dict 
del smp_dict   
dp_1=dp_1[np.where(dp_1[:,1]>=0)]
exc_indx=np.where(dp_1[:,2]>=0)
dp_2=np.zeros((len(exc_indx[0]),2))
dp_2[:,1]=dp_1[exc_indx,2]
dp_2[:,0]=dp_1[exc_indx,0]  
dp_1_sub=dp_1[:,0:2] 
dist_alsub=np.vstack((dp_1_sub,dp_2)) 

f2=h5py.File("/scratch/GAMNSCM2/disperse/%s/%s/snapshot_0%s/test.h5"%(sim_type,cosmology,snapshot), 'w')
f2.create_dataset('/test',data=dist_alsub)
f2.close()
