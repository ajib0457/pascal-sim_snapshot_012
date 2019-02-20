import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mth
import sklearn.preprocessing as skl
from scipy.spatial import distance
import collections
import random

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

np.random.seed(1)
tot_no_haloes=200
#create haloes
haloes_pos=np.random.rand(tot_no_haloes,3)*50000
haloes_spin=np.random.rand(tot_no_haloes,3)
spin_norm=skl.normalize(haloes_spin[:]) 
halo_mass=np.random.rand(tot_no_haloes)
halo_id=np.asarray(random.sample(range(1, 1000), tot_no_haloes))
catlog=np.column_stack((halo_id,spin_norm,halo_mass))
#import skeleton
data = pd.read_csv('/import/oth3/ajib0457/disperse/3d_vis_recreate/simu_32_id.gad.NDnet_s3.5.up.NDskl.a.NDskl', sep='delimiter',engine='python')#error_bad_lines=False
data_arr=np.asarray(data)
#indices of data
sim_dim=','.join(data_arr[2]).split()
sim_sz=50000.0
strt_indx=0
#Find cp of type 3 (maxima) closest to haloes
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

cp_store_fnl=cp_store[np.where(cp_store[:,3]>=0)]#filter out empty rows
halo_cp_dist=distance.cdist(haloes_pos,cp_store_fnl[:,0:3],metric='euclidean')#2D distance matrix
min_dist=np.min(halo_cp_dist,axis=1,keepdims=True)#Find the closest cp to each halo
min_indx=np.where(halo_cp_dist==min_dist)[1]#I can use np.argmin() but for now stick with this
del halo_cp_dist
cp_store_filt=cp_store_fnl[min_indx,3]#the cp id for each halo, in order of halo catalog

fil_loc=np.where(data_arr=="[FILAMENTS]")[0]#these seperate distinct datasets
strt_indx_fil=fil_loc+1
no_fil=','.join(data.values[int(strt_indx_fil)]).split()
no_fil=int(np.asarray(map(int,list(no_fil))))
smp_dict=collections.defaultdict(dict)#dictionary with filament id as keys and no subkeys
dist_dict=collections.defaultdict(dict)#halo id as keys and filament id as subkeys

fil_extr=np.zeros((no_fil,2))
for i in range(no_fil): 
    strt_indx_fil+=1
    cp_fil=','.join(data.values[int(strt_indx_fil)]).split()[0:2] 
    no_smp=int(','.join(data.values[int(strt_indx_fil)]).split()[2])
    cp_ex1_indx=np.where(cp_store_filt==int(cp_fil[0]))#find where the first fil cp hits clos halo cp
    cp_ex2_indx=np.where(cp_store_filt==int(cp_fil[1]))#and the second.
    cp_ex1=np.shape(cp_ex1_indx)[1]#find how many hits there was
    cp_ex2=np.shape(cp_ex2_indx)[1]#same as above for cp2
    fil_extr[i]=cp_fil
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
count=0
abs_dist=[]#[[halo_mass,dp],[]] adopt dp code and count how many for second scenario and lim dist
#make an empty array with length of number of haloes. and width of 3 or so, and replace all elements
#with -1. but then when I want to use it
for key in halo_id:
    distance=np.zeros((len(dist_dict[key]),4))#[halo_id,fil_id,seg_id,min_dist]
    i=0
    for subkey in dist_dict[key].keys():
        distance[i,0]=key
        distance[i,1]=subkey
        distance[i,2:4]=dist_dict[key][subkey]
        i+=1
    min_dist=distance[np.argmin(distance[:,3]),:]#find the min distance of the options in the form: [fil_id,smp-point_id,min_dist]
    halo=catlog[np.where(catlog[:,0]==min_dist[0])]
    hmass=halo[0][4]
    hspin=halo[0][1:4]
    
    if min_dist[2]==0:#If closest smp point is at the beginning of filament
        #This takes dp of two segments, one from each filament       
        segs=dist_dict[min_dist[0]].keys()        
        seg_1=smp_dict[segs[0]][0]
        seg_2=smp_dict[segs[0]][1]
        seg_3=smp_dict[segs[1]][1]
        sepvecs=np.vstack((seg_1-seg_2,seg_1-seg_3))
        sepvecs=skl.normalize(sepvecs)
        dp_val=abs(np.inner(sepvecs,hspin))
        abs_dist.append(np.column_stack(([hmass,hmass],dp_val)))
#        abs_dist.append(np.array([[hmass],[hmass]]))
                
    elif min_dist[2]==len(smp_dict[min_dist[1]])-1:#if closest smp point is at the end of filament
        #note: this only takes one segment. figure out second segment if count is high!
         seg_1=smp_dict[int(min_dist[1])][int(min_dist[2])]
         seg_2=smp_dict[int(min_dist[1])][int(min_dist[2]-1)]
         sepvecs=np.array([seg_1-seg_2])
         sepvecs=skl.normalize(sepvecs)
         dp_val=abs(np.inner(sepvecs,hspin))
         abs_dist.append(np.column_stack((hmass,dp_val)))
#         abs_dist.append(np.array([hmass]))
         
    else:# if in the middle somehwere, simple.
        #This scenario takes the dp of two adjacent segments
        seg_1=smp_dict[int(min_dist[1])][int(min_dist[2])]
        seg_2=smp_dict[int(min_dist[1])][int(min_dist[2]+1)]
        seg_3=smp_dict[int(min_dist[1])][int(min_dist[2]-1)]        
        sepvecs=np.vstack((seg_1-seg_2,seg_1-seg_3))        
        sepvecs=skl.normalize(sepvecs)
        dp_val=abs(np.inner(sepvecs,hspin))
        abs_dist.append(np.column_stack(([hmass,hmass],dp_val)))
#        abs_dist.append(np.array([[hmass],[hmass]]))

#just need to figure out how to convert this list/object into a 2d array. then feed into halo_lss_2 code
    
    
 
