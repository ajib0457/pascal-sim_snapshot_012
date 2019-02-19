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
def dist(halo_pos,smp_points,halo_id):
    '''
    This function will produce: distance=[halo_qty,halo_ids,seg_index1,min_dist1,seg_index2,min_dist2,...]
    '''
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
        
    return distance

np.random.seed(1)
no_haloes=200
#create haloes
haloes_pos=np.random.rand(no_haloes,3)*50000
haloes_spin=np.random.rand(no_haloes,3)
spin_norm=skl.normalize(haloes_spin[:]) 
halo_mass=np.random.rand(no_haloes)
halo_id=np.asarray(random.sample(range(1, 1000), no_haloes))
#import skeleton
data = pd.read_csv('/import/oth3/ajib0457/disperse/3d_vis_recreate/simu_32_id.gad.NDnet_s3.5.up.NDskl.a.NDskl', sep='delimiter',engine='python')#error_bad_lines=False
data_arr=np.asarray(data)
#indices of data
sim_dim=data_arr[2]
strt_indx=0
#Find cp closest to haloes
cp_loc=np.where(data_arr=="[CRITICAL POINTS]")[0]#these seperate distinct datasets
no_cp=','.join(data.values[int(cp_loc+1)]).split()#number of critical points 
no_cp=int(np.asarray(map(int,list(no_cp))))
strt_indx=cp_loc+2#first cp
cp_store=np.zeros((no_cp,4))-1
sddl_sv=np.zeros((no_cp,2))
for i in range(no_cp):
    #check that it has connecting filaments
    fil_index=strt_indx+1
    no_fil=','.join(data.values[int(fil_index)]).split()
    no_fil=int(np.asarray(map(int,list(no_fil))))
    #store the cp type of each cp and the cp ID
    cp_id=','.join(data.values[int(strt_indx)]).split()[5]
    cp_type=','.join(data.values[int(strt_indx)]).split()[0]
    sddl_sv[i]=np.array([cp_id,cp_type])
    
    if no_fil>0:    
    
        cp_pos=','.join(data.values[int(strt_indx)]).split()[1:4]
        cp_id=','.join(data.values[int(strt_indx)]).split()[5]
        #Then store in np array
        cp_store[i,0:3]=cp_pos
        cp_store[i,3]=int(cp_id)
        #get to next cp by reading filament data as to skip        
        strt_indx=strt_indx+int(no_fil)+2

cp_store_fnl=cp_store[np.where(cp_store[:,3]>=0)]#filter out cps with no connecting filaments
halo_cp_dist=distance.cdist(haloes_pos,cp_store_fnl[:,0:3],metric='euclidean')#2D distance matrix
min_dist=np.min(halo_cp_dist,axis=1,keepdims=True)#Find the closest cp to each halo
min_indx=np.where(halo_cp_dist==min_dist)[1]#I can use np.argmin() but for now stick with this
del halo_cp_dist
cp_store_filt=cp_store_fnl[min_indx,3]#the cp id for each halo, in order of halo catalog

fil_loc=np.where(data_arr=="[FILAMENTS]")[0]#these seperate distinct datasets
strt_indx_fil=fil_loc+1
no_fil=','.join(data.values[int(strt_indx_fil)]).split()
no_fil=int(np.asarray(map(int,list(no_fil))))
smp_dict=collections.defaultdict(dict)
dist_dict=collections.defaultdict(dict)

count_1=0
count_2=0
count_3=0
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
        smp_dict[i]=smp_pos#save all relevant filament sampling points
        #find which halo spins corresp. w/ filament cp
        if cp_ex1>0: 
            
            halo_ex1_pos=haloes_pos[cp_ex1_indx]
            halo_ex1_id=halo_id[cp_ex1_indx]
            #save the min dist and smp_point index in dictionary
            fil_halo_dist=dist(halo_ex1_pos,smp_pos,halo_ex1_id)
            no_haloes=int(fil_halo_dist[0])
#            count_1+=no_haloes
            for k in range(no_haloes):            
                dist_dict[int(fil_halo_dist[k+1])][i]=fil_halo_dist[(no_haloes+1+k):(no_haloes+1+k+2)]

        if cp_ex2>0:
            
            halo_ex2_pos=haloes_pos[cp_ex2_indx]
            halo_ex2_id=halo_id[cp_ex2_indx]
            #save the min dist and smp_point index in dictionary
            fil_halo_dist=dist(halo_ex2_pos,smp_pos,halo_ex2_id)
            no_haloes=int(fil_halo_dist[0])
#            count_2+=no_haloes
            for k in range(no_haloes):            
                dist_dict[int(fil_halo_dist[k+1])][i]=fil_halo_dist[(no_haloes+1+k):(no_haloes+1+k+2)]
            
#        if cp_ex1>0 and cp_ex2>0:
#            count_3+=1
    else:#if filament not related, go to next filament.
         strt_indx_fil+=no_smp                                          
 
#now I have the minimum distance for each filament, I can find the absolute minimum for each halo. I need a new loop because i need the thing to run to the end.
#for i in range(len(dist_dict.keys())):
    
print len(dist_dict.keys())    
 

'''
dist_dict[996]
Out[129]: 
{312: array([9.00000000e+00, 5.40273354e+04]),
 318: array([    0.        , 52161.74555037]),
 324: array([8.00000000e+00, 5.40273354e+04]),
 1021: array([    0.       , 52197.1647502])}
There are two issues, the first is the above which shows that this halo is equidistant from two different filaments?
the second issue is that I am getting only len(dist_dict.keys())=161 haloes when I should get 200.
it seems the second issue is not an issue as I am finding that filaments can share sampling points... 

figure out exactly what cp are at the extremes of filaments because just checked that cp type 2 do sit at filament
extremes... weird how 719 as a cp type of 1 with no filaments apparently still has 2 filaments:
    np.where(fil_extr==719)
Out[245]: (array([374, 375]), array([0, 0]))

1 49630 24764.7 1921.57 5.80169e-10 719 0
 0
This implies that there are no filaments connected to it, but then I do find there are 2 filaments connected.
Check the processing that has gone into creating this catalog because perhaps the system has trimmed or something
and does not indicate it in the catalog?????
'''

    
    
 
