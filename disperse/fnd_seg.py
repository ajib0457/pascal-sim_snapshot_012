import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mth
import sklearn.preprocessing as skl
from scipy.spatial import distance
import collections
import random
import h5py

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
spine_dist=10000 #kpc
strt_indx=0
#Find cp of type 3 (maxima) closest to haloes==========================================
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
    
dp_1=dp_1[np.where(dp_1[:,1]>=0)]
exc_indx=np.where(dp_1[:,2]>=0)
dp_2=np.zeros((len(exc_indx[0]),2))
dp_2[:,1]=dp_1[exc_indx,2]
dp_2[:,0]=dp_1[exc_indx,0]  
dp_1_sub=dp_1[:,0:2] 
dist_alsub=np.vstack((dp_1_sub,dp_2)) 

f2=h5py.File("test.h5", 'w')
f2.create_dataset('/test',data=dist_alsub)
f2.close()

#delete all large variables/dicts after using them. and delete non-used diagnostic arrays etc.
#repalce dummy data with catalog using archive code on artemis  
    
'''
mr_min=np.min(dist_alsub[0,:])
mr_max=np.max(dist_alsub[0,:])
results_dp=np.zeros((tot_dist_bins,6))# [Mass_min, Mass_max, dpValue, dpError+,dpError-,bin_centre]

diction_dp={}
def bootstrap(data,hist_bins,sigma_area,w):
    sys.path.insert(0, '/project/GAMNSCM2/funcs')
    from error_func import *
    #Calculating error using bootstrap resampling
    a=np.random.randint(low=0,high=w,size=(runs,w))
    mean_set=np.mean(data[a],axis=1)
    a=np.histogram(mean_set,density=True,bins=hist_bins)
    x=a[1]
    dx=(np.max(x)-np.min(x))/hist_bins
    x=np.delete(x,len(x)-1,0)+dx/2
    y=a[0]
    perc_lo,err_neg,perc_hi,err_pos,bst_val,area=error(x,y,dx,sigma_area)
    return err_neg,err_pos,bst_val,mean_set
diction={}#store raw dp values
#increm moving bin method
bin_wdth=(mr_max-mr_min)/(2*(1-(bin_overlap/100.0))+(tot_dist_bins-2)*(1-(2*bin_overlap/100.0))+(tot_dist_bins-1)*(bin_overlap/100.0))
len_move=(1-bin_overlap/100.0)*bin_wdth
no_dist=np.zeros((tot_dist_bins))    
for dist_bin in range(tot_dist_bins):
    low_int=mr_min+dist_bin*len_move
    hi_int=low_int+bin_wdth+0.000000001

    results_dp[dist_bin,0]=low_int#Store interval
    results_dp[dist_bin,1]=hi_int#Store interval    
    results_dp[dist_bin,5]=low_int+(hi_int-low_int)/2 #bin value, at the end to avoid changing other
    
    bn=np.logical_and(dist_alsub[0,:]>=low_int,dist_alsub[0,:]<hi_int)
    store_dp=dist_alsub[1,bn]
    diction[dist_bin]=store_dp
    #l,w=np.shape(store_dp)
    w=len(store_dp)
    no_dist[dist_bin]=w#store qty of bin value for plot

    #bootstrap ml val and error
    results_dp[dist_bin,4],results_dp[dist_bin,3],results_dp[dist_bin,2],diction_dp[dist_bin]=bootstrap(store_dp,hist_bins,sigma_area,w)
    
lss_type='_'.join(map(str, lss_type))
filehandler = open('/scratch/GAMNSCM2/halo_lss/%s/%s/resultsdp_%s_%s_%s_%s_%s_lss%s_dpw%s_bins%s.pkl'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,smooth_scl,lss_type,dp_with,tot_dist_bins),"wb")       
pickle.dump(results_dp,filehandler)
filehandler.close()

plt.figure()#plot dp vs mr
ax2=plt.subplot2grid((1,1), (0,0))
ax2.axhline(y=0.5, xmin=0, xmax=15, color = 'k',linestyle='--')
if dp_with==6: plt.ylabel('<|J$\cdot$$e_{3}$|>')
if dp_with==3: plt.ylabel('<|V$\cdot$$e_{3}$|>')
plt.xlabel('$log_{10}[M_{\odot}]$')   
plt.title('%s_%s_%s_%s_smth_scl:%sMpc/h'%(sim_type,cosmology,snapshot,particles_filt,smooth_scl))
ax2.plot(results_dp[:,0],results_dp[:,2],'g-')
ax2.fill_between(results_dp[:,0], results_dp[:,2]-abs(results_dp[:,4]), results_dp[:,2]+abs(results_dp[:,3]),facecolor='green',alpha=0.3)
#plt.legend()
plt.savefig('/scratch/GAMNSCM2/halo_lss/%s/%s/halodp%s_%s_%s_%s_smthscl%s_lss%s_dpw%s_bins%s.png'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,smooth_scl,lss_type,dp_with,tot_dist_bins))

c_samples={}#dictionary for output data
results=np.zeros((tot_dist_bins,6))#[Mass_min, Mass_max, Value, Error+,Error-,bin_centre]
no_halos=np.zeros((len(diction),1)) 
results[:,0]=results_dp[:,0]
results[:,1]=results_dp[:,1]   
results[:,5]=results_dp[:,5]
for mass_bins in range(tot_dist_bins):

    dot_val=diction[mass_bins]
    no_halos[mass_bins]=len(dot_val)
    c_samples[mass_bins],results[mass_bins,2],results[mass_bins,3],results[mass_bins,4]=grid_mthd(dot_val,grid_density,sigma_area)
method='grid'   #Not optional for this code
dp_mthd='increm'
posterior_plt(cosmology,c_samples,results,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,dp_with,grid_density=grid_density,no_halos=no_halos,bin_overlap=bin_overlap)     
# posterior_plt(cosmology,diction_2,results,bins,     sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,dp_with,**args): 

mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,dp_bins=15,bin_overlap=bin_overlap,dp_with=dp_with)
# mod_data_ovrplt(cosmology,diction,results,sim_sz,grid_nodes,smooth_scl,tot_mass_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,**args):
 
   
filehandler = open('/scratch/GAMNSCM2/halo_lss/%s/%s/myden_gridresults_LSS%s_snap%s_vec%s_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s_overlap%sperc.pkl'%(sim_type,cosmology,lss_type,snapshot,dp_with,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,dp_mthd,bin_overlap),"wb")
pickle.dump(results,filehandler)
filehandler.close()  
'''
