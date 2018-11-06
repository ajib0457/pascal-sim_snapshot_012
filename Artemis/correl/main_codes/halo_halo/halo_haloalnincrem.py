import numpy as np
import h5py
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
import sys
import pickle
sim_type=sys.argv[1]   #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=sys.argv[3]            #'12  '11'...
particles_filt=int(sys.argv[4])
den_type=sys.argv[5]      #'DTFE' 'my_den'
sim_sz=500           #Size of simulation in physical units Mpc/h cubed
smooth_scl=sys.argv[6]       #Smoothing scale in physical units Mpc/h
grid_nodes=1250      #Density Field grid resolution
runs=20000
hist_bins=300
tot_dist_bins=17
sigma_area=0.3413    #sigma percentile area as decimal
bin_overlap=60
lss_type=[3,2,1,0]           #Cluster-3 Filament-2 Sheet-1 Void-0
f=h5py.File("/scratch/GAMNSCM2/halo_halo_lss/%s/%s/%s_%s_snap%s_smth%s_%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_mask_exyz.h5"%(sim_type,cosmology,sim_type,cosmology,snapshot,smooth_scl,den_type), 'r')
data=f['/halo_lss'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()
#Filter out halos with N particles
partcl_halo_flt=np.where(data[:,11]>=particles_filt)#filter for halos with <N particles
data=data[partcl_halo_flt]#Filter out halos with <N particles

#filter out which LSS you want
x,y=np.shape(data)
filt_data=np.zeros(y)
for i in lss_type:
    d=data[np.where(data[:,12]==i)]
    filt_data=np.vstack((filt_data,d))
data=np.delete(filt_data,(0),axis=0)
del filt_data

a_len=len(data)
#a_len=1000
dist_sp=distance.pdist(data[:,0:3], 'euclidean')
#dist_sp=distance.pdist(data[0:a_len,0:3], 'euclidean')

#range of halo-halo alignment signal
dist_min=10**(-1.2)
dist_max=10**(0.8)#Specific as Trowland+12 is.
#Filter out entire range to save memory
bn=np.logical_and(dist_sp>=dist_min,dist_sp<dist_max)
a=np.where(bn==True)
a=np.asarray(a)
log_dist=np.vstack((np.log10(dist_sp[a]),a))
del dist_sp
#log of distance and extremes
dist_min=np.min(log_dist[0,:])
dist_max=np.max(log_dist[0,:])
dist_intvl=(dist_max-dist_min)/tot_dist_bins#log_mass value used to find mass interval
results_dp=np.zeros((tot_dist_bins,6))# [Mass_min, Mass_max, dpValue, dpError+,dpError-,bin_centre]
results_mr=np.zeros((tot_dist_bins,6))# [Mass_min, Mass_max, mrValaue, mrError+,mrError-,bin_centre]

diction_dp={}
diction_mr={}
def bootstrap(data,hist_bins,sigma_area):
    sys.path.insert(0, '/project/GAMNSCM2/funcs')
    from error_func import *
    #Calculating error using bootstrap resampling
    a=np.random.randint(low=0,high=len(data),size=(runs,len(data)))
    mean_set=np.mean(data[a],axis=1)
    a=np.histogram(mean_set,density=True,bins=hist_bins)
    x=a[1]
    dx=(np.max(x)-np.min(x))/hist_bins
    x=np.delete(x,len(x)-1,0)+dx/2
    y=a[0]
    perc_lo,err_neg,perc_hi,err_pos,bst_val,area=error(x,y,dx,sigma_area)
    return err_neg,err_pos,bst_val,mean_set

#increm moving bin method
bin_wdth=(dist_max-dist_min)/(2*(1-(bin_overlap/100.0))+(tot_dist_bins-2)*(1-(2*bin_overlap/100.0))+(tot_dist_bins-1)*(bin_overlap/100.0))
len_move=(1-bin_overlap/100.0)*bin_wdth
no_dist=np.zeros((tot_dist_bins))    
for dist_bin in range(tot_dist_bins):
    low_int=dist_min+dist_bin*len_move
    hi_int=low_int+bin_wdth+0.000000001

    results_dp[dist_bin,0]=low_int#Store interval
    results_dp[dist_bin,1]=hi_int#Store interval    
    results_dp[dist_bin,5]=low_int+(hi_int-low_int)/2 #bin value, at the end to avoid changing other
    results_mr[dist_bin,:]=results_dp[dist_bin,:]
    bn=np.logical_and(log_dist[0,:]>=low_int,log_dist[0,:]<hi_int)
    a=np.where(bn==True)
    a=np.asarray(a)
    l,w=np.shape(a)
    no_dist[dist_bin]=w
    #Find halos associated with distance range
    indxs=np.zeros((w))
    indxd=np.zeros((w))
    for j in range(w): 
        i=0
        div=log_dist[1,a[0,j]]-(a_len-1)    
        while div>=0:
            i+=1
            div=div-(a_len-1-i)
        indxs[j]=i    
        indxd[j]=indxs[j]+1+div+(a_len-1-i)
        
    indxs=indxs.astype(int)#the indices for the halos
    indxd=indxd.astype(int) 
    #Now take the dotproduct and then fit a line, make the 3 plots Trowland+13 makes.
    #normalize vectors
    vecs_indxs=skl.normalize(data[indxs,6:9])
    vecs_indxd=skl.normalize(data[indxd,6:9])
    #calculate mass ratio
    mass_indxs=data[indxs,9]
    mass_indxd=data[indxd,9]
    store_mratio=mass_indxs/mass_indxd
    zer=np.where(store_mratio<1)
    store_mratio[zer]=1/store_mratio[zer]

    #take dp
    store_dp=np.zeros(len(vecs_indxs))
    for i in range(len(vecs_indxs)):
        store_dp[i]=np.inner(vecs_indxs[i,:],vecs_indxd[i,:])#take the dot product between vecs, row by row   
    store_dp=abs(store_dp) 
    #bootstrap ml val and error
    results_dp[dist_bin,4],results_dp[dist_bin,3],results_dp[dist_bin,2],diction_dp[dist_bin]=bootstrap(store_dp,hist_bins,sigma_area)
    results_mr[dist_bin,4],results_mr[dist_bin,3],results_mr[dist_bin,2],diction_mr[dist_bin]=bootstrap(store_mratio,hist_bins,sigma_area)
lss_type='_'.join(map(str, lss_type))
filehandler = open('/scratch/GAMNSCM2/halo_halo_plts/%s/%s/resultsdp_%s_%s_%s_%s_lsstype%s_increm%s_bns%s.pkl'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,lss_type,bin_overlap,tot_dist_bins),"wb")       
pickle.dump(results_dp,filehandler)
filehandler.close()

filehandler = open('/scratch/GAMNSCM2/halo_halo_plts/%s/%s/resultsmr_%s_%s_%s_%s_lsstype%s_increm%s_bns%s.pkl'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,lss_type,bin_overlap,tot_dist_bins),"wb")       
pickle.dump(results_mr,filehandler)
filehandler.close()

plt.figure()#plot original dp vs dist
ax2=plt.subplot2grid((1,1), (0,0))
ax2.axhline(y=0.5, xmin=0, xmax=15, color = 'k',linestyle='--')
plt.ylabel('<J(x).J(x+r)>')
plt.xlabel('log r[Mpc/h]')   
plt.title('%s_%s_%s_%s'%(sim_type,cosmology,snapshot,particles_filt))
plt.legend(loc='upper right')
ax2.plot(results_dp[:,0],results_dp[:,2],'g-',label='spin_spin')
ax2.fill_between(results_dp[:,0], results_dp[:,2]-abs(results_dp[:,4]), results_dp[:,2]+abs(results_dp[:,3]),facecolor='green',alpha=0.3)
plt.savefig('/scratch/GAMNSCM2/halo_halo_plts/%s/%s/halohalodpincrem%s_%s_%s_%s_lsstype%s_newest_test.png'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,lss_type))

#plt.figure()#plot mr vs dp
#ax2=plt.subplot2grid((1,1), (0,0))
#plt.ylabel('<J(x).J(x+r)>')
#plt.xlabel('$M_{1}/M_{2}$')
#plt.axhline(y=0.5, xmin=0, xmax=100, color = 'k',linestyle='--')
#plt.axvline(x=1.0, ymin=0, ymax=15, color = 'k',linestyle='--')
#plt.errorbar(results_mr[:,2],results_dp[:,2],[results_mr[:,4],results_mr[:,3]],[results_dp[:,4],results_dp[:,3]])
#plt.title('%s_%s_%s_%s'%(sim_type,cosmology,snapshot,particles_filt))
##plt.legend(loc='upper right')
#plt.savefig('/scratch/GAMNSCM2/halo_halo_plts/%s/%s/halohalodpmrincrem%s_%s_%s_%s_lsstype%smrmorethanone.png'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,lss_type))


method='bootstrap'   #Not optional for this code
dp_mthd='increm'             # 'increm', 'hiho' & 'subdiv'
smooth_scl='na'       #Smoothing scale in physical units Mpc/h. 2  3.5  5

sys.path.insert(0, '/project/GAMNSCM2/main_codes/correl/halo_halo/funcs') 
from plotter_funcs import *

posterior_plt(cosmology,diction_dp,results_dp,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,no_dist=no_dist,data_type='spindp_vs_dist_newest_test')
posterior_plt(cosmology,diction_mr,results_mr,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,no_dist=no_dist,data_type='mr_vs_dist_newest_test')
