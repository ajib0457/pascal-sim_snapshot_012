import numpy as np
import h5py
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
import sys
import pickle
sim_type='dm_only'         #'dm_only' 'dm_gas'
cosmology='lcdm'          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot='11'            #'12  '11'...
den_type='DTFE'           #'DTFE' 'my_den'
smooth_scl=3.5             #Smoothing scale in physical units Mpc/h. 2  3.5  5
tot_mass_bins=12      #Number of Halo mass bins MIN=3 for 'increm'
particles_filt=100   #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
bin_overlap=90  #'increm' bins overlap in % (percentage) 
dp_with=6       #spin(6) or velocity(3)
sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250      #Density Field grid resolution
lss_type=[2]           #Cluster-3 Filament-2 Sheet-1 Void-0
runs=100000          #bootstrap resampling runs
method='bootstrap'   #Not optional for this code
hist_bins=200        #histogram to form gaussian pdf
sigma_area=0.3413    #sigma percentile area as decimal
tot_dist_bins=17

f=h5py.File("/scratch/GAMNSCM2/halo_halo_lss/%s/%s/%s_%s_snap%s_smth%s_%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_mask_exyz.h5"%(sim_type,cosmology,sim_type,cosmology,snapshot,smooth_scl,den_type), 'r')
data=f['/halo_lss'][:]#halos array: (Pos)XYZ(Mpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
f.close()

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

#find distance between ALL halos up to this point
a_len=len(data)
dist_sp=distance.pdist(data[:,0:3], 'euclidean')

#range of halo-halo alignment signal
dist_min=10**(-1.2)
dist_max=10**(0.6)#Specific as Trowland+12 is.
#Filter out entire range to save memory
bn=np.logical_and(dist_sp>=dist_min,dist_sp<dist_max)
#del dist_sp
a=np.where(bn==True)
a=np.asarray(a)
l,w=np.shape(a)
#Find halos associated with distance range
indxs=np.zeros((w))
indxd=np.zeros((w))
for j in range(w): 
    i=0
    div=a[0,j]-(a_len-1)    
    while div>=0:
        i+=1
        div=div-(a_len-1-i)
    indxs[j]=i    
    indxd[j]=indxs[j]+1+div+(a_len-1-i)

indxs=indxs.astype(int)#the indices for the halos
indxd=indxd.astype(int) 

#now take the halos and find the dp between halo-lss, then subtract and save.
lxyz_indxs=skl.normalize(data[indxs,6:9])#Lxyz
lxyz_indxd=skl.normalize(data[indxd,6:9])
exyz_indxs=skl.normalize(data[indxs,13:16])#exyz
exyz_indxd=skl.normalize(data[indxd,13:16])
store_alsub=np.zeros(len(indxd))
for i in range(len(indxd)):
    indxs_dp=abs(np.inner(lxyz_indxs[i,:],exyz_indxs[i,:]))#take the dot product between vecs, row by row 
    indxd_dp=abs(np.inner(lxyz_indxd[i,:],exyz_indxd[i,:]))#take the dot product between vecs, row by row 
    store_alsub[i]=indxs_dp-indxd_dp
store_alsub=abs(store_alsub)

dist_alsub=np.vstack((np.log10(dist_sp[a]),store_alsub))

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
    #l,w=np.shape(store_dp)
    w=len(store_dp)
    no_dist[dist_bin]=w#store qty of bin value for plot

    #bootstrap ml val and error
    results_dp[dist_bin,4],results_dp[dist_bin,3],results_dp[dist_bin,2],diction_dp[dist_bin]=bootstrap(store_dp,hist_bins,sigma_area,w)

filehandler = open('/scratch/GAMNSCM2/halo_halo_lss/%s/%s/resultsdpminusdist_%s_%s_%s_%s.pkl'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt),"wb")       
pickle.dump(results_dp,filehandler)
filehandler.close()
dist_min,dist_max=round(dist_min,3),round(dist_max,3)
plt.figure()#plot dp vs mr
ax2=plt.subplot2grid((1,1), (0,0))
ax2.axhline(y=0, xmin=0, xmax=15, color = 'k',linestyle='--')
plt.ylabel('<||J(x)$\cdot$$e_{3}$|-|J(x+r)$\cdot$$e_{3}$||>')
plt.xlabel('log r[Mpc/h]')   
plt.title('%s_%s_%s_%s_smth_scl:%sMpc/h'%(sim_type,cosmology,snapshot,particles_filt,smooth_scl))
ax2.plot(results_dp[:,0],results_dp[:,2],'g-',label='distance: %s - %s Mpc/h'%(dist_min,dist_max))
ax2.fill_between(results_dp[:,0], results_dp[:,2]-abs(results_dp[:,4]), results_dp[:,2]+abs(results_dp[:,3]),facecolor='green',alpha=0.3)
plt.legend()
plt.savefig('/scratch/GAMNSCM2/halo_halo_lss/%s/%s/halodpminusdist%s_%s_%s_%s_smthscl%s.png'%(sim_type,cosmology,sim_type,cosmology,snapshot,particles_filt,smooth_scl))

#lss_type=[3,2,1,0]           #Cluster-3 Filament-2 Sheet-1 Void-0
method='bootstrap'   #Not optional for this code
dp_mthd='increm'             # 'increm', 'hiho' & 'subdiv'
#smooth_scl='na'       #Smoothing scale in physical units Mpc/h. 2  3.5  5

sys.path.insert(0, '/project/GAMNSCM2/main_codes/correl/halo_halo_lss/funcs') 
from plotter_funcs import *
posterior_plt(cosmology,diction_dp,results_dp,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_dist_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,no_dist=no_dist,data_type='spindp_vs_mr')
