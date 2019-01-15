import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from error_func import *
#may only for work dm_only as npart for dm_gas is unclear as two types of particles exist.
sim_type=sys.argv[1]   #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
runs=100000          #bootstrap resampling runs
method='bootstrap'   #Not optional for this code
hist_bins=200        #histogram to form gaussian pdf
sigma_area=0.3413    #sigma percentile area as decimal
particles_filt=100   #Halos to filter out based on number of particles
den_type=sys.argv[3]           #'DTFE' 'my_den'
smooth_scl=2             #Smoothing scale in physical units Mpc/h. 2  3.5  5
sim_sz=500           #Size of simulation in physical units Mpc/h cubed
grid_nodes='na'      #Density Field grid resolution
#lss_type=int(sys.argv[4])           #Cluster-3 Filament-2 Sheet-1 Void-0
lss_type=[3,2,1,0]
dp_mthd='subdiv'     #the binning nature
snapshot='na'
filehandler = open('/scratch/GAMNSCM2/Treefrog/%s/%s/treefrog_dict_%s_%s_smth%s_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct_mask_exyz.pkl'%(sim_type,cosmology,sim_type,cosmology,smooth_scl),"rb")
diction=pickle.load(filehandler)#[id,npart](011), [id,npart](010)...
filehandler.close()

z=np.array([0.0,0.26,0.59,1.0,1.5,2.16,2.98])
ms_ft=np.zeros((len(diction),2))
mr_test=np.zeros(len(diction))
j=0
for i in diction.keys():
    if np.asarray(diction[i])[0,11]>=particles_filt: #if np.asarray(diction[i])[0,16]==lss_type:
        
        #FIrst save mass ratio of parent and progenitor
        hlo_tm=len(diction[i])
        mr_test[j]=1.0*np.asarray(diction[i])[0,9]/np.asarray(diction[i])[-1,9] 
        #Find formation time and mass at z=0
        ms_ft[j,1]=z[hlo_tm-1]
        ms_ft[j,0]=np.asarray(diction[i])[0,9]
        j+=1
    
cull_hls_indx=np.where(mr_test>=2)#filter out progenitor halos with more than half z=0 mass
ms_ft=ms_ft[cull_hls_indx]

z_det=np.sort(np.unique(ms_ft[:,1]))
results=np.zeros((len(z_det),4))# [Value, Error+,Error-,bin_centre] 
tot_bins=len(z_det)
i=0
diction_dp={}
no_dist=np.zeros((len(z_det))) 
for redshift in z_det:
    results[i,3]=z_det[i]
    red_indx=np.where(ms_ft[:,1]==redshift)
    store_dp=np.log10(ms_ft[red_indx,0]).flatten()
    no_dist[i]=len(store_dp)
    #Calculating error using bootstrap resampling
    a=np.random.randint(low=0,high=len(store_dp),size=(runs,len(store_dp)))
    mean_set=np.mean(store_dp[a],axis=1)
    diction_dp[i]=mean_set
    a=np.histogram(mean_set,density=True,bins=hist_bins)
    x=a[1]
    dx=(np.max(x)-np.min(x))/hist_bins
    x=np.delete(x,len(x)-1,0)+dx/2
    y=a[0]
    perc_lo,results[i,2],perc_hi,results[i,1],results[i,0],area=error(x,y,dx,sigma_area)  
    i+=1

plt.plot()
plt.plot(results[:,3],results[:,0],'g-',label='spin_filament')
plt.fill_between(results[:,3], results[:,0]-abs(results[:,2]), results[:,0]+abs(results[:,1]),facecolor='green',alpha=0.3)

plt.xlabel('z')
plt.ylabel('<$log_{10}(M_\odot)$>')
lss_type='_'.join(map(str, lss_type))
filehandler = open('/scratch/GAMNSCM2/Treefrog/ft_plts/%s/%s/resultsft_%s_%s_lss%s.pkl'%(sim_type,cosmology,sim_type,cosmology,lss_type),"wb")       
pickle.dump(results,filehandler)
filehandler.close()

plt.savefig('/scratch/GAMNSCM2/Treefrog/ft_plts/%s/%s/ft_plot_%s_%s_lss%s.png'%(sim_type,cosmology,sim_type,cosmology,lss_type))

sys.path.insert(0, '/project/GAMNSCM2/plotting/ft_plot/funcs') 
from plotter_funcs import *

posterior_plt(cosmology,diction_dp,results,hist_bins,sim_sz,grid_nodes,smooth_scl,tot_bins,particles_filt,lss_type,method,sim_type,snapshot,den_type,dp_mthd,no_dist=no_dist)

