import pickle
import numpy as np
import matplotlib.pyplot as plt
from error_func import *
#may only for work dm_only as npart for dm_gas is unclear as two types of particles exist.
sim_type='dm_only'   #'dm_only' 'DTFE'
cosmology='lcdm'          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
runs=100000          #bootstrap resampling runs
method='bootstrap'   #Not optional for this code
hist_bins=200        #histogram to form gaussian pdf
sigma_area=0.3413    #sigma percentile area as decimal
filehandler = open('ft_dict_%s_%s.pkl'%(sim_type,cosmology),"rb")
diction=pickle.load(filehandler)#[id,npart](011), [id,npart](010)...
filehandler.close()

z=np.array([0.0,0.26,0.59,1.0,1.5,2.16,2.98])
ms_ft=np.zeros((len(diction),2))
mr_test=np.zeros(len(diction))
j=0
for i in diction.keys():
    #FIrst save mass ratio of parent and progenitor
    hlo_tm=len(diction[i])
    mr_test[j]=1.0*diction[i][0][1]/diction[i][hlo_tm-1][1] 
    #Find formation time and mass at z=0
    ms_ft[j,1]=z[hlo_tm-1]
    ms_ft[j,0]=diction[i][0][1]*8.16823864*10**10
    j+=1
    
cull_hls_indx=np.where(mr_test>=2)#filter out progenitor halos with more than half z=0 mass
ms_ft=ms_ft[cull_hls_indx]

z_det=np.sort(np.unique(ms_ft[:,1]))
results=np.zeros((len(z_det),4))# [Value, Error+,Error-,bin_centre] 
i=0
for redshift in z_det:
    results[i,3]=z_det[i]
    red_indx=np.where(ms_ft[:,1]==redshift)
    store_dp=np.log10(ms_ft[red_indx,0])

    #Calculating error using bootstrap resampling
    a=np.random.randint(low=0,high=len(store_dp),size=(runs,len(store_dp)))
    mean_set=np.mean(store_dp[a],axis=1)
    
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
plt.ylabel('$log_{10}(M_\odot)$')
#plt.ylabel('$(M_\odot)$')

