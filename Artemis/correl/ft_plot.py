import pickle
import numpy as np
import matplotlib.pyplot as plt
from error_func import *
#may only for work dm_only as npart for dm_gas is unclear as two types of particles exist.
sim_type='dm_only'   #'dm_only' 'DTFE'
cosmology='lcdm'          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'

filehandler = open('ft_dict_%s_%s.pkl'%(sim_type,cosmology),"rb")
diction=pickle.load(filehandler)#[id,npart](011), [id,npart](010)...
filehandler.close()

z=np.array([0.0,0.26,0.59,1.0,1.5,2.16,2.98])
ms_ft=np.zeros((len(diction),2))
mr_test=np.zeros(len(diction))
for i in range(len(diction)):
    #FIrst save mass ratio of parent and progenitor
    hlo_tm=len(diction[i])
    mr_test[i]=1.0*diction[i][0][1]/diction[i][hlo_tm-1][1] 
    #Find formation time and mass at z=0
    ms_ft[i,1]=z[hlo_tm-1]
    ms_ft[i,0]=diction[i][0][1]*8.16823864*10**10

cull_hls_indx=np.where(mr_test>=2)#filter out progenitor halos with more than half z=0 mass
ms_ft=ms_ft[cull_hls_indx]



results=np.zeros((len(z),4))# [Value, Error+,Error-,bin_centre] 
#for redshft in z:
    #now bin and take average of mass in each z bin.
#    perc_lo,results[mass_bin,4],perc_hi,results[mass_bin,3],results[mass_bin,2],area=error(x,y,dx,sigma_area)  
