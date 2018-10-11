import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs')
from signal_funcs import * 
plt.rcParams['axes.grid'] = True

sim_type='dm_gas'                        #'dm_gas' 'dm_only'
snapshot=['08','09','10','11']  #'12  '11' 
method='boostrap'                             #'grid', 'mcmc', 'bootstrap'
sim_sz=500                                #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250                           #Density Field grid resolution

particles_filt=100                        #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
#grid method initial cond.
if sim_type=='dm_only':
    cosmology=['lcdm','wdm2','cde0']      #DM_ONLY
    den_type='DTFE'                       #'my_den' 'DTFE'
if sim_type=='dm_gas':    
    cosmology=['lcdm','cde000','cde050','cde099']  #DM_GAS
    den_type='my_den'                     #'my_den' 'DTFE'
fiducial_cos='lcdm'                       #Cosmology to be as the reference for residuals

diction=collections.defaultdict(dict)
for snap in snapshot:
    
    for cos in cosmology:
        filehandler = open('/import/oth3/ajib0457/pascal_sims/halo_halo_plts/%s/%s/results_%s_%s_%s_%s.pkl'%(sim_type,cos,sim_type,cos,snap,particles_filt),"rb")       
        diction[snap][cos]=pickle.load(filehandler)
        filehandler.close()
'''

diction=collections.defaultdict(dict)
for snap in snapshot:
    for cos in cosmology:
        diction[snap][cos]=np.random.rand(9,5)
'''        
color=['green','blue','red','yellow']
line_color=['g-','b-','r-','y-']
plt.figure(figsize=(30,15))
plt.suptitle('halo-halo alignment %s partcls %s'%(sim_type,particles_filt),fontsize=15)
l=0
indx=np.array([[0,0,0,1,1,1,2],[0,1,2,0,1,2,0]])
for snap in snapshot:
    k=0
    ax2=plt.subplot2grid((3,3), (int(indx[1,l]),int(indx[0,l])))
    ax2.axhline(y=0.5, xmin=0, xmax=15, color = 'k',linestyle='--')
    plt.title('snapshot %s'%snap)
    for cos in cosmology:

        plt.legend(loc='upper right')
        
        ax2.plot(diction[snap][cos][:,0],diction[snap][cos][:,2],line_color[k],label=cos)
        ax2.fill_between(diction[snap][cos][:,0], diction[snap][cos][:,2]-abs(diction[snap][cos][:,4]), diction[snap][cos][:,2]+abs(diction[snap][cos][:,3]),facecolor=color[k],alpha=0.3)
        plt.legend()        
        k+=1
        plt.xlim(-1.3,1)
        plt.ylim(0.45,0.65)
        if l==2: ax1=plt.xlabel('$log_{10}$(r[Mpc/h])',fontsize=20)
        if l==1: ax1=plt.ylabel('$\langle$|$\mathbf{J}$(x)$\cdot$$\mathbf{J}$(x+r)|$\langle$',fontsize=20)
    l+=1        

plt.savefig('halo_halo_%s.png'%sim_type)
       
    

    
