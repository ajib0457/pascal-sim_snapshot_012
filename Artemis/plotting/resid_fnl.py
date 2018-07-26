import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import sys
sys.path.insert(0, '/project/GAMNSCM2/funcs')
from signal_funcs import * 

snapshot='08'                             #'12  '11' 
method='grid'                             #'grid', 'mcmc', 'bootstrap'
sim_sz=500                                #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250                           #Density Field grid resolution
smooth_scales=[2,3.5,5]                         #Smoothing scale in physical units Mpc/h
mass_bins=[20,20,20]                            #Number of Halo mass bins
dp_mthd='increm'                          # 'increm', 'hiho' & 'subdiv'
particles_filt=100                        #Halos to filter out based on number of particles, ONLY for Dot Product Spin-LSS(SECTION 5.)
lss_type=2                                #Cluster-3 Filament-2 Sheet-1 Void-0
#grid method initial cond.
grid_density=11000                        #density c value parameter space ranging from -0.99 to 0.99
sim_type='dm_only'                        #'dm_gas' 'dm_only'
bin_overlap=[90,90,90]                          #'increm' bins overlap in % (percentage)
if sim_type=='dm_only':
    cosmology=['lcdm','cde0','wdm2']      #DM_ONLY
    den_type='DTFE'                       #'my_den' 'DTFE'
if sim_type=='dm_gas':    
    cosmology=['lcdm','cde000','cde099']  #DM_GAS
    den_type='my_den'                     #'my_den' 'DTFE'
fiducial_cos='lcdm'                       #Cosmology to be as the reference for residuals
plt_buffr=0.3                             #Buffer for x_axis 
#load data
d=collections.defaultdict(dict)
diction=collections.defaultdict(dict)
for cos in cosmology: 
    i=0
    for smth_scl in smooth_scales:
        no_mass_bins=mass_bins[i]
        overlap=bin_overlap[i]
        if dp_mthd=='subdiv': 
            filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s.pkl'%(sim_type,cos,snapshot,den_type,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd),"rb")
        if dp_mthd=='increm': 
            filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s_overlap%sperc.pkl'%(sim_type,cos,snapshot,den_type,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd,overlap),"rb")
        if dp_mthd=='hiho': 
            filehandler = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/mod_fit/grid_mthd/myden_gridresults_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s.pkl'%(sim_type,cos,snapshot,den_type,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd),"rb")
        
        d[cos][smth_scl]=pickle.load(filehandler)
        filehandler.close() 
      
        if dp_mthd=='subdiv': 
            filehandler_dotprod = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s.pkl'%(sim_type,cos,snapshot,den_type,cos,snapshot,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd),"rb")
        if dp_mthd=='increm': 
            filehandler_dotprod = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s_overlap%sperc.pkl'%(sim_type,cos,snapshot,den_type,cos,snapshot,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd,overlap),"rb")
        if dp_mthd=='hiho': 
            filehandler_dotprod = open('/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/dotproduct/spin_lss/%s_snapshot_0%s_dp_LSS%s_spin_sim%sMpc_grid%s_smth%sMpc_%sbins_partclfilt%s_dpmthd_%s.pkl'%(sim_type,cos,snapshot,den_type,cos,snapshot,lss_type,sim_sz,grid_nodes,smth_scl,no_mass_bins,particles_filt,dp_mthd),"rb")
        
        diction[cos][smth_scl]=pickle.load(filehandler_dotprod)
        filehandler_dotprod.close()

        i+=1    

#plotting first row, which are the signals, including the fiducial
min_maxs=np.zeros((len(smooth_scales),2))#[min,max]
color=['green','blue','red','yellow']
line_color=['g-','b-','r-','y-']
plt.figure(figsize=(5*len(smooth_scales),15))
plt.suptitle('%s-Method Mode Fitted snapshot_0%s dpmethod_%s overlap_%sperc'%(method,snapshot,dp_mthd,bin_overlap))
i=0
for smth_scl in smooth_scales: 
    bin_cent=[]
    j=0
    ax1=plt.subplot2grid((4,len(smooth_scales)), (0,i))
    ax1=plt.axhline(y=0, xmin=0, xmax=15, color = 'black',linestyle='--')
    if i==1: ax1=plt.xlabel('$log_{10}(M_\odot)$')
    if i==0: ax1=plt.ylabel('c')
    plt.title('%sMpc/h(%s bins)'%(smth_scl,mass_bins[i]))    
    plt.ylim(-0.2,1)
    if i>0: plt.yticks([])
    plt.xticks([])
    for cos in cosmology:     
        if dp_mthd=='increm':
            ax1=plt.plot(d[cos][smth_scl][:,5],d[cos][smth_scl][:,2],line_color[j],label=cos)# [Mass_min, Mass_max, Value, Error+,Error-,bin_centre]
            ax1=plt.fill_between(d[cos][smth_scl][:,5], d[cos][smth_scl][:,2]-abs(d[cos][smth_scl][:,4]), d[cos][smth_scl][:,2]+abs(d[cos][smth_scl][:,3]),facecolor=color[j],alpha=0.3)                                                                                                                                                                                                                                      
            j+=1 
            bin_cent.append(d[cos][smth_scl][:,5])            
    min_maxs[i,0]=np.min(bin_cent)
    min_maxs[i,1]=np.max(bin_cent)    
    plt.xlim(min_maxs[i,0]-plt_buffr,min_maxs[i,1]+plt_buffr) 
    if i==0: plt.legend(loc=1)
    i+=1 

#I have to convert errors into real units, that is not the width of errors but into coordiantes of the errors.
for smth_scl in smooth_scales:
    for cos in cosmology:
        d[cos][smth_scl][:,4]=d[cos][smth_scl][:,2]-abs(d[cos][smth_scl][:,4])
        d[cos][smth_scl][:,3]=d[cos][smth_scl][:,2]+abs(d[cos][smth_scl][:,3])
        
#Calculate signal residuals
signal_resids=collections.defaultdict(dict)
if dp_mthd=='increm':    
    cosmology.remove(fiducial_cos) 
    for cos in cosmology: 
        for smth_scl in smooth_scales:
            
            res,err_pos,err_neg=calc_resid(d[fiducial_cos][smth_scl][:,5],d[fiducial_cos][smth_scl][:,2],d[fiducial_cos][smth_scl][:,3],d[fiducial_cos][smth_scl][:,4],d[cos][smth_scl][:,5],d[cos][smth_scl][:,2],d[cos][smth_scl][:,3],d[cos][smth_scl][:,4])             
            signal_resids[cos][smth_scl]= np.column_stack((res,err_pos,err_neg))# [res(2 cols),err_pos(1 cols),err_neg(1 cols)]                       

#plot signal residuals
i=0
for smth_scl in smooth_scales: 
    j=0
    ax1=plt.subplot2grid((4,len(smooth_scales)), (1,i))
    ax1=plt.axhline(y=0, xmin=0, xmax=15, color = 'black',linestyle='--')
    if i==0: ax1=plt.ylabel('|c residuals|')    
    plt.ylim(-0.2,1)
    plt.xlim(min_maxs[i,0]-plt_buffr,min_maxs[i,1]+plt_buffr)   
    if i>0: plt.yticks([])
    plt.xticks([])    
    for cos in cosmology:     
        if dp_mthd=='increm':
            ax1=plt.plot(signal_resids[cos][smth_scl][:,0],signal_resids[cos][smth_scl][:,1],line_color[j+1],label=cos)
            ax1=plt.fill_between(signal_resids[cos][smth_scl][:,0], signal_resids[cos][smth_scl][:,3], signal_resids[cos][smth_scl][:,2],facecolor=color[j+1],alpha=0.3)                                                                                                                                                                                                               
            j+=1    
    if i==0: plt.legend(loc=1)
    i+=1 

#calculate no. halos
no_halos=collections.defaultdict(dict)

cosmology.insert(0,fiducial_cos)

for cos in cosmology:   
    i=0
    for smth_scl in smooth_scales:
        no_mass_bins=mass_bins[i]
        mass_bins_temp=np.zeros((no_mass_bins))
        for j in range(no_mass_bins):
            mass_bins_temp[j]=len(diction[cos][smth_scl][j])            
        no_halos[cos][smth_scl]=mass_bins_temp
        i+=1
                                           
#plot no. halos
i=0
for smth_scl in smooth_scales: 
    j=0
    ax1=plt.subplot2grid((4,len(smooth_scales)), (2,i))
    ax1=plt.axhline(y=0, xmin=0, xmax=15, color = 'black',linestyle='--')
    if i==0: ax1=plt.ylabel('no. halos')    
    
    plt.xlim(min_maxs[i,0]-plt_buffr,min_maxs[i,1]+plt_buffr)   
    if i>0: plt.yticks([])
    plt.xticks([])    
    for cos in cosmology:     
        if dp_mthd=='increm':
            ax1=plt.plot(d[cos][smth_scl][:,5],no_halos[cos][smth_scl],line_color[j],label=cos)
            j+=1    
    if i==0: plt.legend(loc=1)
    i+=1 

#calculate no. halos residuals
cosmology.remove(fiducial_cos)
halo_no_resids=collections.defaultdict(dict)
for cos in cosmology: 

    for smth_scl in smooth_scales:
        a_b_res,a_b_x=subtract_grad(d[fiducial_cos][smth_scl][:,5],no_halos[fiducial_cos][smth_scl],d[cos][smth_scl][:,5],no_halos[cos][smth_scl],sig_type='signal')
        b_a_res,b_a_x=subtract_grad(d[cos][smth_scl][:,5],no_halos[cos][smth_scl],d[fiducial_cos][smth_scl][:,5],no_halos[fiducial_cos][smth_scl],sig_type='signal')
        res_tot_y=np.hstack((a_b_res,b_a_res))
        res_tot_x=np.hstack((a_b_x,b_a_x))
        res_nohalos=np.column_stack((res_tot_x,res_tot_y))#x,y columns
        halo_no_resids[cos][smth_scl] = res_nohalos[res_nohalos[:,0].argsort()]

#plot no. halo residuals
i=0
for smth_scl in smooth_scales: 
    j=0
    ax1=plt.subplot2grid((4,len(smooth_scales)), (3,i))
    ax1=plt.axhline(y=0, xmin=0, xmax=15, color = 'black',linestyle='--')    
    if i==0: ax1=plt.ylabel('|no. halos residuals|')    
    if i==1: ax1=plt.xlabel('$log_{10}(M_\odot)$')
    plt.xlim(min_maxs[i,0]-plt_buffr,min_maxs[i,1]+plt_buffr)   
    if i>0: plt.yticks([])    
    for cos in cosmology:     
        if dp_mthd=='increm':
            ax1=plt.plot(halo_no_resids[cos][smth_scl][:,0],halo_no_resids[cos][smth_scl][:,1],line_color[j+1],label=cos)
            j+=1    
    if i==0: plt.legend(loc=1)
    i+=1 
plt.subplots_adjust(wspace=0)
plt.subplots_adjust(hspace=0)

smth='_'.join(map(str, smooth_scales))
overlap='_'.join(map(str, bin_overlap))
mass='_'.join(map(str, mass_bins))
plt.savefig('/scratch/GAMNSCM2/combined_plts/pascal_modfitted_residuals_%s_snapshot_0%s_%s_fitmethod_%s_lss%s_grid%s_particles%s_smth%s_mass%s_dpmthd_%s_%s_perc.png'%(sim_type,snapshot,method,den_type,lss_type,grid_nodes,particles_filt,smth,mass,dp_mthd,overlap))    

