import numpy as np
import h5py
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
sys.path.insert(0, '/project/GAMNSCM2/funcs') 
from plotter_funcs import *

sim_sz=500         #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250     #Density Field grid resolution
smooth_scl=3.5     #Smoothing scale in physical units Mpc/h
cosmology='lcdm'           #'lcdm'  'cde0'  'wdm2'
snapshot=11                #'12  '11'
total_lss_parts=8

#my_den==============
recon_vecs_x_myden=np.zeros((grid_nodes**3))
recon_vecs_y_myden=np.zeros((grid_nodes**3))
recon_vecs_z_myden=np.zeros((grid_nodes**3))
mask_myden=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):

    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d.h5" %(cosmology,snapshot,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    recon_vecs_x_myden[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y_myden[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z_myden[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d_mask.h5" %(cosmology,snapshot,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    mask_myden[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
recon_vecs_x_myden=np.reshape(recon_vecs_x_myden,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_y_myden=np.reshape(recon_vecs_x_myden,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_z_myden=np.reshape(recon_vecs_x_myden,(grid_nodes,grid_nodes,grid_nodes))
mask_myden=np.reshape(mask_myden,(grid_nodes,grid_nodes,grid_nodes))
f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/lss_assign/my_den/%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_lss.h5"%(cosmology,snapshot,cosmology,snapshot), 'r')
halos_myden=f['/halo_lss'][:]
f.close()

f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/lss_assign/my_den/snapshot_0%s_lss.h5"%(cosmology,snapshot,snapshot), 'r')
partcls_myden=f['/partcls_lss'][:]
f.close()
partcls_myden[:,0:3]=partcls_myden[:,0:3]/1000
#=====================

#DTFE=================
recon_vecs_x_DTFE=np.zeros((grid_nodes**3))
recon_vecs_y_DTFE=np.zeros((grid_nodes**3))
recon_vecs_z_DTFE=np.zeros((grid_nodes**3))
mask_DTFE=np.zeros((grid_nodes**3))
for part in range(total_lss_parts):

    nrows_in=int(1.*(grid_nodes**3)/total_lss_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/total_lss_parts)
    f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d.h5" %(cosmology,snapshot,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    recon_vecs_x_DTFE[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y_DTFE[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z_DTFE[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/correl/my_den/files/output_files/eigvecs/%s_sim%s_recon_vecs_sim%s_smth%sMpc_gd%d_%d_mask.h5" %(cosmology,snapshot,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes,part), 'r')
    mask_DTFE[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
recon_vecs_x_DTFE=np.reshape(recon_vecs_x_DTFE,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_y_DTFE=np.reshape(recon_vecs_x_DTFE,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_z_DTFE=np.reshape(recon_vecs_x_DTFE,(grid_nodes,grid_nodes,grid_nodes)) 
mask_DTFE=np.reshape(mask_DTFE,(grid_nodes,grid_nodes,grid_nodes))   
f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/lss_assign/DTFE/%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_lss.h5"%(cosmology,snapshot,cosmology,snapshot), 'r')
halos_DTFE=f['/halo_lss'][:]
f.close()

f=h5py.File("/scratch/GAMNSCM2/dm_only/%s/snapshot_0%s/lss_assign/DTFE/snapshot_0%s_lss.h5"%(cosmology,snapshot,snapshot), 'r')
partcls_DTFE=f['/partcls_lss'][:]
partcls_DTFE[:,0:3]=partcls_DTFE[:,0:3]/1000
f.close()
#====================

#slices options
slc=625#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
grid_den=1250#density field grid resolution
partcl_thkns=1#Thickness of the particle slice, Mpc
box=np.max(partcls_DTFE[:,y])#subset box length
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn

#Filter particles and halos
halos_myden_new=np.array([0,0,0,0,0,0,0,0,0,0])
halos_DTFE_new=np.array([0,0,0,0,0,0,0,0,0,0])
partcls_DTFE_new=np.array([0,0,0,0])
partcls_myden_new=np.array([0,0,0,0])

for i in range(len(partcls_DTFE)):

    if (lo_lim_partcl<partcls_myden[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_partclsmyden=partcls_myden[i,:]
        partcls_myden_new=np.row_stack((partcls_myden,result_partclsmyden))

    if (lo_lim_partcl<partcls_DTFE[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_partclsDTFE=partcls_DTFE[i,:]
        partcls_DTFE_new=np.row_stack((partcls_DTFE,result_partclsDTFE))       

for i in range(len(halos_DTFE)):
    
    if (lo_lim_partcl<halos_myden[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_halos_myden=halos_myden[i,0:9]#which features do I want from the halos and particles?
        result_halos_myden=np.column_stack((result_halos_myden,halos_myden[i,11]))
        halos_myden_new=np.row_stack((halos_myden,result_halos_myden))

    if (lo_lim_partcl<halos_DTFE[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_halos_DTFE=halos_myden[i,0:9]#which features do I want from the halos and particles?
        result_halos_DTFE=np.column_stack((result_halos_DTFE,halos_DTFE[i,11]))        
        halos_DTFE_new=np.row_stack((halos_DTFE,result_halos_DTFE))
        
halos_myden_new = np.delete(halos_myden, (0), axis=0)
halos_DTFE_new = np.delete(halos_DTFE, (0), axis=0)
partcls_myden_new = np.delete(partcls_myden, (0), axis=0)
partcls_DTFE_new = np.delete(partcls_DTFE, (0), axis=0)
'''
#Begin plotting================================================================
fig, ax = plt.subplots(figsize=(100,14),dpi=100)
no_of_plots=6
no_of_comp=2
#Scatter plot
def colorscatter(data,lss_typ):
    i=0#initiate mask for plot loop
    for color in ['red', 'green', 'blue','yellow']:
        lss_plt_filt=np.where(lss_typ==i)
        lss=['voids','sheets','filaments','clusters']  
        ax.scatter(data[lss_plt_filt,0],data[lss_plt_filt,2],c=color,label=lss[i],alpha=0.9, edgecolors='none')
        i+=1
    
    #ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
    plt.xlabel('x[Mpc/h]') 
    plt.ylabel('y[Mpc/h]')
    plt.title('Color Scatter')
    ax.legend()
    ax.grid(True)
    #ax.set_xlim([0,sim_sz])
    #ax.set_ylim([0,sim_sz])
    
    return
#myden particles
#ax=plt.subplot2grid((no_of_plots,no_of_comp), (0,0))    
#colorscatter(partcls_myden,partcls_myden[:,3])
#DTFE particles
#ax=plt.subplot2grid((no_of_plots,no_of_comp), (0,1))    
#colorscatter(partcls_DTFE,partcls_DTFE[:,3])
#myden halos
ax=plt.subplot2grid((no_of_plots,no_of_comp), (1,0))    
colorscatter(halos_myden,halos_myden[:,9])
#DTFE halos
ax=plt.subplot2grid((no_of_plots,no_of_comp), (1,1))    
colorscatter(halos_DTFE,halos_DTFE[:,9])

#spin myden
ax=plt.subplot2grid((no_of_plots,no_of_comp), (2,0))
fil_filt=np.where(halos_myden[:,10]==2)
fil_halos_myden=halos_myden[fil_filt]
catalog_vec_norm_spin=skl.normalize(fil_halos_myden[:,6:9])
plt.quiver(fil_halos_myden[:,0],fil_halos_myden[:,2],catalog_vec_norm_spin[:,0],catalog_vec_norm_spin[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,500])
ax.set_ylim([0,500])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
#spin DTFE
ax=plt.subplot2grid((no_of_plots,no_of_comp), (2,1)) 
fil_filt=np.where(halos_DTFE[:,10]==2)
fil_halos_DTFE=halos_DTFE[fil_filt]
catalog_vec_norm_spin=skl.normalize(fil_halos_DTFE[:,6:9])
plt.quiver(fil_halos_DTFE[:,0],fil_halos_DTFE[:,2],catalog_vec_norm_spin[:,0],catalog_vec_norm_spin[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,500])
ax.set_ylim([0,500])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
#vel myden
ax=plt.subplot2grid((no_of_plots,no_of_comp), (3,0))
fil_filt=np.where(halos_myden[:,10]==2)
fil_halos_myden=halos_myden[fil_filt]
catalog_vec_norm_spin=skl.normalize(fil_halos_myden[:,3:6])
plt.quiver(fil_halos_myden[:,0],fil_halos_myden[:,2],catalog_vec_norm_spin[:,0],catalog_vec_norm_spin[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,500])
ax.set_ylim([0,500])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
#vel DTFE
ax=plt.subplot2grid((no_of_plots,no_of_comp), (3,1)) 
fil_filt=np.where(halos_DTFE[:,10]==2)
fil_halos_DTFE=halos_DTFE[fil_filt]
catalog_vec_norm_spin=skl.normalize(fil_halos_DTFE[:,3:6])
plt.quiver(fil_halos_DTFE[:,0],fil_halos_DTFE[:,2],catalog_vec_norm_spin[:,0],catalog_vec_norm_spin[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,500])
ax.set_ylim([0,500])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
#eigenvectors myden
ax=plt.subplot2grid((no_of_plots,no_of_comp), (4,0))
#Now I just need to filter out all vecs and make em 0 whre they aren't filamnets. Do the same with velocity vecs and spin.
fil_filt=np.where(mask_myden==2)
recon_vecs_x_myden[fil_filt]=0
recon_vecs_y_myden[fil_filt]=0
recon_vecs_z_myden[fil_filt]=0
plt.quiver(recon_vecs_x_myden[:,slc,:],recon_vecs_z_myden[:,slc,:],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
#eigenvectors DTFE
ax=plt.subplot2grid((no_of_plots,no_of_comp), (4,1))
#Now I just need to filter out all vecs and make em 0 whre they aren't filamnets. Do the same with velocity vecs and spin.
fil_filt=np.where(mask_myden==2)
recon_vecs_x_DTFE[fil_filt]=0
recon_vecs_y_DTFE[fil_filt]=0
recon_vecs_z_DTFE[fil_filt]=0
plt.quiver(recon_vecs_x_DTFE[:,slc,:],recon_vecs_z_DTFE[:,slc,:],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
#mask plotting myden
ax=plt.subplot2grid((no_of_plots,no_of_comp), (5,0))
classify_mask(mask_myden,grid_nodes,slc,smooth_scl,plane=y)
#mask plotting DTFE
ax=plt.subplot2grid((no_of_plots,no_of_comp), (5,1))
classify_mask(mask_DTFE,grid_nodes,slc,smooth_scl,plane=y)

plt.savefig('test.png')
'''
