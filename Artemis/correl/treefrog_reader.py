import numpy as np
import pandas as pd
import h5py
import sys
import pickle
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), 
#(tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure),ID(1) ,
#ID_mbp(2), hostHaloID(3) and numSubStruct(4)
sim_type=sys.argv[1]   #'dm_only' 'DTFE'
cosmology=sys.argv[2]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snap=['11','10','09','08','07','06','05']
cat_lengths=np.zeros(len(snap))
i=0      
for snapshot in snap:    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
    cats=f['/halo'][:]
    f.close()
    cat_lengths[i]=len(cats)
    i+=1
del cats
#memorise first two snaps to get parent and progenitor id and nparts
snapshot='11'
f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
cat_11=f['/halo'][:]
f.close()
snapshot='10'
f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snapshot,sim_type,cosmology,snapshot), 'r')
cat_p=f['/halo'][:]
f.close()
#Read treefrog data 
data = pd.read_csv('/project/GAMNSCM2/main_codes/Treefrog/stf/treefrog_%s_%s_005_011_tree_revd0'%(sim_type,cosmology),header=None)

#skip header
no_halos=int(len(data)-2)
data=data[0][3:no_halos+3]
data=np.asarray(data)

#rid of space and \t within data
a=np.ones((len(data),2))*-1
for i in range(len(data)):
    
    val=np.fromstring(data[i],  sep="\t")
    a[i,0:len(val)]=val
    
diction={}#For storing all evoltn_tree
a_in=1#initialise jumping over subheader for each snapshot
for j in range(int(cat_lengths[0])):#
    if j>0:#This statement jumps a through each parent halo in snap 11
        a_in=a_in+int(a[a_in,1])+1
    evoltn_tree=[]#[id,npart]...
    fnd_hlo=np.where(cat_11[:,12]==a[a_in,0])#Find the npart of the halo to be traced
    parent_halo_npart=cat_11[fnd_hlo,11]
    parent=np.vstack((a[a_in,0],parent_halo_npart)).flatten()

    fnd_hlo=np.where(cat_p[:,12]==a[a_in+1,0])#find its first progenitor.
    progenitor_npart=cat_p[fnd_hlo,11]
    progenitor=np.vstack((a[a_in+1,0],progenitor_npart)).flatten()
    
    evoltn_tree.append(parent)#Store parent and first progenitor info
    evoltn_tree.append(progenitor)
    
    i=1
    while progenitor_npart>0.5*parent_halo_npart:#while progentior greater than half parent
        a_min=int(np.array(np.where(a[:,1]==cat_lengths[i])))
        a_max=int(np.array(np.where(a[:,1]==cat_lengths[i+1])))
        a_cutout=a[a_min+1:a_max]                
        b_in=np.asarray(np.where(a_cutout[:,0]==progenitor[0])).flatten()#find next progentior in next branch, where col 1 is >=1
        
        #Create if statement which deciphers which b_in index belongs to a parent halos
        if len(b_in)>1:
            b_indx=np.where(a_cutout[b_in,1]>=1)
            b_in=b_in[b_indx]
            
        f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/catalogs/%s_%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart_id_idmbp_hosthaloid_numsubstruct.h5"%(sim_type,cosmology,snap[i+1],sim_type,cosmology,snap[i+1]), 'r')
        cat_px=f['/halo'][:]
        f.close()
        
        fnd_hlo=np.where(cat_px[:,12]==a_cutout[b_in+1,0])#find its first progenitor.
        progenitor_npart=cat_px[fnd_hlo,11]
        progenitor=np.vstack((a_cutout[b_in+1,0],progenitor_npart)).flatten()
        evoltn_tree.append(progenitor)
        i+=1
    diction[j]=evoltn_tree

filehandler = open('/scratch/GAMNSCM2/ft/%s/%s/ft_dict_%s_%s.pkl'%(sim_type,cosmology,sim_type,cosmology),"wb")
pickle.dump(diction,filehandler)
filehandler.close() 
