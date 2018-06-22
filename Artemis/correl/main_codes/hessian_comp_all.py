import h5py
import numpy as np
import sys
from decimal import Decimal

job_no=int(sys.argv[1])      #will iterate through each job
sim_type=sys.argv[2]          #'dm_only' 'DTFE'
cosmology=sys.argv[3]          #DMONLY:'lcdm'  'cde0'  'wdm2'DMGAS: 'lcdm' 'cde000' 'cde050' 'cde099'
snapshot=int(sys.argv[4])      #'12  '11'...
den_type=sys.argv[5]           #'DTFE' 'my_den'
smooth_scl=Decimal(sys.argv[6])#Smoothing scale in physical units Mpc/h. 2  3.5  5
sim_sz=500                   #Size of simulation in physical units Mpc/h cubed
grid_nodes=1250              #field density

f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/smth_fld/%s_snapshot_0%s_myden_pascal_partcls_sim%sMpch_smth%sMpch_%sgrid.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'r')
smthd_image=f['/myden_smthd'][:]
f.close()

smthd_image=np.reshape(smthd_image,(grid_nodes,grid_nodes,grid_nodes))
fft_smthd_image=np.fft.fftn(smthd_image)

#Create k-space grid
k_grid = range(grid_nodes)/np.float64(grid_nodes)
for i in range(grid_nodes/2+1, grid_nodes):
    k_grid[i] = -np.float64(grid_nodes-i)/np.float64(grid_nodes)

rc=1.*sim_sz/(grid_nodes)#physical space interval
k_grid = k_grid*2*np.pi/rc# k=2pi/lambda as per the definition of a wavenumber. see wiki

k_z=np.reshape(k_grid,(1,grid_nodes))
k_y=np.reshape(k_z,(grid_nodes,1))
if job_no==1:
    
    a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_z[:]=k_z
    a_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_x=a_z.transpose()   
    a_xx=-1*np.multiply(a_x,a_x)
    del a_z
    del a_x
    dxx=fft_smthd_image*a_xx
    del a_xx
    dxx=np.fft.ifftn(dxx).real
    dxx=dxx.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dxx.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dxx',data=dxx)
    f.close()

if job_no==2:
    
    a_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_y[:]=k_y
    a_yy=-1*np.multiply(a_y,a_y)
    del a_y
    dyy=fft_smthd_image*a_yy
    del a_yy
    dyy=np.fft.ifftn(dyy).real
    dyy=dyy.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dyy.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dyy',data=dyy)
    f.close()

if job_no==3:
    
    
    a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_z[:]=k_z
    a_zz=-1*np.multiply(a_z,a_z)
    del a_z
    dzz=fft_smthd_image*a_zz
    del a_zz
    dzz=np.fft.ifftn(dzz).real
    dzz=dzz.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dzz.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dzz',data=dzz)
    f.close()

if job_no==4:
    
    a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_z[:]=k_z
    a_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_y[:]=k_y
    a_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_x=a_z.transpose()
    a_xy=-1*np.multiply(a_x,a_y)
    del a_z
    del a_y
    del a_x
    dxy=fft_smthd_image*a_xy
    del a_xy
    dxy=np.fft.ifftn(dxy).real
    dxy=dxy.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dxy.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dxy',data=dxy)
    f.close()
    
if job_no==5:
        
    a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_z[:]=k_z
    a_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_x=a_z.transpose()
    a_xz=-1*np.multiply(a_x,a_z)
    del a_z
    del a_x
    dxz=fft_smthd_image*a_xz
    del a_xz
    dxz=np.fft.ifftn(dxz).real
    dxz=dxz.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dxz.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dxz',data=dxz)
    f.close()

if job_no==6:
    
    a_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_z[:]=k_z
    a_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
    a_y[:]=k_y
    a_yz=-1*np.multiply(a_y,a_z)
    del a_z
    del a_y
    dyz=fft_smthd_image*a_yz
    del a_yz
    dyz=np.fft.ifftn(dyz).real
    dyz=dyz.flatten()
    
    f=h5py.File("/scratch/GAMNSCM2/%s/%s/snapshot_0%s/correl/%s/files/hessian_comp/%s_snapshot_0%s_sim%s_smth%sMpc_gd%d_ifft_dyz.h5" %(sim_type,cosmology,snapshot,den_type,cosmology,snapshot,sim_sz,smooth_scl,grid_nodes), 'w')
    f.create_dataset('/hessian_ingr/ifft_dyz',data=dyz)
    f.close()


    
