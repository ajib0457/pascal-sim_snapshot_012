import numpy as np
import pandas as pd
import h5py
'''
'ID(1) ID_mbp(2) hostHaloID(3) numSubStruct(4) npart(5) Structuretype(6) Mvir(7) Xc(8) Yc(9) Zc(10) Xcmbp(11) Ycmbp(12) Zcmbp(13) 
VXc(14) VYc(15) VZc(16) VXcmbp(17) VYcmbp(18) VZcmbp(19) Mass_tot(20) Mass_FOF(21) Mass_200mean(22) Mass_200crit(23) Mass_BN97(24) 
Efrac(25) Rvir(26) R_size(27) R_200mean(28) R_200crit(29) R_BN97(30) R_HalfMass(31) Rmax(32) Vmax(33) sigV(34) veldisp_xx(35) veldisp_xy(36) 
veldisp_xz(37) veldisp_yx(38) veldisp_yy(39) veldisp_yz(40) veldisp_zx(41) veldisp_zy(42) veldisp_zz(43) lambda_B(44) Lx(45) Ly(46) Lz(47) 
q(48) s(49) eig_xx(50) eig_xy(51) eig_xz(52) eig_yx(53) eig_yy(54) eig_yz(55) eig_zx(56) eig_zy(57) eig_zz(58) cNFW(59) Krot(60) Ekin(61) 
Epot(62) RVmax_sigV(63) RVmax_veldisp_xx(64) RVmax_veldisp_xy(65) RVmax_veldisp_xz(66) RVmax_veldisp_yx(67) RVmax_veldisp_yy(68) RVmax_veldisp_yz(69) 
RVmax_veldisp_zx(70) RVmax_veldisp_zy(71) RVmax_veldisp_zz(72) RVmax_lambda_B(73) RVmax_Lx(74) RVmax_Ly(75) RVmax_Lz(76) RVmax_q(77) RVmax_s(78) 
RVmax_eig_xx(79) RVmax_eig_xy(80) RVmax_eig_xz(81) RVmax_eig_yx(82) RVmax_eig_yy(83) RVmax_eig_yz(84) RVmax_eig_zx(85) RVmax_eig_zy(86) RVmax_eig_zz(87) '
'''
cosmology='lcdm'           #'lcdm'  'cde0'  'wdm2'
snapshot=11                #'12  '11'
data = pd.read_csv('/import/oth3/ajib0457/VELOCIraptor/files/input_files/catalogs/%s/%s_snapshot_0%s.properties'%(cosmology,cosmology,snapshot), header = None)
print data
no_halos=int(len(data)-3)
data=data[0][3:no_halos+3]
data=np.asarray(data)

halos=[]
for i in range(no_halos):
    
    halos.append(data[i].split(' '))
    
halos=np.asarray(halos)
#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (Vir. Mass)Mvir(10^10Msun/h) & (Vir. Rad)Rvir(kpc/h) 
#halos_fnl=np.vstack((halos[:,7],halos[:,8],halos[:,9],halos[:,13],halos[:,14],halos[:,15],halos[:,44],halos[:,45],halos[:,46],halos[:,6],halos[:,25]))

#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (Vir. Mass)Mvir(10^10Msun/h), (Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
#halos_fnl=np.vstack((halos[:,7],halos[:,8],halos[:,9],halos[:,13],halos[:,14],halos[:,15],halos[:,44],halos[:,45],halos[:,46],halos[:,6],halos[:,25],halos[:,4]))

#halos array: (Pos)XYZ(kpc/h), (Vel)VxVyVz(km/s), (Ang. Mom)JxJyJz((Msun/h)*(kpc/h)*km/s), (tot. Mass)Mtot(10^10Msun/h),(Vir. Rad)Rvir(kpc/h) & npart (no. particles for each sructure)
halos_fnl=np.vstack((halos[:,7],halos[:,8],halos[:,9],halos[:,13],halos[:,14],halos[:,15],halos[:,44],halos[:,45],halos[:,46],halos[:,19],halos[:,25],halos[:,4]))

halos_fnl=np.float64(halos_fnl)
halos_fnl=halos_fnl.transpose()


#change to proper coordinates
hub_val=0.67 #check for each cosmology just in case. CDE seems to have h=0.67 also
halos_fnl[:,0]=halos_fnl[:,0]*hub_val 
halos_fnl[:,1]=halos_fnl[:,1]*hub_val
halos_fnl[:,2]=halos_fnl[:,2]*hub_val

#Correct for periodicity
for i in range(len(halos_fnl)):
    if (halos_fnl[i,0]>500000 or halos_fnl[i,0]<0):
        halos_fnl[i,0]=abs(abs(halos_fnl[i,0])-500000)
        
    if (halos_fnl[i,1]>500000 or halos_fnl[i,1]<0):
        halos_fnl[i,1]=abs(abs(halos_fnl[i,1])-500000)
        
    if (halos_fnl[i,2]>500000 or halos_fnl[i,2]<0):
        halos_fnl[i,2]=abs(abs(halos_fnl[i,2])-500000)

#Mass is currently (Vir. Mass)Mvir(10^10Msun/h), change to Msun/h
halos_fnl[:,9]=halos_fnl[:,9]*10**10

#Position is currently kpc/h, change to Mpc/h
halos_fnl[:,0]=halos_fnl[:,0]/1000.0
halos_fnl[:,1]=halos_fnl[:,1]/1000.0
halos_fnl[:,2]=halos_fnl[:,2]/1000.0
print 'check if this:',len(halos_fnl), 'matches no. of halos within header'

f=h5py.File("/import/oth3/ajib0457/VELOCIraptor/files/output_files/catalogs/%s/%s_snapshot_0%s_pascal_VELOCIraptor_allhalos_xyz_vxyz_jxyz_mtot_r_npart.h5"%(cosmology,cosmology,snapshot) , 'w')
f.create_dataset('/halo',data=halos_fnl)
f.close()
