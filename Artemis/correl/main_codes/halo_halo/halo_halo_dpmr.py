from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle
snap='09'
cosm='lcdm'
filehandler = open('/import/oth3/ajib0457/pascal_sims/halo_halo_plts/dm_only/%s/resultsdp_dm_only_%s_%s_100.pkl'%(cosm,cosm,snap),"rb")       
results_dp=pickle.load(filehandler)
filehandler.close()

filehandler = open('/import/oth3/ajib0457/pascal_sims/halo_halo_plts/dm_only/%s/resultsmr_dm_only_%s_%s_100.pkl'%(cosm,cosm,snap),"rb")       
results_mr=pickle.load(filehandler)
filehandler.close()

'''
#Worth doing this for all cosmologies
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(results_mr[:,2],results_dp[:,2],results_mr[:,5],label='lcdm')
ax.set_zlabel('log r[Mpc/h]')
ax.set_xlabel('$M_{1}/M_{2}$')
ax.set_ylabel('<J(x).J(x+r)>')
plt.legend()
'''

'''
plt.figure()#plot original dp vs dist
ax2=plt.subplot2grid((1,1), (0,0))
#ax2.axhline(y=0.5, xmin=0, xmax=15, color = 'k',linestyle='--')
plt.ylabel('$M_{1}/M_{2}$')
plt.xlabel('log r[Mpc/h]')   
ax2.plot(results_mr[:,0],results_mr[:,2],'g-',label='mass')
ax2.fill_between(results_mr[:,0], results_mr[:,2]-abs(results_mr[:,4]), results_mr[:,2]+abs(results_mr[:,3]),facecolor='green',alpha=0.3)

#ax2.plot(results_dp[:,0],results_dp[:,2],'r-',label='spin')
#ax2.fill_between(results_dp[:,0], results_dp[:,2]-abs(results_dp[:,4]), results_dp[:,2]+abs(results_dp[:,3]),facecolor='red',alpha=0.3)

plt.legend(loc='upper right')
'''

'''
plt.figure()#plot mr vs dp
ax2=plt.subplot2grid((1,1), (0,0))
plt.ylabel('<J(x).J(x+r)>')
plt.xlabel('$M_{1}/M_{2}$')
plt.axhline(y=0.5, xmin=0, xmax=100, color = 'k',linestyle='--')
#plt.axvline(x=1.0, ymin=0, ymax=15, color = 'k',linestyle='--')
#The error goes as [y],[x]. The error goes as [-,+]
plt.errorbar(results_mr[:,2],results_dp[:,2],[abs(results_dp[:,4]),abs(results_dp[:,3])],[abs(results_mr[:,4]),abs(results_mr[:,3])],fmt='o')
'''




