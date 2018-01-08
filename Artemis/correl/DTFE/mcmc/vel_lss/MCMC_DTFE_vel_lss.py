import numpy as np
import scipy.optimize as op
import pickle
import math as mth
import emcee
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)#does not truncate arrays in console
import sklearn.preprocessing as skl
#import corner
from emcee.utils import MPIPool
import sys

#number of dot product bins
n_dot_prod=0
#FILTER HALO VIA MASS: [12.39 - 13.00](273,923halos) [13.00 - 13.61](75,398) [13.61 - 14.22](18,452) [14.22 - 14.82](3,462) [14.82 - 15.43](264)
#----
mass_bin=3 # 0 to 4
mass_intvl=0.60941991122086459
low_int_mass=12.387083597569344+mass_intvl*mass_bin
hi_int_mass=low_int_mass+mass_intvl
#----
grid_nodes=2000
sim_sz=500000#kpc
in_val,fnl_val=-140,140
tot_parts=8
s=1.96

#Calculate the std deviation in physical units
grid_phys=sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys


#likelihood function
#c=np.linspace(-0.99,0.99,3000)#-1.5 to0.99 for the log-likelihood since (1-c). -1 to 1 for likelihood
#dotprodval=np.round(np.linspace(0,0.99,n_dot_prod/2),3)#value for each index in costheta array
#costheta=0.02*np.around(50*np.random.uniform(-1,1,4950))
inputfile=open("/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/dotproduct/vel_lss/DTFE_grid%d_spin_store_%dbins_fil_Log%s-%s_smth%skpc_binr.pkl"%(grid_nodes,n_dot_prod,round(low_int_mass,2),round(hi_int_mass,2),std_dev_phys),'rb')
costheta=pickle.load(inputfile)
#costheta=costheta[:n_dot_prod/2]+costheta[:n_dot_prod/2:n_dot_prod]#Fold the dot product distribution in half so that it ranges 0 to 1 instead of -1 to 1
costheta=abs(costheta)#To change the dot products to be between 0-1 since we only care about alignment, not specific direction

#MCMC
  
def lnlike(c,costheta):
    loglike=np.zeros((1))
    #for j in range(len(costheta)):
    #likelihood[0]=likelihood[0]*((1-c)*np.sqrt(1+(c/2))*(1-c*(1-3*(dotprodval[j]*dotprodval[j]/2)))**(-1.5))**costheta[j]#likelihood
    loglike[0]=sum(np.log((1-c)*np.sqrt(1+(c/2))*(1-c*(1-3*(costheta*costheta/2)))**(-1.5)))#log-likelihood 

    
    return loglike
    
def lnprior(c):
    
    if (-1.5 < c < 0.99):#Assumes a flat prior, uninformative prior
        return 0.0
    return -np.inf
    
def lnprob(c,costheta):
    lp = lnprior(c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(c,costheta)
#Parallel MCMC - initiallizes pool object; if process isn't running as master, wait for instr. and exit
pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

#Initial conditions
ndim, nwalkers = 1, 500
initial_c=0.4

pos = [initial_c+1e-2*np.random.randn(ndim) for i in range(nwalkers)]#initial positions for walkers "Gaussian ball"
 
#MCMC Running
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=[costheta],pool=pool)
#Burn-in
#print("Running burn-in...") #get rid of burn-in bit if you want to see walekrs chain from start to finish
burn_in=500
pos, _, _=sampler.run_mcmc(pos,burn_in)#running of emcee burn-in period
sampler.reset()
#MCMC Running
#print("Running production...")
steps_wlk=3000
sampler.run_mcmc(pos, steps_wlk)#running of emcee for steps specified, using pos as initial walker positions

pool.close()
#Plotting
c_samples=sampler.flatchain[:,0]
output=open("/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/mcmc/vel_lss/flt_chain_%sburnin_%ssteps_%snwalkrs_grid%d_fil_Log%s-%s_smth%skpc_vel_lss.pkl"%(burn_in,steps_wlk,nwalkers,grid_nodes,round(low_int_mass,2),round(hi_int_mass,2),std_dev_phys), 'wb')
pickle.dump(c_samples,output)
output.close() 
#inputfile=open("c_samples.pkl",'rb')
#c_samples=pickle.load(inputfile)
#X,Y,_=plt.hist(c_samples,bins=1000,normed=True)
#z=np.linspace(np.max(c_samples),np.min(c_samples),1000)
#plt.plot(z,x)#this is for the pdf
#print("--- %s seconds ---" %(time.time()-start_time))
'''
plt.xlabel("c values")
plt.title("posterior distribution")
#Gaussian plotting
mu=0.0007
sigma=0.0013
x=np.linspace(np.min(c_samples),np.max(c_samples),1000) 
#normstdis=np.zeros((1000,1))
normstdis=1/(np.sqrt(2*(sigma**2)*mth.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))
plt.plot(x,normstdis,label='normal distribution fitted')
'''
'''
for i in range(n_dist_bins):
    
    plt.suptitle('Chain for each walker')
    p=plt.subplot(5,2,i+1)
    
    
    plt.title('Walker %i'%(i+1),fontsize=10)
    plt.rc('font', **{'size':'10'})
    plt.plot(sampler.chain[i,:,:])
'''
    
'''
# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.scatter(x,y)
plt.plot(x,m_true*x+b_true)
'''
