import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
from signal_funcs import *
#Firstly import all modules needed. Should I make resid_fnl.py a function? I think that would be best.
#then import all files needed which are: grid_mthd.py output as well as dotprod.py output.
#Then calculate halo resids
#Then plot signals as I do in per_snapshot plots (from plot_fig_5.py)
#then plot 

#Random data
a_x=np.random.rand(20)*10-10
a_x=np.sort(a_x)
a_y=np.random.rand(len(a_x))+np.random.randint(5)
a_err_pos=np.random.rand(len(a_x))+a_y
a_err_neg=a_y-np.random.rand(len(a_x))

b_x=np.random.rand(20)*10-10
b_x=np.sort(b_x)
b_y=np.random.rand(len(b_x)) 
b_err_pos=np.random.rand(len(b_x))+b_y
b_err_neg=b_y-np.random.rand(len(b_x))

'''
#non-random data
a_x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b_x=a_x+0.001

wdm=np.array([10,2,-7,-7,3,10,-10,-2,7,7,-2,-10])*2
wdm_err_pos=np.array([3.01,4.04,6.05,1.02,6.09,5.08,3.06,4,6,1,6,5])+wdm
wdm_err_mns=wdm-np.array([2.08,8.06,6.02,1.04,9.03,5.03,2.05,8.02,6.02,1.05,9.001,5.03])

lcdm=np.array([-10,-2,7,7,-2,-10,10,2,-7,-7,2,10])
lcdm_err_pos=np.array([3,4,6,1,6,5,2,8,6,1,9,5])+lcdm
lcdm_err_mns=lcdm-np.array([2,8,6,1,9,5,2,8,6,1,9,5])

#variables conversion
a_y=wdm
b_y=lcdm
a_err_pos=wdm_err_pos
a_err_neg=wdm_err_mns
b_err_pos=lcdm_err_pos
b_err_neg=lcdm_err_mns
'''
'''
#non-random data
a_x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b_x=a_x-5

wdm=np.array([10,10,10,10,10,10,10,10,10,10,10,10])
wdm_err_pos=np.array([3,3,3,3,3,3,3,3,3,3,3,3])+wdm
wdm_err_mns=wdm-np.array([3,3,3,3,3,3,3,3,3,3,3,3])

lcdm=np.array([20,20,20,20,20,20,20,20,20,20,20,20])
lcdm_err_pos=np.array([3,3,3,3,3,3,3,3,3,3,3,3])+lcdm
lcdm_err_mns=lcdm-np.array([3,3,3,3,3,3,3,3,3,3,3,3])

#variables conversion
a_y=wdm
b_y=lcdm
a_err_pos=wdm_err_pos
a_err_neg=wdm_err_mns
b_err_pos=lcdm_err_pos
b_err_neg=lcdm_err_mns
'''    
res,err_pos,err_neg=calc_resid(a_x,a_y,a_err_pos,a_err_neg,b_x,b_y,b_err_pos,b_err_neg)    
#plotting
plt.figure(figsize=(10,7))
#original signals  
ax1=plt.subplot2grid((2,1), (0,0))
ax1.axhline(y=0, xmin=0, xmax=15, color = 'k',linestyle='--')
ax1.plot(a_x,a_y,'r-',label='a')
ax1.fill_between(a_x, a_err_neg, a_err_pos,facecolor='red',alpha=0.3)
ax1.plot(b_x,b_y,'b-',label='b')
ax1.fill_between(b_x, b_err_neg, b_err_pos,facecolor='blue',alpha=0.3)
plt.title('signals')
ax1.legend(loc='best')
#plt.xlim(np.min(res_lcdm_wdm_np[:,0])-1,np.max(res_lcdm_wdm_np[:,0])+1)#an important question is how to 
plt.grid()
#residual
ax2=plt.subplot2grid((2,1), (1,0))
ax2.axhline(y=0, xmin=0, xmax=15, color = 'k',linestyle='--')
ax2.plot(res[:,0],res[:,1],'g-',label='wdm resid')
ax2.fill_between(res[:,0], err_neg, err_pos,facecolor='green',alpha=0.3)
plt.title('residuals')
#plt.xlim(np.min(res_lcdm_wdm_np[:,0])-1,np.max(res_lcdm_wdm_np[:,0])+1)
plt.grid()

