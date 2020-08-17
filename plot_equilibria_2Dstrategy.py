# -*- coding: utf-8 -*-

"""
PLOT SYSTEM'S EQUILIBRIA IN THE LAND USE STRATEGY SPACE.
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib.style as style
import seaborn as sns
from scipy import interpolate


def tradeoff(b):
    k0=0.5*np.ones(len(b))
    return k0+(np.ones(len(b))-b)*4.5


filename="DATA_CBORDER_R1_D_1_E_1_F_1.dat"
border=np.loadtxt(filename)

brev=border[:,1]
Krev=border[:,0]

b=np.flip(brev)
K=np.flip(Krev)

maxind=np.amax(np.where(K==5))
print(maxind)

f = interpolate.interp1d(b[maxind:], K[maxind:], kind='cubic')

db=0.01
bnew = np.arange(np.amin(b[maxind:]), np.amax(b[maxind:])-db, db)

# print(b)
# print(K)
# print(bnew)

Knew = f(bnew)

sns.set_context('paper')
style.use('seaborn-paper')

fig, (ax1) = plt.subplots(1, 1)

bnew=np.concatenate((b[:maxind+1],bnew))
Knew=np.concatenate((K[:maxind+1],Knew))

Ktrade = tradeoff(bnew)

issmaller=Knew<Ktrade
collrange=bnew[np.where(issmaller==True)]
Krange=Knew[np.where(issmaller==True)]
#, facecolor='tab:blue' , facecolor='tab:orange'
ax1.fill_between(bnew, 0.5, Knew,alpha=0.9)
ax1.fill_between(bnew, 5 , Knew,alpha=0.9)

#,color='black'
ax1.plot(bnew[maxind:],Knew[maxind:],linewidth=0.5)
ax1.annotate('Collapse equilibrium', xy=(0.6, 3.5))
ax1.annotate('Viable equilibrium', xy=(0.275, 1.5))

# point1 = [bnew[0], 0.5+4.5*(1-bnew[0])]
# point2 = [bnew[-1], 0.5+4.5*(1-bnew[-1])]
# x_values = [point1[0], point2[0]]
# y_values = [point1[1], point2[1]]
ax1.plot(bnew, Ktrade,linewidth=1.0,linestyle='solid',color='0.1')
ax1.annotate(s=r'$K=K_0+\Delta K(1-\beta)$', xy=(0.72,2.1), color='0.1')

# ax1.scatter([collrange[0],collrange[-1]],[Krange[0],Krange[-1]],color='k')

ax1.vlines(collrange[0],Krange[0],0.5,color='0.2',linewidth=0.7,linestyle='--')
ax1.vlines(collrange[-1],Krange[-1],0.5,color='0.2',linewidth=0.7,linestyle='--')
ax1.arrow(collrange[0],0.75,collrange[-1]-collrange[0],0.0,color='0.1',linewidth=0.3,head_width=0.08, head_length=0.01, length_includes_head=True)
ax1.arrow(collrange[-1],0.75,collrange[0]-collrange[-1],0.0,color='0.1',linewidth=0.3,head_width=0.08, head_length=0.01, length_includes_head=True)
ax1.annotate(s=r'$\Delta \beta$', xy=(0.55,0.85))

ax1.set_ylabel(r'$K$')
ax1.set_xlabel(r'$\beta$')
sns.despine()

plt.show()
