# -*- coding: utf-8 -*-

"""
TEMPORAL DYANMICS PLOTS IN THE K,I PLANE
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import time
import pylab as pl
from matplotlib import cm
from matplotlib import colors
import matplotlib.colors
import seaborn as sns
import matplotlib.style as style

t0 = time.time()

# """
# FUNCTION DECLARATION
# """
# def foodProduction(state,b):
#     l0=state[0];l1=state[1];
#
#     effective_land=(l0+l1*(1-b))
#
#     a=1-l0-l1
#     prod=b*np.power(a,b)*np.power(effective_land,1-b)
#
#     return prod
#
# beta=np.linspace(0,1,100)
#
# state1=[0.5,0.25]
# state2=[0.25,0.25]
# state3=[0.1,0.5]
#
# prod1=foodProduction(state1,beta)
# prod2=foodProduction(state2,beta)
# prod3=foodProduction(state3,beta)
#
# sns.set_style("ticks")
#
# fig, ax =plt.subplots()
#
# ax.plot(beta,prod1,linewidth=2,label=r'$N=0.1,\, A=0.9$',color="tab:olive",alpha=0.8)
# ax.plot(beta,prod2,linewidth=2,label=r'$N=0.5,\, A=0.5$',color="tab:orange",alpha=0.8)
# ax.plot(beta,prod3,linewidth=2,label=r'$N=0.1,\, A=0.5$',color="tab:purple",alpha=0.8)
#
# ax.set(xlabel=r"$\beta$");ax.set(ylabel="Production")
#
# fig.legend(loc='upper right', bbox_to_anchor=(0.6, 0.9))
# sns.despine()
# plt.show()
# plt.close()

def production(A,N,b,F):
    X, Y=np.meshgrid(A, N)
    Z = b*b*X + b*F*np.sqrt(X)*(1-b)*(Y+(1-b)*(1-Y-X))
    return Z

A=np.linspace(0,1,400)
N=np.linspace(0,1,400)
b1=1e-4
b2=0.2
b3=0.4
b4=0.6
b5=0.8
b6=1

prod1=production(A,N,b1,1);prod2=production(A,N,b2,1);prod3=production(A,N,b3,1)
prod4=production(A,N,b4,1);prod5=production(A,N,b5,1);prod6=production(A,N,b6,1)


sns.set_context('paper')
style.use('seaborn-paper')

# print(prod1.shape)

#print(prod1)

for i in range(prod1.shape[0]):
    prod1[i,prod1.shape[0]-i:]=np.nan
    prod2[i,prod1.shape[0]-i:]=np.nan
    prod3[i,prod1.shape[0]-i:]=np.nan
    prod4[i,prod1.shape[0]-i:]=np.nan
    prod5[i,prod1.shape[0]-i:]=np.nan
    prod6[i,prod1.shape[0]-i:]=np.nan

prod1=prod1/np.nanmax(prod1);prod2=prod2/np.nanmax(prod2);prod3=prod3/np.nanmax(prod3)
prod4=prod4/np.nanmax(prod4);prod5=prod5/np.nanmax(prod5);prod6=prod6/np.nanmax(prod6)
print(np.nanmax(prod6))

X,Y=np.meshgrid(A,N)

fig ,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,sharex='col',sharey='row',gridspec_kw={'hspace':0.2,'wspace': 0.2},figsize=(7,5.25))

ax1.set_facecolor('0.2')
ax2.set_facecolor('0.2')
ax3.set_facecolor('0.2')
ax4.set_facecolor('0.2')
ax5.set_facecolor('0.2')
ax6.set_facecolor('0.2')

# cp1=ax1.contourf(X,Y,prod1,alpha=0.9)
# cp2=ax2.contourf(X,Y,prod2,alpha=0.9)
# cp3=ax3.contourf(X,Y,prod3,alpha=0.9)
# cp4=ax4.contourf(X,Y,prod4,alpha=0.9)
# cp5=ax5.contourf(X,Y,prod5,alpha=0.9)
# cp6=ax6.contourf(X,Y,prod6,alpha=0.9)


ax1.set_ylabel('N')
ax4.set_ylabel('N')
ax4.set_xlabel('A')
ax5.set_xlabel('A')
ax6.set_xlabel('A')


plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)

pcm1 = ax1.pcolormesh(X, Y, prod1,
                   cmap='viridis',alpha=0.8)
pcm2 = ax2.pcolormesh(X, Y, prod2,
                   norm = colors.Normalize(vmin=0.,vmax=1.),
                   cmap='viridis',alpha=0.8)
pcm3 = ax3.pcolormesh(X, Y, prod3,
                   norm = colors.Normalize(vmin=0.,vmax=1.),
                   cmap='viridis',alpha=0.8)
pcm4 = ax4.pcolormesh(X, Y, prod4,
                   norm = colors.Normalize(vmin=0.,vmax=1.),
                   cmap='viridis',alpha=0.8)
pcm5 = ax5.pcolormesh(X, Y, prod5,
                   norm = colors.Normalize(vmin=0.,vmax=1.),
                   cmap='viridis',alpha=0.8)
pcm = ax6.pcolormesh(X, Y, prod6,
                   norm = colors.Normalize(vmin=0.,vmax=1.),
                   cmap='viridis',alpha=0.8)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
fig.colorbar(pcm,cax=cbar_ax)

# fig.colorbar(cp1,ax=ax1,ticks=cp1.levels)
# fig.colorbar(cp2,ax=ax2,ticks=cp2.levels)
# fig.colorbar(cp3,ax=ax3,ticks=cp3.levels)
# fig.colorbar(cp4,ax=ax4,ticks=cp4.levels)
# fig.colorbar(cp5,ax=ax5,ticks=cp5.levels)
# fig.colorbar(pcm,ax=ax6,ticks=cp6.levels)



ax1.set_title(r'$\beta=10^{-4}$')
ax2.set_title(r'$\beta=0.2$')
ax3.set_title(r'$\beta=0.4$')
ax4.set_title(r'$\beta=0.6$')
ax5.set_title(r'$\beta=0.8$')
ax6.set_title(r'$\beta=1$')

# fig.suptitle('Agricultural Production')
plt.savefig("agricultural_production_landuse.jpg",dpi=1000)
plt.show()
