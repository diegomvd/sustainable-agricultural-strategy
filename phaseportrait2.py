# -*- coding: utf-8 -*-

"""
TEMPORAL DYANMICS PLOTS IN THE K,I PLANE
"""

import numpy as np
from scipy import optimize # LIBRARY NEEDED QOR ROOT FINDING
from scipy import linalg # LIBRARY NEEDED FOR EIGENVALUES
import os
from matplotlib import pyplot as plt
import time
import pylab as pl
from matplotlib import cm
import matplotlib.colors
from scipy.ndimage import gaussian_filter
import matplotlib.style as style
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


#plt.style.use('seaborn')

t0 = time.time()

"""
FUNCTION DECLARATION
"""

def foodProduction(p,n,a,param):
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n

    effective_land=(n+d*(1-b))

    prod=b*(b*a + Q*(1-b)*effective_land*np.sqrt(a))

    return prod

###############################################################################
# dynamical equations

def pEq(p,n,a,param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    prod=foodProduction(p,n,a,param)

    return p*(1-p/prod)

def nEq(p,n,a,param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n
    ret = (R*np.power(n,0.5)-b*D*np.power(d,0.5))*np.power(d*n,0.5)-(Kmin+(K-Kmin)*(1-b))*p*n
    return ret

def aEq(p,n,a,param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    ret=(Kmin+(K-Kmin)*(1-b))*p*n-b*E*a

    return ret

###################################################################################
##

###############################################################################
# runge kutta solver

def solver_RK4(state,param,dt):

    p=state[0]; n=state[1]; a=state[2]

    k1p=dt*pEq(p,n,a,param);k1n=dt*nEq(p,n,a,param);k1a=dt*aEq(p,n,a,param);
    k1vec=np.array([k1p,k1n,k1a])
    state1=state+0.5*k1vec

    p1=state1[0]; n1=state1[1]; a1=state1[2]
    k2p=dt*pEq(p1,n1,a1,param);k2n=dt*nEq(p1,n1,a1,param);k2a=dt*aEq(p1,n1,a1,param);
    k2vec=np.array([k2p,k2n,k2a])
    state2=state+0.5*k2vec

    p2=state2[0]; n2=state2[1]; a2=state2[2]
    k3p=dt*pEq(p2,n2,a2,param);k3n=dt*nEq(p2,n2,a2,param);k3a=dt*aEq(p2,n2,a2,param);
    k3vec=np.array([k3p,k3n,k3a])
    state3=state+k3vec

    p3=state3[0]; n3=state3[1]; a3=state3[2]
    k4p=dt*pEq(p3,n3,a3,param);k4n=dt*nEq(p3,n3,a3,param);k4a=dt*aEq(p3,n3,a3,param);
    k4vec=np.array([k4p,k4n,k4a])

    var_state=np.array((k1vec+2*k2vec+2*k3vec+k4vec)/6)

    return state+var_state

###############################################################################
# functions for root finding

def initial_guess(param):

    T=300;dt=0.001;t=0
    state=np.array([0.001,0.98,0.01])
    state_new=np.zeros(3)

    while t<T:

        state_new=solver_RK4(state,param,dt)
        state=state_new

        t=t+dt

    return state

def jacobian(state,param):

    p=state[0];n=state[1];a=state[2]
    dstate=[1e-9,1e-9,1e-9]
    dp=dstate[0];dn=dstate[1];da=dstate[2]

    # states with the added perturbation
    pstate=[p+dp,n,a]
    nstate=[p,n+dn,a]
    astate=[p,n,a+da]

    pert_state=[pstate,nstate,astate]

    Jac=np.zeros((3,3))
    for i in range(3):
        Jac[0,i]=(pEq(pert_state[i][0],pert_state[i][1],pert_state[i][2],param)-pEq(p,n,a,param))/dstate[i]
        Jac[1,i]=(nEq(pert_state[i][0],pert_state[i][1],pert_state[i][2],param)-nEq(p,n,a,param))/dstate[i]
        Jac[2,i]=(aEq(pert_state[i][0],pert_state[i][1],pert_state[i][2],param)-aEq(p,n,a,param))/dstate[i]

    ret = Jac
    return ret

def analytical_jacobian(state,param):

    p=state[0];n=state[1];a=state[2]
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    Jac=np.zeros((3,3))

    d=1-a-n
    C=Kmin+(K-Kmin)*(1-b)
    Y=foodProduction(p,n,a,param)
    dYN=b*b*(1-b)*Q*np.sqrt(a)
    dYA=b*b + Q*b*(1-b)*0.5*( n/np.sqrt(a) + (1-b)*(1-n)/np.sqrt(a) - 3*(1-b)*np.sqrt(a) )
    # dYA=b*b + Q*b*(1-b)*( 0.5*(b*n+(1-a)*(1-b))*np.sqrt(a) - np.sqrt(a)*(1-b)*(1-b) )

    Jac[0,0]=1-2*p/Y
    Jac[0,1]=(p/Y)*(p/Y)*dYN
    Jac[0,2]=(p/Y)*(p/Y)*dYA

    Jac[1,0]=-C*n
    Jac[1,1]=-C*p + R*np.sqrt(d)*(1-0.5*n/d) + D*b*np.sqrt(n)*(1-0.5*d/n)
    Jac[1,2]=D*b*np.sqrt(n)*(1-0.5*(R/D/b)*np.sqrt(n/d))

    Jac[2,0]=C*n
    Jac[2,1]=C*p
    Jac[2,2]=-E*b

    ret=Jac
    return ret

def savearray(data,array):

    array=np.concatenate((array,data),axis=0)
    return array

def fun(state,param):
    p=state[0];n=state[1];a=state[2]
    return [pEq(p,n,a,param), nEq(p,n,a,param), aEq(p,n,a,param)]

def trajectories(istate,param,T,dt,dt_save):

    t=0;t_save=0;
    state=istate

    traj=np.array([[],[],[]]).T

    while(t<T):

        if t>=t_save:
            state_data=np.array([np.append(state[2],state[1])])
            state_data=np.array([np.append(state_data,state[0])])
            # state_data=np.array([state[2],state[1],state[0]])
            traj=savearray(state_data,traj)
            t_save+=dt_save

        state=solver_RK4(state,param,dt)

        t=t+dt

    return traj

###############################################################################
## encontrar el punto fijo en el regimen critico

R=1.0
D=1.0
E=1
F=1
b=0
K=4.5
Kmin=0.5

initial_state=np.array([0.01,0.98,0.01])
dstate=np.ones(5)*1e-7

datacritical1=np.array([[],[],[],[],[]]).T
"""
BIFURCATION ANALYSIS FOR CONTROL PARAMETER b
"""

bmin=0.2;bmax=0.8002;db=0.001
b=bmin
param=[b,K,R,D,E,F,Kmin]
old_fp=initial_guess(param)
old_stability=1
while b<bmax:

    param=[b,K,R,D,E,F,Kmin]

    state0=old_fp

    # solve the root problem and get the fixed point
    #sol = optimize.root(fun,state0,args=(param,),method='lm',jac=analytical_jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})
    sol = optimize.root(fun,state0,args=(param,),method='lm',jac=analytical_jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

    fixed_point = sol.x # steady states
    # print(fixed_point)

    # test if the root finder actually converged to a fp
    btest=np.array(fun(fixed_point,param))
    l2test=np.sqrt(np.dot(btest,btest))
    # print(l2test)
    if(l2test>1e-15):
        # print(l2test)
        print("b = "+str(b)+"; R="+str(R)+"; D="+str(D))
        # break
    old_fp=np.array(fixed_point)

    # get the eigenvalues of the jacobian matrix to calculate stability
    jac=analytical_jacobian(fixed_point,param)
    # la, v =linalg.eig(jac)
    la, v =np.linalg.eig(jac)
    stability=0
    # print(la)
    if all(la.real<0):
        stability=1

    # if stability==1:
    print(la)

    if stability!=old_stability: #changed branch
        print("stability changed")
        print(fixed_point)
        print("b = "+str(b))

    old_stability=stability

    # calculate a to put it in data file
    fixed_point_save=np.append(fixed_point,stability)
    data=np.array([np.append(b,fixed_point_save)])

    #print(data.shape)
    #print(datacritical1.shape)

    # save branch1
    datacritical1=savearray(data,datacritical1)

    b=b+db


###############################################################################

#saving bifurcation data

filename="BIFURCATION_K_"+str(K)+"_R_"+str(R)+"_D_"+str(D)+"_E_"+str(E)+"_F_"+str(F)+"_K0_"+str(Kmin)+".dat"
np.savetxt(filename,datacritical1)
print("saved")

# i have the steady states, now i am getting the indexes from the points i want to plot
ib1=np.where(datacritical1[:,0]>0.2)[0][0]
ib2=np.where(datacritical1[:,4]<1)[0][0]-1
# ib2=np.where(datacritical1[:,0]>0.43)[0][0]
ib3=np.where(datacritical1[:,4]<1)[0][0]
# ib3=np.where(datacritical1[:,0]>0.436)[0][0]

ib4=np.where(datacritical1[:,0]>0.8)[0][0]
ib5=np.where(datacritical1[:,4]<1)[0][-1]
# ib5=np.where(datacritical1[:,0]>0.75)[0][-1]
ib6=np.where(datacritical1[:,0]>0.6)[0][0]

# ib1=np.where(datacritical1[:,4]<1)[0][0]-1;ib2=ib1;ib3=ib1
# ib4=np.where(datacritical1[:,4]<1)[0][0];ib5=ib4;ib6=ib4

iblist=[ib1,ib2,ib3,ib4,ib5,ib6]

sns.set_context('paper')
style.use('seaborn-paper')

# nrow = 2;ncol=3;
# fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex='col',sharey='row',gridspec_kw={'hspace':0.2,'wspace': 0.2})
#
# sns.set_style("ticks")
#
# axs.ravel()[0].set_ylabel('N')
# axs.ravel()[3].set_ylabel('N')
# axs.ravel()[3].set_xlabel('A')
# axs.ravel()[4].set_xlabel('A')
# axs.ravel()[5].set_xlabel('A')
#
# T=200;dt=0.01;
# dt_save=0.01;
# for i in range(len(iblist)):
#     print(i)
#     ib=iblist[i]
#     b=datacritical1[ib,0]
#     p=datacritical1[ib,1]
#     n=datacritical1[ib,2]
#     a=datacritical1[ib,3]
#     s=datacritical1[ib,4]
#
#     Avec=np.linspace(0,1,200)
#     Nvec=np.linspace(0,1,200)
#     A, N = np.meshgrid(Avec, Nvec)
#     P=np.ones((A.shape[0],A.shape[1]))*p
#
#     # make a direction field plot with quiver
#     param=[b,K,R,D,E,F,Kmin]
#     dA = aEq(P,N,A,param)
#     dN = nEq(P,N,A,param)
#     dP = pEq(P,N,A,param)
#
#     #axs.ravel()[i].quiver(A[::20,::20], N[::20,::20], dA[::20,::20], dN[::20,::20],angles='xy',scale_units='xy',scale=2,color='tab:blue')
#     axs.ravel()[i].streamplot(A,N,dA, dN, density=3.0, linewidth=1.0, color=(0.1,0.5,1,0.4))
#     # plot nullclines
#     axs.ravel()[i].contour(A, N, dA, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(A, N, dN, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(A, N, dP, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     # axs.ravel()[i].contourf(A, N, np.sign(dP), alpha=0.9)
#     axs.ravel()[i].fill_between(Avec, np.ones(len(Avec))-Avec, np.ones(len(Avec)),color='0.2',alpha=1.0)
#     axs.ravel()[i].fill_between(Avec, (np.ones(len(Avec))-Avec)*b*b/(1+b*b), np.zeros(len(Avec)),color='tab:orange',alpha=0.7)
#
#     # plot steadystate
#     if s>0:
#         axs.ravel()[i].plot(a,n,'o',linewidth=3,color='red')
#     else:
#         axs.ravel()[i].plot(a,n,'o',linewidth=3,color='purple')
#
#     # trajectories
#     # istate=np.array([p,0.42136,0.3219075])
#     # trj=trajectories(istate,param,T,dt,dt_save)
#     # axs.ravel()[i].plot(trj[int(0/10*len(trj[:,0])):,0],trj[int(0/10*len(trj[:,1])):,1],color='k',linewidth='1')
#
#     # points = np.array([trj[:,0], trj[:,1]]).T.reshape(-1, 1, 2)
#     # segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     # lc = LineCollection(segments, cmap='viridis')
#     # lc.set_array(trj[:,2])
#     # lc.set_linewidth(1)
#     # line=axs.ravel()[i].add_collection(lc)
#     # fig.colorbar(line,ax=axs.ravel(i))
#     # axs.ravel()[i].plot(trj[:,0],trj[:,1],color=cm.inferno(trj[:,2]),linewidth='1')
#
#     # if i==0:
#     #     istate=np.array([p,0.05,0.94])
#     #     trj=trajectories(istate,param,T,dt,dt_save)
#     #     axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#     if i==1:
#         # T=100
#         dstate=1*np.array([1e-2,1e-2,1e-2])
#         istate=np.array([p,n,a])+dstate
#         trj=trajectories(istate,param,T,dt,dt_save)
#         axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#     # if i==2:
#     #     dstate=3.5*np.array([1e-2,1e-2,1e-2])
#     #     istate=np.array([p,n,a])+dstate
#     #     trj=trajectories(istate,param,T,dt,dt_save)
#     #     axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#     # if i==3:
#     #     T=50
#     #     istate=np.array([p,0.3,0.69])
#     #     trj=trajectories(istate,param,T,dt,dt_save)
#     #     axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#     # if i==4:
#     #     # T=50
#     #     dstate=np.array([1e-4,1e-4,1e-4])
#     #     istate=np.array([p,n,a])+dstate
#     #     trj=trajectories(istate,param,T,dt,dt_save)
#     #     axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#     # if i==5:
#     #     # T=200
#     #     dstate=3.5*np.array([1e-2,1e-2,1e-2])
#     #     istate=np.array([p,n,a])+dstate
#     #     trj=trajectories(istate,param,T,dt,dt_save)
#     #     axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#
#     # istate=np.array([p,0.5,0.49])
#     # trj=trajectories(istate,param,T,dt,dt_save)
#     # axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
#
#     axs.ravel()[i].set_title(r'$\beta=$'+str(np.around(b,3)))
#
# #fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
# sns.despine()
# plt.show()
# ############################################################################
# nrow = 2;ncol=3;
# fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex='col',sharey='row',gridspec_kw={'hspace':0.2,'wspace': 0.2})
#
# sns.set_style("ticks")
#
# axs.ravel()[0].set_ylabel('P')
# axs.ravel()[3].set_ylabel('P')
# axs.ravel()[3].set_xlabel('A')
# axs.ravel()[4].set_xlabel('A')
# axs.ravel()[5].set_xlabel('A')
#
# T=200;dt=0.01;
# dt_save=0.01;
# for i in range(len(iblist)):
#     print(i)
#     ib=iblist[i]
#     b=datacritical1[ib,0]
#     p=datacritical1[ib,1]
#     n=datacritical1[ib,2]
#     a=datacritical1[ib,3]
#     s=datacritical1[ib,4]
#
#     Avec=np.linspace(0,1,200)
#     Pvec=np.linspace(0,0.25,200)
#     A, P = np.meshgrid(Avec, Pvec)
#     N=np.ones((A.shape[0],A.shape[1]))*n
#
#     # make a direction field plot with quiver
#     param=[b,K,R,D,E,F,Kmin]
#     dA = aEq(P,N,A,param)
#     dN = nEq(P,N,A,param)
#     dP = pEq(P,N,A,param)
#
#     #axs.ravel()[i].quiver(A[::20,::20], N[::20,::20], dA[::20,::20], dN[::20,::20],angles='xy',scale_units='xy',scale=2,color='tab:blue')
#     axs.ravel()[i].streamplot(A,P,dA, dP, density=3.0, linewidth=1.0, color=(0.1,0.5,1,0.4))
#     # plot nullclines
#     axs.ravel()[i].contour(A, P, dA, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(A, P, dN, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(A, P, dP, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     # axs.ravel()[i].contourf(A, N, np.sign(dP), alpha=0.9)
#
#     # plot steadystate
#     if s>0:
#         axs.ravel()[i].plot(a,p,'o',linewidth=3,color='red')
#     else:
#         axs.ravel()[i].plot(a,p,'o',linewidth=3,color='purple')
#
# sns.despine()
# plt.show()

# ############################################################################
# nrow = 2;ncol=3;
# fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex='col',sharey='row',gridspec_kw={'hspace':0.2,'wspace': 0.2})
#
# sns.set_style("ticks")
#
# axs.ravel()[0].set_ylabel('P')
# axs.ravel()[3].set_ylabel('P')
# axs.ravel()[3].set_xlabel('N')
# axs.ravel()[4].set_xlabel('N')
# axs.ravel()[5].set_xlabel('N')
#
# T=200;dt=0.01;
# dt_save=0.01;
# for i in range(len(iblist)):
#     print(i)
#     ib=iblist[i]
#     b=datacritical1[ib,0]
#     p=datacritical1[ib,1]
#     n=datacritical1[ib,2]
#     a=datacritical1[ib,3]
#     s=datacritical1[ib,4]
#
#     Nvec=np.linspace(0,1,200)
#     Pvec=np.linspace(0,0.25,200)
#     N, P = np.meshgrid(Nvec, Pvec)
#     A=np.ones((N.shape[0],N.shape[1]))*a
#
#     # make a direction field plot with quiver
#     param=[b,K,R,D,E,F,Kmin]
#     dA = aEq(P,N,A,param)
#     dN = nEq(P,N,A,param)
#     dP = pEq(P,N,A,param)
#
#     #axs.ravel()[i].quiver(A[::20,::20], N[::20,::20], dA[::20,::20], dN[::20,::20],angles='xy',scale_units='xy',scale=2,color='tab:blue')
#     axs.ravel()[i].streamplot(N,P,dN, dP, density=3.0, linewidth=1.0, color=(0.1,0.5,1,0.4))
#     # plot nullclines
#     axs.ravel()[i].contour(N, P, dA, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(N, P, dN, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     axs.ravel()[i].contour(N, P, dP, levels=[0], linewidths=1.5, colors='0.2',linestyles='dashed')
#     # axs.ravel()[i].contourf(A, N, np.sign(dP), alpha=0.9)
#
#     # plot steadystate
#     if s>0:
#         axs.ravel()[i].plot(n,p,'o',linewidth=3,color='red')
#     else:
#         axs.ravel()[i].plot(n,p,'o',linewidth=3,color='purple')
#
#     axs.ravel()[i].set_title(r'$\beta=$'+str(np.around(b,3)))
#
# sns.despine()
# plt.show()

####################################################################################

ib1=np.where(datacritical1[:,0]>0.35)[0][0]
ib2=np.where(datacritical1[:,0]>0.5)[0][0]
ib3=np.where(datacritical1[:,0]>0.73)[0][0]
ib4=np.where(datacritical1[:,0]>0.75)[0][0]
iblist=[ib1,ib2,ib3,ib4]

nrow = 2;ncol=2;
fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex='col',sharey='row',gridspec_kw={'hspace':0.4,'wspace': 0.2})

sns.set_style("ticks")

axs.ravel()[0].set_ylabel('N');axs.ravel()[0].set_xlabel('\n(a)')
axs.ravel()[1].set_xlabel('\n(b)')
axs.ravel()[2].set_ylabel('N')
axs.ravel()[2].set_xlabel('A\n(c)')
axs.ravel()[3].set_xlabel('A\n(d)')

axs.ravel()[0].set_title(r'$\beta<\beta_{c,1}$')
axs.ravel()[1].set_title(r'$\beta>\beta_{c,1}$')
axs.ravel()[3].set_title(r'$\beta>\beta_{c,2}$')
axs.ravel()[2].set_title(r'$\beta<\beta_{c,2}$')

T=50;dt=0.01;
dt_save=0.01;
for i in range(len(iblist)):
    print(i)
    ib=iblist[i]
    b=datacritical1[ib,0]
    p=datacritical1[ib,1]
    n=datacritical1[ib,2]
    a=datacritical1[ib,3]
    s=datacritical1[ib,4]

    Avec=np.linspace(0,1,200)
    Nvec=np.linspace(0,1,200)
    A, N = np.meshgrid(Avec, Nvec)
    P=np.ones((A.shape[0],A.shape[1]))*p

    # make a direction field plot with quiver
    param=[b,K,R,D,E,F,Kmin]
    dA = aEq(P,N,A,param)
    dN = nEq(P,N,A,param)
    dP = pEq(P,N,A,param)

    #axs.ravel()[i].quiver(A[::20,::20], N[::20,::20], dA[::20,::20], dN[::20,::20],angles='xy',scale_units='xy',scale=2,color='tab:blue')
    axs.ravel()[i].streamplot(A,N,dA, dN, density=1.0, linewidth=1.0, color=(0.1,0.5,1,0.4))
    # plot nullclines
    axs.ravel()[i].contour(A, N, dA, levels=[0], linewidths=1.0, colors='0.2',linestyles='dashed')
    axs.ravel()[i].contour(A, N, dN, levels=[0], linewidths=1.0, colors='0.2',linestyles='dashed')
    axs.ravel()[i].contour(A, N, dP, levels=[0], linewidths=1.0, colors='0.2',linestyles='dashed')
    # axs.ravel()[i].contourf(A, N, np.sign(dP), alpha=0.9)
    axs.ravel()[i].fill_between(Avec, np.ones(len(Avec))-Avec, np.ones(len(Avec)),color='0.2',alpha=1.0)
    axs.ravel()[i].fill_between(Avec, (np.ones(len(Avec))-Avec)*b*b/(1+b*b), np.zeros(len(Avec)),color='tab:orange',alpha=0.7)

    # plot steadystate
    if s>0:
        axs.ravel()[i].plot(a,n,'o',linewidth=3,color='red')
    else:
        axs.ravel()[i].plot(a,n,'o',linewidth=3,color='purple')

    # trajectories
    istate=np.array([p,0.98,0.01])
    trj=trajectories(istate,param,T,dt,dt_save)
    axs.ravel()[i].plot(trj[int(0/10*len(trj[:,0])):,0],trj[int(0/10*len(trj[:,1])):,1],color='k',linewidth='1')

    # points = np.array([trj[:,0], trj[:,1]]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap='viridis')
    # lc.set_array(trj[:,2])
    # lc.set_linewidth(1)
    # line=axs.ravel()[i].add_collection(lc)
    # fig.colorbar(line,ax=axs.ravel(i))
    # axs.ravel()[i].plot(trj[:,0],trj[:,1],color=cm.inferno(trj[:,2]),linewidth='1')

    if i==0:
        istate=np.array([p,0.27,0.72])
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
        istate=np.array([p,0.28,0.71])
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
    if i==1:
        dstate=1*np.array([1e-2,1e-2,1e-2])
        istate=np.array([p,n,a])+dstate
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
    if i==3:
        istate=np.array([p,0.65,0.34])
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
        istate=np.array([p,0.66,0.33])
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')
    if i==2:
        istate=np.array([p,0.7,0.29])
        trj=trajectories(istate,param,T,dt,dt_save)
        axs.ravel()[i].plot(trj[:,0],trj[:,1],color='k',linewidth='1')



#fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
sns.despine()
plt.show()
