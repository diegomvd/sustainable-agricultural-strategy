# -*- coding: utf-8 -*-

"""
LONG TERM EQUILIBRIUMS IN THE 2D LAND USE STRATEGY SPACE (EXPANSION - INTENSIFICATION). 
IDENTIFICATION OF COLLAPSE AND SUSTAINABLE REGIONS.

model variables:
p=human population density
n=natural Land
a=agricultural land

model parameters:

1) b=intensification 
2) K=conversion effort

4) R=recovery rate of degraded land
5) D=degradation rate of natural land

6) E=maximum degradation rate of agricultural land
NB: actual degradation rate = E*b

7) Q=relative importance of natural land to agricultural production
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
import seaborn as sns

#plt.style.use('seaborn')

t0 = time.time()

"""
FUNCTION DECLARATION
"""

def foodProduction(p,n,a,param):
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5]

    d=1-a-n
    if a<0:
        a=0
    if n<0:
        n=0
    if d<0:
        d=0

    effective_land=(n+d*(1-b))
    prod=b*(b*a+Q*(1-b)*effective_land*np.power(a,0.5))

    return prod

###############################################################################
# dynamical equations

def pEq(p,n,a,param):
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5]

    prod=foodProduction(p,n,a,param)

    return p*(1-p/prod)

def nEq(p,n,a,param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5]

    d=1-a-n

    if a<0:
        a=0
    if n<0:
        n=0
    if d<0:
        d=0

    ret = (R*np.power(n,0.5)-b*D*np.power(d,0.5))*np.power(d*n,0.5)-K*p*n
    return ret

def aEq(p,n,a,param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5]

    ret=K*p*n-b*E*a

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

###################################################################################

def simtdyn(state,param,T,dt,bvec,Kvec):

    mat=np.zeros((len(Kvec),len(bvec),))

    parsim=param
    for b in bvec:
        parsim[0]=b;
        for K in Kvec:
            print([b,K])

            parsim[1]=K

            state_old=state
            t=0

            while t<T:
                state_new=solver_RK4(state_old,parsim,dt)

                # test if equilibrium has been reached
                sderiv=(state_new-state_old)/dt
                l2sderiv=np.dot(sderiv,sderiv)
                if l2sderiv<1e-6:
                    # print("Equilibrium reached before T")
                    # print(state_new)
                    break

                # test if collapse has been reached
                diff=state_new-np.array([0,0,0])
                l2diff=np.dot(diff,diff)
                if l2diff<1e-4:
                    print('collapse break')
                    break

                state_old=state_new
                t=t+dt

            # print(state_new)

            #assess kind of equilibrium
            diff=state_new-np.array([0,0,0])
            l2diff=np.dot(diff,diff)
            if l2diff<1e-4:
                mat[np.where(Kvec==K),np.where(bvec==b)]=-1 # collapse
            diff=state_new-np.array([0,1,0])
            l2diff=np.dot(diff,diff)
            if l2diff<1e-5:
                mat[np.where(Kvec==K),np.where(bvec==b)]=1 # collapse nature
    return mat

def simtdyn2(state,param,T,dt):

    res=0

    state_old=state
    t=0

    while t<T:
        state_new=solver_RK4(state_old,param,dt)

        # test if equilibrium has been reached
        sderiv=(state_new-state_old)/dt
        l2sderiv=np.dot(sderiv,sderiv)
        if l2sderiv<1e-6:
            # print("Equilibrium reached before T")
            # print(state_new)
            break

        # test if collapse has been reached
        diff=state_new-np.array([0,0,0])
        l2diff=np.dot(diff,diff)
        if l2diff<1e-4:
            # print('collapse break')
            break

        state_old=state_new
        t=t+dt

    # print(state_new)

    #assess kind of equilibrium
    diff=state_new-np.array([0,0,0])
    l2diff=np.dot(diff,diff)
    if l2diff<1e-4:
        res=1 # collapse
    diff=state_new-np.array([0,1,0])
    l2diff=np.dot(diff,diff)
    if l2diff<1e-5:
        res=-1 # collapse nature

    return res

def save(coord,datafile):

    string=str(coord[0])+" "+str(coord[1])
    datafile.write(string+"\n")

###################################################################################
T=100;dt=0.001
state=np.array([0.01,0.98,0.01])

R=1
D=1
E=1
F=1

filename = "DATA_CBORDER_R"+str(R)+"_D_"+str(D)+"_E_"+str(E)+"_F_"+str(F)+".dat"
datafile = open(filename,"w+")

param=[0,0,R,D,E,F]

bmin=0.1;bmax=1.0;db=0.05
Kmin=1.0;Kmax=5;
K=Kmin

b=bmax
while b>bmin:
    K0=K;K1=Kmax;
    deltaK=K1-K0
    attempts=0;att1=0;att0=0
    while deltaK>0.01:
        K=0.5*(K0+K1)
        param[0]=b;param[1]=K
        res=simtdyn2(state,param,T,dt)
        if res==1:
            att1+=1
            K1=K
        elif res==0:
            att0+=1
            K0=K
        else:
            print("error in res")
        attempts+=1
        deltaK=K1-K0

    if att1==attempts:
        K=K0
    if att0==attempts:
        K=K1

    coord=[K,b]
    print(coord)
    save(coord,datafile)
    b-=db

datafile.close()

# bvec=np.arange(0.1,1.1,0.1)
# Kvec=np.arange(1.0,11.0,1)
# print(len(bvec))
# print(len(Kvec))
#
# res=simtdyn(state,param,T,dt,bvec,Kvec)
#
# b,K=np.meshgrid(bvec,Kvec)
# fig, (ax) = plt.subplots(1)
# origin='lower'
# cp=ax.contourf(K,b,res,cmap='RdBu',origin=origin,alpha=0.9)
# plt.show()
#
# fig, (ax) = plt.subplots(1)
# cp=ax.pcolormesh(K,b,res,cmap='RdBu',shading='goraud')
# plt.show()
