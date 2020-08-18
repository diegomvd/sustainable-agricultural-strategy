# -*- coding: utf-8 -*-

"""
TEMPORAL DYNAMICS (plots for 3 different values of the land use strategy)

model variables:
pop=human population density
l0=natural Land
l1=degraded land

model parameters:

1) b=land use strategy (0->extensive agriculture; 1->intensive agriculture)
2) K=maximum land conversion effort
3) Kmin=minimum land conversion effort
NB: actual conversion effort = Kmin+(K-Kmin)*(1-b)

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
import matplotlib.style as style
import seaborn as sns

#plt.style.use('seaborn')

t0 = time.time()

"""
FUNCTION DECLARATION
"""
def foodProduction(state,param):
    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    effective_land=(l0+l1*(1-b))
    a=1-l0-l1
    prod=b*(b*a+Q*(1-b)*effective_land*np.power(a,0.5))
    # prod=b*(b*a+Q*(1-b)*l0*np.power(a,0.5))

    return prod

def popEq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    prod=foodProduction(state,param)

    return pop*(1-pop/prod)

def l0Eq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    if l0<0:
        l0=0

    ret = (R*np.power(l0,0.5)-b*D*np.power(l1,0.5))*np.power(l1*l0,0.5)-(Kmin+(K-Kmin)*(1-b))*pop*l0
    # ret = (R*np.power(l0,0.5)-b*D*np.power((1-l0),0.5))*np.power((1-l0)*l0,0.5)-(Kmin+(K-Kmin)*(1-b))*pop*l0
    return ret

def l1Eq(state,param):

    pop=state[0];l0=state[1];l1=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    if l0<0:
        l0=0

    a=1-l0-l1
    # ret= -(R*np.power(l0,0.5)-b*D*np.power((1-l0),0.5))*np.power((1-l0)*l0,0.5)+b*E*a
    ret= -(R*np.power(l0,0.5)-b*D*np.power(l1,0.5))*np.power(l1*l0,0.5)+E*a*b
    return ret

def solver_RK4(state,param,dt):

    k1p=dt*popEq(state,param);k1l0=dt*l0Eq(state,param);k1l1=dt*l1Eq(state,param);
    k1vec=np.array([k1p,k1l0,k1l1])
    state1=state+0.5*k1vec

    k2p=dt*popEq(state1,param);k2l0=dt*l0Eq(state1,param);k2l1=dt*l1Eq(state1,param);
    k2vec=np.array([k2p,k2l0,k2l1])
    state2=state+0.5*k2vec

    k3p=dt*popEq(state2,param);k3l0=dt*l0Eq(state2,param);k3l1=dt*l1Eq(state2,param);
    k3vec=np.array([k3p,k3l0,k3l1])
    state3=state+k3vec

    k4p=dt*popEq(state3,param);k4l0=dt*l0Eq(state3,param);k4l1=dt*l1Eq(state3,param);
    k4vec=np.array([k4p,k4l0,k4l1])

    var_state=np.array((k1vec+2*k2vec+2*k3vec+k4vec)/6)

    return state+var_state

def sim(param,initial_state,T,dt,datafile,dt_save_state):

    t=0
    state=initial_state
    state_new=np.zeros(3)
    t_save_state=0

    while t<T:

        if t>=t_save_state:
            t_save_state+=dt_save_state
            prodpp=foodProduction(state,param)
            save(state,prodpp,t,datafile)

        # param[0]=0.1+0.7/(1+np.exp(-(t-T/4)*0.02))
        print("t="+str(t)+", b="+str(param[0]))
        state_new=solver_RK4(state,param,dt)

        # deriv=(state_new-state)/dt
        # if np.sqrt(np.dot(deriv,deriv))<1e-6:# steadystate reached
        #     break

        state=state_new

        t=t+dt

    datafile.close()
    return None

def save(state,prod,t,datafile):

    string=str(t)
    for i in state:
        string=string+" "+str(i)

    string=string+" "+str(prod)
    datafile.write(string+"\n")

    return None

def plot(filename1,filename2,filename3):

    sns.set_context('paper')
    style.use('seaborn-paper')

    data2plot1=np.loadtxt(filename1)
    data2plot2=np.loadtxt(filename2)
    data2plot3=np.loadtxt(filename3)

    sns.set_style("ticks")

    fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2,3,sharex='col',sharey='row',gridspec_kw={'hspace':0.01,'wspace': 0.1},figsize=(7,5.25))
################################################################################

    time=data2plot1[:,0]
    population=data2plot1[:,1]
    l0=data2plot1[:,2]
    l1=data2plot1[:,3]
    a=np.ones(len(data2plot1[:,0]))-l0-l1

    ax1.plot(time,population,color='tab:blue',linewidth=2,alpha=0.7)

    ax2.plot(time,1*l0,linewidth=2,color='tab:green',label="N",alpha=0.7)
    ax2.plot(time,1*a,linewidth=2,color='tab:orange',label="A",alpha=0.7)
    ax2.plot(time,1*l1,linewidth=2,color='tab:red',label="D",alpha=0.7)

    #ax1.grid(b=True);ax2.grid(b=True);ax3.grid(b=True);ax4.grid(b=True)
    ax1.set(ylabel="Population");ax2.set(ylabel=r"Land");
    ax1.set_title(r"$\beta=0.2$")

################################################################################

    time=data2plot2[:,0]
    population=data2plot2[:,1]
    l0=data2plot2[:,2]
    l1=data2plot2[:,3]
    a=np.ones(len(data2plot2[:,0]))-l0-l1

    ax3.plot(time,population,color='tab:blue',linewidth=2,alpha=0.7)
    ax3.set_title(r"$\beta=0.5$")

    ax4.plot(time,1*l0,linewidth=2,color='tab:green',alpha=0.7)
    ax4.plot(time,1*a,linewidth=2,color='tab:orange',alpha=0.7)
    ax4.plot(time,1*l1,linewidth=2,color='tab:red',alpha=0.7)


################################################################################

    time=data2plot3[:,0]
    population=data2plot3[:,1]
    l0=data2plot3[:,2]
    l1=data2plot3[:,3]
    a=np.ones(len(data2plot3[:,0]))-l0-l1

    ax5.plot(time,population,color='tab:blue',linewidth=2,alpha=0.7)
    ax5.set_title(r"$\beta=0.8$")

    ax6.plot(time,1*l0,linewidth=2,color='tab:green',alpha=0.7)
    ax6.plot(time,1*a,linewidth=2,color='tab:orange',alpha=0.7)
    ax6.plot(time,1*l1,linewidth=2,color='tab:red',alpha=0.7)


    ax2.set(xlabel="Time");ax4.set(xlabel="Time");ax6.set(xlabel="Time");

    # fig.legend(loc='upper right', bbox_to_anchor=(1.0, 0.5))
    fig.legend()
    sns.despine()
    plt.savefig("dynamics.jpg",dpi=1000)
    plt.show()
    plt.savefig('testremote.png')
    plt.close()

"""
PARAMETER & VARIABLE DECLARATION
"""

T=100;dt=0.01;
dt_save_state=0.1;

bdatapoints=5
bmin=0.05
bmax=0.9
barray=np.linspace(bmin,bmax,bdatapoints)
# barray=np.array([0.727199,0.727202,0.727201])
# barray=np.array([0.4106108,0.4106109,0.41061099])
# 1.20e-01 3.45e-01 3.36e-01
# 1.22e-01 3.43e-01 3.03e-01
# 1.21e-01 3.44e-01 2.55e-01

barray=np.array([0.2,0.5,0.8])

R=1.0 #1.0
D=1.0 #1.0
E=1.0 #1.0
b=bmin
K=4.5 #4.5
Kmin=0.5 #0.5
Q=1.0
param=[b,K,R,D,E,Q,Kmin]
initial_state=np.array([0.001, 0.98, 0.01])
# initial_state=np.array([1.20e-01, 3.45e-01, 3.36e-01])

filename=['f1','f2','f3']
i=0
for b in barray:

    print(b)

    filename[i] = "DATA_2LEVEL_T_"+str(T)+"_dt_"+str(dt)+"_b_"+str(b)+"_K_"+str(K)+"_R_"+str(R)+"_D_"+str(D)+"_E_"+str(E)+".dat"
    datafile = open(filename[i],"w+")


    param[0]=b
    # param[2]=R

    sim(param,initial_state,T,dt,datafile,dt_save_state)

    i=i+1

print(str(time.time()-t0))
plot(filename[0],filename[1],filename[2])
