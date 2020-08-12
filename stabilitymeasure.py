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
#import seaborn as sns

#plt.style.use('seaborn')

t0 = time.time()

"""
FUNCTION DECLARATION
"""

def foodProduction(state,param):
    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n
    if a<0:
        a=0
    if d<0:
        a=0
    if n<0:
        a=0

    effective_land=(n+d*(1-b))
    prod=b*(b*a+Q*(1-b)*effective_land*np.power(a,0.5))

    return prod

###############################################################################
# dynamical equations

def pEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    prod=foodProduction(state,param)

    return p*(1-p/prod)

def nEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    d=1-a-n

    if a<0:
        a=0
    if d<0:
        a=0
    if n<0:
        a=0   
    ret = (R*np.power(n,0.5)-b*D*np.power(d,0.5))*np.power(d*n,0.5)-(Kmin+(K-Kmin)*(1-b))*p*n
    return ret

def aEq(state,param):

    p=state[0];n=state[1];a=state[2];
    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    ret=(Kmin+(K-Kmin)*(1-b))*p*n-b*E*a

    return ret

###############################################################################
# runge kutta solver

def solver_RK4(state,param,dt):

    k1p=dt*pEq(state,param);k1n=dt*nEq(state,param);k1a=dt*aEq(state,param);
    k1vec=np.array([k1p,k1n,k1a])
    state1=state+0.5*k1vec

    k2p=dt*pEq(state1,param);k2n=dt*nEq(state1,param);k2a=dt*aEq(state1,param);
    k2vec=np.array([k2p,k2n,k2a])
    state2=state+0.5*k2vec

    k3p=dt*pEq(state2,param);k3n=dt*nEq(state2,param);k3a=dt*aEq(state2,param);
    k3vec=np.array([k3p,k3n,k3a])
    state3=state+k3vec

    k4p=dt*pEq(state3,param);k4n=dt*nEq(state3,param);k4a=dt*aEq(state3,param);
    k4vec=np.array([k4p,k4n,k4a])

    var_state=np.array((k1vec+2*k2vec+2*k3vec+k4vec)/6)

    return state+var_state

###############################################################################
# functions for root finding

def initial_guess(param):

    T=100;dt=0.001;t=0
    state=np.array([0.001,0.98,0.01])
    state_new=np.zeros(3)

    while t<T:

        state_new=solver_RK4(state,param,dt)
        state=state_new

        t=t+dt

    return state

def jacobian(state,param):

    p=state[0];n=state[1];a=state[2]
    dstate=[1e-6,1e-6,1e-6]
    dp=dstate[0];dn=dstate[1];da=dstate[2]

    # states with the added perturbation
    pstate=[p+dp,n,a]
    nstate=[p,n+dn,a]
    astate=[p,n,a+da]

    pert_state=[pstate,nstate,astate]

    Jac=np.zeros((3,3))
    for i in range(3):
        Jac[0,i]=(pEq(pert_state[i],param)-pEq(state,param))/dstate[i]
        Jac[1,i]=(nEq(pert_state[i],param)-nEq(state,param))/dstate[i]
        Jac[2,i]=(aEq(pert_state[i],param)-aEq(state,param))/dstate[i]

    ret = Jac
    return ret

def savearray(data,array):

    array=np.concatenate((array,data),axis=0)
    return array

def fun(state,param):

    return [pEq(state,param), nEq(state,param), aEq(state,param)]

##############################################################################
# get equilibria

def findSSbeta(param,db_save):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    bmin=0.1
    bmax=0.9

    dstate=np.ones(5)*1e-7
    dataarr=np.array([[],[],[],[],[]]).T

    b=bmin
    param=[b,K,R,D,E,Q,Kmin]

    old_fp=initial_guess(param)
    old_stability=1

    b_save=b
    db=0.005
    while b<bmax:

        param=[b,K,R,D,E,Q,Kmin]

        state0=old_fp

        # solve the root problem and get the fixed point
        sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

        fixed_point = sol.x # steady states

        # test if the root finder actually converged to a fp
        btest=np.array(fun(fixed_point,param))
        l2test=np.sqrt(np.dot(btest,btest))
        if(l2test>1e-6):
            print("root finder didn't converge")
            break
        old_fp=np.array(fixed_point)

        # get the eigenvalues of the jacobian matrix to calculate stability
        jac=jacobian(fixed_point,param)
        la, v =linalg.eig(jac)
        stability=0
        if all(la.real<0):
            stability=1

        old_stability=stability

        if b>=b_save:
            b_save+=db_save
            fixed_point_save=np.append(fixed_point,stability)
            data=np.array([np.append(b,fixed_point_save)])
            dataarr=savearray(data,dataarr)

        b=b+db

    ret=dataarr
    return ret

##############################################################################
# choose random points in the phase plane

def normal2D(equilibria,std,Np):
    equilibrium=equilibria[1:] # this excludes population
    var=std*std
    cov=[[var,0],[0,var]]
    points=np.array([[],[],[]]).T
    i=0
    while i<Np:
        point = np.random.multivariate_normal(equilibrium,cov)
        if np.sum(point)<=1 and point[0]>=0 and point[1]>=0:
            point_new=np.array([np.append(equilibria[0],point)])
            points=savearray(point_new,points)
            i+=1

    ret=points
    return ret

def normal3D(equilibria,std,Np):
    return 0

##############################################################################
# simulation

def sim(param,initial_state,dt_save,T,dt):

    t=0
    state=initial_state
    state_new=np.zeros(3)
    t_save_state=dt_save

    dataarray=np.array([[],[],[],[]]).T

    while t<T:

        if t>=t_save_state:
            t_save_state+=dt_save
            data=np.array([np.append(t,state)])
            dataarray=savearray(data,dataarray)

        state_new=solver_RK4(state,param,dt)

        deriv=(state_new-state)/dt
        if np.sqrt(np.dot(deriv,deriv))<1e-7:# steadystate reached
            data=np.array([np.append(t,state_new)])
            dataarray=savearray(data,dataarray)
            break

        state=state_new

        t=t+dt

    ret=state
    return ret

def sim2(param,initial_state,dt_save,T,dt):

    t=0
    state=initial_state
    state_new=np.zeros(3)
    t_save_state=dt_save

    dataarray=np.array([[],[],[],[]]).T

    while t<T:

        if t>=t_save_state:
            t_save_state+=dt_save
            data=np.array([np.append(t,state)])
            dataarray=savearray(data,dataarray)

        state_new=solver_RK4(state,param,dt)

        deriv=(state_new-state)/dt
        if np.sqrt(np.dot(deriv,deriv))<1e-7:# steadystate reached
            data=np.array([np.append(t,state_new)])
            dataarray=savearray(data,dataarray)
            break

        state=state_new

        t=t+dt

    ret=dataarray
    return ret

def safetytest(equilibria,state):
    ret=0
    l2test=np.sqrt(np.dot(equilibria-state,equilibria-state))
    if(l2test<1e-6):
        ret=1

    return ret

def minimumDistance(equilibria,statearray):
    ret=0
    if statearray.size!=0:
        mindist=np.sqrt(np.dot(equilibria-statearray[0],equilibria-statearray[0]))
        for state in statearray:
            euclidiandistance=np.sqrt(np.dot(equilibria-state,equilibria-state))
            if euclidiandistance<mindist:
                mindist=euclidiandistance
        ret=mindist
    return ret

###############################################################################
def plot_stability_metrics(Psafe_array,euc_array,param):

    sns.set_style("ticks")

    beta_psafe=Psafe_array[:,0]; Psafe=Psafe_array[:,1]
    beta_euc=euc_array[:,0]; euc=euc_array[:,1]

    fig, (ax1,ax2) = plt.subplots(2,1,sharex='col')

    ax1.plot(beta_psafe,Psafe,'o',color='tab:blue',linewidth=2,alpha=0.7)
    ax2.plot(beta_euc,euc,'o',color='tab:orange',linewidth=2,alpha=0.7)
    ax2.set(xlabel=r'$\beta$');
    ax1.set(ylabel=r'$\mathcal{P}_{safe}$');
    ax2.set(ylabel=r'$\mathcal{D}$');

    sns.despine()
    plt.show()

    # fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey='row')
    #
    # ax1.plot(beta_psafe,Psafe,'o',color='tab:blue',linewidth=2,alpha=0.7)
    # ax2.plot(beta_euc,euc,'o',color='tab:orange',linewidth=2,alpha=0.7)
    # ax2.set(xlabel=r'$\beta$');
    # ax1.set(ylabel=r'$\mathcal{P}_{safe}$');
    # ax2.set(ylabel=r'$\mathcal{D}$');
    #
    # sns.despine()
    # plt.show()


###############################################################################
# main program

# finding equilibriums in function of b
R=1.0;D=1.0;E=1.0;b=0;K=4.5;Kmin=0.5;Q=1.0
param=[b,K,R,D,E,Q,Kmin]
db_save=0.05

equilibria=findSSbeta(param,db_save)

###############################################################################
# analysis on A,N plane. P set to equilibrium value

# iterating over equilibria
Psafe_array=np.array([[],[]]).T
euc_array=np.array([[],[]]).T
counteq=0

# euc=np.array([],[],[])
# safelist=[[],[],[]]
# unsafelist=[[],[],[]]
# i=0
filename1="PSAFE"+"_b_"+str(b)+"_r_"+str(R)+".dat"
filename2="DEUC"+"_b_"+str(b)+"_r_"+str(R)+".dat"
file1 = open(filename1,"w+");file2=open(filename2,"w+")
for member in equilibria:
    counteq+=1
    # safe_array=np.array([[],[],[]]).T
    unsafe_array=np.array([[],[],[]]).T
    if member[-1]>0:
        safecount=0
        equilibrium=member[1:-1]
        beta=member[0]
        param[0]=beta
        # randomly choosing Np points in phase space
        Np=1000;std=0.5
        pert_states=normal2D(equilibrium,std,Np)

        dt_save=200
        T=dt_save
        dt=0.001
        countpoint=0
        for pert_state in pert_states:
            countpoint+=1
            print("now in eq number "+str(counteq)+", point "+str(countpoint)+ " out of "+str(Np))
            final_state=sim(param,pert_state,dt_save,T,dt)
            safety=safetytest(equilibrium,final_state)
            if safety==1:
                #safe_array=savearray([pert_state],safe_array)
                safecount=safecount+1
            else:
                unsafe_array=savearray([pert_state],unsafe_array)
    else:
        safecount=0

    Psafe=safecount/Np
    # safe_array=savearray(np.array([np.append(beta,Psafe)]),Psafe_array)
    data1=str(beta)+' '+str(Psafe);file1.write(data1+"\n")

    euc_mindist=minimumDistance(equilibrium,unsafe_array)
    # euc_array=savearray(np.array([np.append(beta,euc_mindist)]),euc_array)
    data2=str(beta)+' '+str(euc_mindist);file2.write(data2+"\n")

file1.close();file2.close()

print(str(time.time()-t0))


#plot_stability_metrics(Psafe_array,euc_array,param)
