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

###############################################################################
# functions for root finding

def initial_guess(param):

    T=100;dt=0.01;t=0
    state=np.array([0.01,0.98,0.01])
    state_new=np.zeros(3)

    while t<T:

        state_new=solver_RK4(state,param,dt)
        state=state_new

        t=t+dt

    return state

def jacobian(state,param):

    p=state[0];n=state[1];d=state[2]
    dstate=[1e-6,1e-6,1e-6]
    dp=dstate[0];dn=dstate[1];dd=dstate[2]

    # states with the added perturbation
    pstate=[p+dp,n,d]
    nstate=[p,n+dn,d]
    astate=[p,n,d+dd]

    pert_state=[pstate,nstate,dstate]

    Jac=np.zeros((3,3))
    for i in range(3):
        Jac[0,i]=(popEq(pert_state[i],param)-popEq(state,param))/dstate[i]
        Jac[1,i]=(l0Eq(pert_state[i],param)-l0Eq(state,param))/dstate[i]
        Jac[2,i]=(l1Eq(pert_state[i],param)-l1Eq(state,param))/dstate[i]

    ret = Jac
    return ret

def savearray(data,array):

    array=np.concatenate((array,data),axis=0)
    return array

def fun(state,param):

    return [popEq(state,param), l0Eq(state,param), l1Eq(state,param)]

def findcriticalbeta(param):

    b=param[0];K=param[1];R=param[2];D=param[3];E=param[4];Q=param[5];Kmin=param[6]

    bmin=0.1
    bmax=0.9

    b=bmin
    param=[b,K,R,D,E,Q,Kmin]

    old_fp=initial_guess(param)
    old_stability=1

    db=0.005
    while b<bmax:
        print("b="+str(b))

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
            print(b)
            break
        old_fp=np.array(fixed_point)

        # get the eigenvalues of the analytical_jacobian matrix to calculate stability
        jac=jacobian(fixed_point,param)
        la, v =linalg.eig(jac)
        stability=0
        if all(la.real<0):
            stability=1

        if stability!=old_stability:
            print("stability change")
            print(b)
            break

        b=b+db

    bc1=b
    bc2=b

    if bc1<bmax-2*db:

        b=bmax
        param=[b,K,R,D,E,Q,Kmin]

        old_fp=initial_guess(param)
        old_stability=1

        db=0.005
        while b>bmin:
            print("b="+str(b))

            param=[b,K,R,D,E,Q,Kmin]

            state0=old_fp

            # solve the root problem and get the fixed point
            sol = optimize.root(fun,state0,args=(param,),method='lm',jac=jacobian,options={'xtol': 1.49012e-16,'ftol': 1.49012e-16,'gtol': 0.0})

            fixed_point = sol.x # steady states

            # test if the root finder actually converged to a fp
            btest=np.array(fun(fixed_point,param))
            l2test=np.sqrt(np.dot(btest,btest))
            if(l2test>1e-6):
                # print("root finder didn't converge")
                break
            old_fp=np.array(fixed_point)

            # get the eigenvalues of the analytical_jacobian matrix to calculate stability
            jac=jacobian(fixed_point,param)
            la, v =linalg.eig(jac)
            stability=0
            if all(la.real<0):
                stability=1

            if stability!=old_stability:
                # print("stability change")
                break

            b=b-db

        bc2=b

    deltab=bc2-bc1
    return deltab

def Plot(datafull):
    sns.set_context('paper')
    style.use('seaborn-paper')
    fig, (ax1) = plt.subplots(1,1)
    sns.set_style("ticks")

    data1=datafull[0]
    data2=datafull[1]
    data3=datafull[2]

    R1=data1[:,0]
    deltab1=data1[:,1]
    R2=data2[:,0]
    deltab2=data2[:,1]
    R3=data3[:,0]
    deltab3=data3[:,1]

    ax1.plot(R1,deltab1,'o',linewidth=2,alpha=0.7,label='D=0.5')
    ax1.plot(R2,deltab2,'o',linewidth=2,alpha=0.7,label='D=1.0')
    ax1.plot(R3,deltab3,'o',linewidth=2,alpha=0.7,label='D=1.5')

    fig.legend()
    sns.despine()
    ax1.set(ylabel=r"$\Delta \beta$")
    ax1.set(xlabel=r'$r$');
    plt.show()
    plt.close()

###############################################################################

R=1.0;D=1.0;E=1.0;b=0;K=4.5;Kmin=0.5;Q=1.0


Rmin=0.5; Rmax=1.5
Rarray=np.linspace(Rmin,Rmax,10)
Darray=np.array([0.5,1.0,1.5])

res1=np.array([[],[]]).T

D=1.5
print(D)
for R in Rarray:
    print(R)
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    data=np.array([np.append(R,deltab)])
    res1=savearray(data,res1)

res2=np.array([[],[]]).T

D=0.75
print(D)
for R in Rarray:
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    data=np.array([np.append(R,deltab)])
    res2=savearray(data,res2)

res3=np.array([[],[]]).T

D=1.0
print(D)
for R in Rarray:
    param=[b,K,R,D,E,Q,Kmin]
    deltab=findcriticalbeta(param)
    data=np.array([np.append(R,deltab)])
    res3=savearray(data,res3)

res=[res1,res2,res3]
Plot(res)
